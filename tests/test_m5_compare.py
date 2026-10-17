"""Unit tests for the run comparison, ``playground/m5_prototype/m5_compare.py``.

A sampled ``m5_run.py`` run aggregates and weights only its own series, so ``m5_compare.py`` must refuse two
runs fitted on different series or with no scored window in common (issue #104). These tests cover those
checks, and that the differences it reports are ``RUN_B - RUN_A`` for each shared window.

Every test compares run directories written by :func:`_write_run`, holding only the two files the script
reads, so none needs the raw M5 data. The script is a playground script, not part of the package, so it is
loaded by path as ``tests/test_m5_score.py`` loads the scorer.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

M5_DIR = Path(__file__).resolve().parents[1] / "playground" / "m5_prototype"


def _load_script(name: str) -> ModuleType:
    """Import a playground M5 script by path and register it under its own name.

    Registering matters for the scripts imported by name: ``m5_compare`` imports ``m5_score_submission``,
    which imports ``m5_common``, and neither resolves from ``tests/`` until it is in ``sys.modules``.

    Parameters
    ----------
    name : str
        Module name, i.e. the file stem under ``playground/m5_prototype/``.

    Returns
    -------
    ModuleType
        The imported module.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, M5_DIR / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover - only on a broken checkout
        raise ImportError(f"cannot load {name} from {M5_DIR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


_load_script("m5_common")
_load_script("m5_score_submission")
m5_compare = _load_script("m5_compare")

LEVELS = np.arange(1, 13)
FIT_IDX = np.array([7, 2, 11, 5])


def _write_run(
    run_dir: Path, fit_idx: np.ndarray = FIT_IDX, windows: tuple[str, ...] = ("validation",), shift: float = 0.0
) -> Path:
    """Write the two files ``m5_compare.py`` reads into a fake run directory.

    Level ``L`` of a window scores ``WRMSSE = L / 10 + shift`` (``+ 1`` on the evaluation window) with a bias of
    ``-L + 10 * shift`` percent, so the differences between two runs are ``shift_b - shift_a`` and ten times that.

    Parameters
    ----------
    run_dir : Path
        Directory to create.
    fit_idx : np.ndarray, optional
        The fitted series, saved to ``batch_params.npz``. Default :data:`FIT_IDX`.
    windows : tuple of str, optional
        Windows in ``wrmsse_per_level.csv``. Default validation only.
    shift : float, optional
        Added to every level's WRMSSE. Default 0.

    Returns
    -------
    Path
        ``run_dir``.
    """
    run_dir.mkdir()
    np.savez(run_dir / "batch_params.npz", forecasts=np.zeros((fit_idx.size, 28)), fit_idx=fit_idx)
    tables = [
        pd.DataFrame(
            {
                "window": window,
                "level": LEVELS,
                "name": [f"level {level}" for level in LEVELS],
                "n_series": 1,
                "wrmsse": LEVELS / 10 + shift + (window == "evaluation"),
                "bias_pct": -LEVELS + 10 * shift,
            }
        )
        for window in windows
    ]
    pd.concat(tables, ignore_index=True).to_csv(run_dir / "wrmsse_per_level.csv", index=False)
    return run_dir


class TestSameSeries:
    """Runs are comparable only when they fitted the same set of series."""

    def test_same_series_in_another_order_compare(self, tmp_path):
        run_a = _write_run(tmp_path / "a")
        run_b = _write_run(tmp_path / "b", fit_idx=FIT_IDX[::-1])

        table = m5_compare.compare_runs(run_a, run_b)

        assert len(table) == 12

    def test_different_series_are_refused(self, tmp_path):
        run_a = _write_run(tmp_path / "a")
        run_b = _write_run(tmp_path / "b", fit_idx=np.array([7, 2, 11, 6]))

        with pytest.raises(ValueError, match=r"different series \(4 and 4 series, 3 in common\)"):
            m5_compare.compare_runs(run_a, run_b)

    def test_a_subset_of_the_series_is_refused(self, tmp_path):
        run_a = _write_run(tmp_path / "a")
        run_b = _write_run(tmp_path / "b", fit_idx=FIT_IDX[:3])

        with pytest.raises(ValueError, match=r"different series \(4 and 3 series, 3 in common\)"):
            m5_compare.compare_runs(run_a, run_b)


class TestSharedWindows:
    """Only the windows both runs scored are compared, and at least one must be."""

    def test_no_shared_window_is_refused(self, tmp_path):
        run_a = _write_run(tmp_path / "a", windows=("validation",))
        run_b = _write_run(tmp_path / "b", windows=("evaluation",))

        with pytest.raises(ValueError, match="share no scored window"):
            m5_compare.compare_runs(run_a, run_b)

    def test_only_shared_windows_are_compared(self, tmp_path):
        run_a = _write_run(tmp_path / "a", windows=("validation", "evaluation"))
        run_b = _write_run(tmp_path / "b", windows=("validation",))

        table = m5_compare.compare_runs(run_a, run_b)

        assert set(table["window"]) == {"validation"}

    def test_shared_windows_follow_the_windows_order(self, tmp_path):
        a = pd.DataFrame({"window": ["evaluation", "validation"]})
        b = pd.DataFrame({"window": ["validation", "evaluation"]})

        assert m5_compare.shared_windows(a, b) == ["validation", "evaluation"]


class TestDeltas:
    """Differences are ``RUN_B - RUN_A``, level by level, for each shared window."""

    def test_deltas_are_b_minus_a(self, tmp_path):
        run_a = _write_run(tmp_path / "a", windows=("validation", "evaluation"), shift=0.25)
        run_b = _write_run(tmp_path / "b", windows=("validation", "evaluation"), shift=0.125)

        table = m5_compare.compare_runs(run_a, run_b)

        assert list(table["window"]) == ["validation"] * 12 + ["evaluation"] * 12
        assert list(table["level"]) == [*LEVELS, *LEVELS]
        np.testing.assert_allclose(table["wrmsse_delta"], -0.125)
        np.testing.assert_allclose(table["bias_pct_delta"], -1.25)
        evaluation = table[table["window"] == "evaluation"]
        np.testing.assert_allclose(evaluation["wrmsse_a"], LEVELS / 10 + 1.25)
        np.testing.assert_allclose(evaluation["wrmsse_b"], LEVELS / 10 + 1.125)


class TestCli:
    """``main`` turns a refused or missing run into an exit with the reason, not a traceback."""

    def test_a_directory_that_is_not_a_run_exits_with_the_reason(self, tmp_path, monkeypatch):
        run_a = _write_run(tmp_path / "a")
        (tmp_path / "b").mkdir()
        monkeypatch.setattr(sys, "argv", ["m5_compare.py", str(run_a), str(tmp_path / "b")])

        with pytest.raises(SystemExit, match="batch_params.npz not found"):
            m5_compare.main()

    def test_refused_comparison_exits_with_the_reason(self, tmp_path, monkeypatch):
        run_a = _write_run(tmp_path / "a")
        run_b = _write_run(tmp_path / "b", fit_idx=FIT_IDX + 1)
        monkeypatch.setattr(sys, "argv", ["m5_compare.py", str(run_a), str(run_b)])

        with pytest.raises(SystemExit, match="different series"):
            m5_compare.main()

    def test_comparison_logs_the_hierarchical_delta(self, tmp_path, monkeypatch, caplog):
        run_a = _write_run(tmp_path / "a", shift=0.25)
        run_b = _write_run(tmp_path / "b", shift=0.125)
        monkeypatch.setattr(sys, "argv", ["m5_compare.py", str(run_a), str(run_b)])

        with caplog.at_level("INFO", logger=m5_compare.logger.name):
            m5_compare.main()

        assert "A 0.900000  B 0.775000  delta -0.125000" in caplog.text
