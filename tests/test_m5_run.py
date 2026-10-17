"""Unit tests for the run log of ``playground/m5_prototype/m5_run.py``.

Every run appends one row to ``output/runs.csv`` (issue #104), and the Kaggle scores of a submitted run are
typed into its row by hand. These tests check that appending keeps such hand-typed values verbatim and that
a column added to the row later does not break an existing log.

The runner is a playground script, not part of the package, so it is loaded by path as
``tests/test_m5_models.py`` loads the model registry.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

pytest.importorskip("optax", reason="needs the optional 'm5' extra: pip install 'bunobee[m5]'")

M5_DIR = Path(__file__).resolve().parents[1] / "playground" / "m5_prototype"


def _load_script(name: str) -> ModuleType:
    """Import a playground M5 script by path and register it under its own name.

    Registering matters for the scripts ``m5_run`` imports by name, which only resolve from ``tests/`` once
    they are in ``sys.modules``.

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


for _name in ("m5_common", "m5_models", "m5_score_submission"):
    _load_script(_name)
m5_run = _load_script("m5_run")


def _row(**overrides: object) -> dict[str, object]:
    """A run-log row with the columns of an ``m5_run.py`` run, with ``overrides`` applied."""
    row: dict[str, object] = {
        "timestamp": "2026-10-16T20:00:00",
        "run_dir": "20261016_200000",
        "model": "ssp_dow",
        "settings": '{"n_iter": 100, "lr": 0.03}',
        "n_series": 3000,
        "origins": "validation",
        "wrmsse_validation": 0.93487,
        "wrmsse_evaluation": None,
        "git_commit": "1234e56",
        "git_dirty": False,
        "kaggle_public": None,
    }
    return {**row, **overrides}


class TestRunLog:
    def test_first_run_creates_the_log(self, tmp_path):
        path = tmp_path / "runs.csv"

        m5_run._append_run_log(path, _row())

        log = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert list(log.columns) == list(_row())
        assert log.loc[0, "settings"] == '{"n_iter": 100, "lr": 0.03}'
        assert log.loc[0, "wrmsse_evaluation"] == ""
        assert log.loc[0, "git_dirty"] == "False"

    def test_appending_keeps_hand_typed_values_verbatim(self, tmp_path):
        path = tmp_path / "runs.csv"
        m5_run._append_run_log(path, _row())
        log = pd.read_csv(path, dtype=str, keep_default_na=False)
        log.loc[0, "kaggle_public"] = "1.08127"
        log.to_csv(path, index=False)

        m5_run._append_run_log(path, _row(run_dir="20261016_210000", git_commit="abc1234"))

        log = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert list(log["run_dir"]) == ["20261016_200000", "20261016_210000"]
        assert list(log["kaggle_public"]) == ["1.08127", ""]
        # A hash that reads as a number in scientific notation stays text.
        assert list(log["git_commit"]) == ["1234e56", "abc1234"]

    def test_a_new_column_is_blank_for_earlier_runs(self, tmp_path):
        path = tmp_path / "runs.csv"
        m5_run._append_run_log(path, _row())

        m5_run._append_run_log(path, _row(peak_rss_gib=3.5))

        log = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert list(log["peak_rss_gib"]) == ["", "3.5"]
