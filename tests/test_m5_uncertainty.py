"""Tests for the M5 Uncertainty scorer, writer, baseline and fetch (issue #157).

One test class per acceptance criterion:

1. The WSPL scorer is checked by hand on a small hierarchy, for the full hierarchy and for a set of bottom series.
2. The baseline writes a submission that matches the sample submission's ids and row order (a small stand-in for the
   771,120-row file here, which CI does not have; the full-size run is in the PR), and its local WSPL is scored.
3. ``m5_fetch.py`` downloads the uncertainty ``sample_submission.csv`` into its own folder and unzips it, leaving the
   accuracy folder's file alone.

The scripts are playground scripts, not part of the package, so they are loaded by path as ``tests/test_m5_score.py``
loads the accuracy scorer.

The synthetic hierarchy is the accuracy tests': one state, category and department, two items in two stores, so four
bottom series. Each sells nothing before day :data:`LAUNCH` and then alternates 1, 0, 1, 0, ..., so a group of ``n``
series sells ``n`` times that pattern and its scale (mean absolute one-step difference) is ``n``. Series 0 sells at 4.0
and the others at 1.0. Every group is forecast at its actuals (all quantiles) except those holding series 0, which are
forecast 1 too high: a pinball loss of ``1 - u`` per day, ``0.5`` on average over the quantiles (their mean is 0.5), so
that group's scaled loss is ``0.5 / n`` and each level's WSPL is that times the group's dollar weight.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import xarray as xr

M5_DIR = Path(__file__).resolve().parents[1] / "playground" / "m5_prototype"


def _load_script(name: str) -> ModuleType:
    """Import a playground M5 script by path and register it under its own name.

    Registering matters for the scripts the others import by name, which do not resolve from ``tests/`` until
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


_load_script("m5_common")
_load_script("m5_score_submission")
score = _load_script("m5u_score")
submission = _load_script("m5u_submission")
baseline = _load_script("m5u_baseline")
fetch = _load_script("m5_fetch")

# Calendar: 40 train days, then one 28-day validation and one 28-day test window, as in the M5 splits.
N_TRAIN = 40
N_STEPS = 28
N_DATES = N_TRAIN + 2 * N_STEPS
WINDOW_DAYS = {
    "prevalidation": slice(N_TRAIN - N_STEPS, N_TRAIN),
    "validation": slice(N_TRAIN, N_TRAIN + N_STEPS),
    "evaluation": slice(N_TRAIN + N_STEPS, N_DATES),
}
LAUNCH = 7  # first day with a sale
TEST_SHIFT = 5.0  # added to the test-window actuals, so scoring the wrong window's actuals cannot pass by accident
PRICE_0 = 4.0

SERIES_IDS = ["FOODS_1_001_CA_1", "FOODS_1_001_CA_2", "FOODS_1_002_CA_1", "FOODS_1_002_CA_2"]
N_SERIES = len(SERIES_IDS)
HIERARCHY = {
    "item_id": ["FOODS_1_001", "FOODS_1_001", "FOODS_1_002", "FOODS_1_002"],
    "dept_id": ["FOODS_1"] * N_SERIES,
    "cat_id": ["FOODS"] * N_SERIES,
    "store_id": ["CA_1", "CA_2", "CA_1", "CA_2"],
    "state_id": ["CA"] * N_SERIES,
}
N_GROUPS = [1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 4]  # per level
N_LEVELS = len(N_GROUPS)

#: Series 0 is forecast off, so every group holding it carries the loss. Its size per level, for the full hierarchy:
#: the single group of all four at levels 1, 2 and 4-7, a pair at levels 3, 8, 9 (store CA_1) and 10, 11 (item 001),
#: and alone at level 12. Dollar weights: 56 for series 0, 14 for each other (last 28 train days, 14 sales each).
SIZE_OF_SERIES_0 = [4, 4, 2, 4, 4, 4, 4, 2, 2, 2, 2, 1]
WEIGHT_OF_SERIES_0 = {4: 1.0, 2: 70 / 98, 1: 56 / 98}
EXPECTED_FULL = np.array([WEIGHT_OF_SERIES_0[n] * 0.5 / n for n in SIZE_OF_SERIES_0])

#: The same for the two bottom series of item 001 (series 0 and 1) alone.
SIZE_OF_SERIES_0_PAIR = [2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1]
WEIGHT_OF_SERIES_0_PAIR = {2: 1.0, 1: 56 / 70}
EXPECTED_PAIR = np.array([WEIGHT_OF_SERIES_0_PAIR[n] * 0.5 / n for n in SIZE_OF_SERIES_0_PAIR])


def _make_ds(test_actuals: bool = True) -> xr.Dataset:
    """Build the synthetic four-series hierarchy in the layout of ``m5_preprocess.build_m5_dataset``.

    Parameters
    ----------
    test_actuals : bool, optional
        If False, the test window's sales are NaN, as in a dataset built from the Kaggle files alone. Default True.

    Returns
    -------
    xr.Dataset
        ``sales`` and ``sell_price`` on ``(series_id, date)``, the hierarchy keys on ``series_id`` and ``split`` on
        ``date``.
    """
    t = np.arange(N_DATES)
    sales = np.tile(np.where(t >= LAUNCH, t % 2, 0).astype(float), (N_SERIES, 1))
    if test_actuals:
        sales[:, WINDOW_DAYS["evaluation"]] += TEST_SHIFT
    else:
        sales[:, WINDOW_DAYS["evaluation"]] = np.nan
    prices = np.ones((N_SERIES, N_DATES))
    prices[0] = PRICE_0
    split = np.array(["train"] * N_TRAIN + ["validation"] * N_STEPS + ["test"] * N_STEPS)
    return xr.Dataset(
        {"sales": (("series_id", "date"), sales), "sell_price": (("series_id", "date"), prices)},
        coords={
            "series_id": SERIES_IDS,
            "date": t,
            **{key: ("series_id", values) for key, values in HIERARCHY.items()},
            "split": ("date", split),
        },
    )


def _forecasts_at_actuals(ds: xr.Dataset, hierarchy, window: str, offset_series_0: float = 1.0) -> list[np.ndarray]:
    """Per-level quantile forecasts equal to the actuals, plus ``offset_series_0`` on groups holding series 0.

    The actuals are summed with plain loops over the groups, not with the scorer's ``aggregate``.
    """
    y = ds["sales"].values[hierarchy.rows][:, WINDOW_DAYS[window]]
    levels = []
    for group_idx, n_groups in zip(hierarchy.group_idx, hierarchy.n_groups, strict=True):
        level = np.zeros((n_groups, len(score.QUANTILES), N_STEPS))
        for g in range(n_groups):
            members = np.flatnonzero(group_idx == g)
            level[g] = y[members].sum(axis=0)
            if 0 in hierarchy.rows[members]:
                level[g] += offset_series_0
        levels.append(level)
    return levels


class TestWsplScorer:
    """Criterion 1: unit tests on a small hierarchy checked by hand, full or a set of bottom series."""

    def test_pinball_loss_hand_values(self):
        # y=3: forecast below it pays u per unit, above it pays 1-u per unit.
        assert score.pinball_loss(np.array(3.0), np.array(1.0), 0.9) == pytest.approx(1.8)
        assert score.pinball_loss(np.array(3.0), np.array(5.0), 0.9) == pytest.approx(0.2)
        assert score.pinball_loss(np.array(3.0), np.array(3.0), 0.9) == 0.0

    def test_scale_ignores_leading_zeros_and_is_absolute(self):
        # First sale on day 2: differences from there are 3, 2, 0 -> mean 5/3, not squared (13/3) or over 5 days.
        scale = score.spl_scale(np.array([[0.0, 0.0, 1.0, 4.0, 2.0, 2.0]]))
        assert scale == pytest.approx([5 / 3])

    @pytest.mark.parametrize("history", [[0.0] * 6, [3.0] * 6, [0.0, 0.0, 0.0, 0.0, 0.0, 7.0]])
    def test_flat_or_unlaunched_series_get_scale_one(self, history):
        assert score.spl_scale(np.array([history])) == pytest.approx([1.0])

    def test_level_wspl_with_quantile_dependent_forecast(self):
        # One series alternating 0, 1 (scale 1). Forecast 0 up to the 0.25 quantile, 1 above it. At forecast 0 the
        # loss is u/2 per day on average, at forecast 1 it is (1-u)/2: (0.445 + 0.945) / 2 / 9 over the 9 quantiles.
        y_train = np.tile([1.0, 0.0], 20)[None, :]
        y_score = np.tile([0.0, 1.0], N_STEPS // 2)[None, :]
        forecast = np.where(np.arange(9) < 4, 0.0, 1.0)[None, :, None] * np.ones((1, 9, N_STEPS))
        assert score.level_wspl(y_train, y_score, forecast, np.array([5.0])) == pytest.approx(0.695 / 9)

    def test_level_wspl_weights_groups_by_dollars(self):
        # Two groups, perfect forecast for one and +1 (loss 0.5, scale 1) for the other; weights 3:1.
        y_train = np.tile([1.0, 0.0], 20)[None, :].repeat(2, axis=0)
        y_score = np.tile([0.0, 1.0], N_STEPS // 2)[None, :].repeat(2, axis=0)
        forecast = np.repeat(y_score[:, None, :], 9, axis=1)
        forecast[1] += 1.0
        assert score.level_wspl(y_train, y_score, forecast, np.array([3.0, 1.0])) == pytest.approx(0.125)
        assert score.level_wspl(y_train, y_score, forecast, np.array([0.0, 0.0])) == pytest.approx(0.25)

    def test_hierarchy_ids_and_sizes(self):
        hierarchy = score.build_hierarchy(_make_ds())
        assert hierarchy.n_groups == N_GROUPS
        by_level = {i + 1: ids.tolist() for i, ids in enumerate(hierarchy.ids)}
        assert by_level[1] == ["Total_X"]
        assert by_level[2] == ["CA_X"]
        assert by_level[3] == ["CA_1_X", "CA_2_X"]
        assert by_level[4] == ["FOODS_X"]
        assert by_level[5] == ["FOODS_1_X"]
        assert by_level[6] == ["CA_FOODS"]
        assert by_level[7] == ["CA_FOODS_1"]
        assert by_level[8] == ["CA_1_FOODS", "CA_2_FOODS"]
        assert by_level[9] == ["CA_1_FOODS_1", "CA_2_FOODS_1"]
        assert by_level[10] == ["FOODS_1_001_X", "FOODS_1_002_X"]
        assert by_level[11] == ["CA_FOODS_1_001", "CA_FOODS_1_002"]  # state first, as in the competition's ids
        assert by_level[12] == sorted(SERIES_IDS)

    @pytest.mark.parametrize("window", ["prevalidation", "validation", "evaluation"])
    def test_full_hierarchy_matches_hand_values(self, window):
        ds = _make_ds()
        hierarchy = score.build_hierarchy(ds)
        forecasts = _forecasts_at_actuals(ds, hierarchy, window)
        per_level = score.compute_hierarchical_wspl(ds, forecasts, hierarchy, window=window)
        assert per_level["n_series"].tolist() == N_GROUPS
        assert per_level["wspl"].to_numpy() == pytest.approx(EXPECTED_FULL)
        assert per_level["wspl"].mean() == pytest.approx(9 / 56)

    @pytest.mark.parametrize("window", ["validation", "evaluation"])
    def test_set_of_bottom_series_is_scored_on_its_own_hierarchy(self, window):
        ds = _make_ds()
        hierarchy = score.build_hierarchy(ds, np.array([0, 1]))
        forecasts = _forecasts_at_actuals(ds, hierarchy, window)
        per_level = score.compute_hierarchical_wspl(ds, forecasts, hierarchy, window=window)
        assert per_level["n_series"].tolist() == [1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2]
        assert per_level["wspl"].to_numpy() == pytest.approx(EXPECTED_PAIR)
        assert per_level["wspl"].mean() == pytest.approx(0.3)

    def test_perfect_forecast_scores_zero(self):
        ds = _make_ds()
        hierarchy = score.build_hierarchy(ds)
        forecasts = _forecasts_at_actuals(ds, hierarchy, "validation", offset_series_0=0.0)
        per_level = score.compute_hierarchical_wspl(ds, forecasts, hierarchy, window="validation")
        assert per_level["wspl"].to_numpy() == pytest.approx(0.0)

    def test_window_without_actuals_is_refused(self):
        ds = _make_ds(test_actuals=False)
        hierarchy = score.build_hierarchy(ds)
        forecasts = [np.zeros((n, 9, N_STEPS)) for n in hierarchy.n_groups]
        with pytest.raises(RuntimeError, match="no evaluation actuals"):
            score.compute_hierarchical_wspl(ds, forecasts, hierarchy, window="evaluation")
        assert len(score.compute_hierarchical_wspl(ds, forecasts, hierarchy, window="validation")) == N_LEVELS

    def test_misshapen_forecasts_are_refused(self):
        ds = _make_ds()
        hierarchy = score.build_hierarchy(ds)
        forecasts = [np.zeros((n, 9, N_STEPS)) for n in hierarchy.n_groups]
        forecasts[2] = forecasts[2][:1]
        with pytest.raises(ValueError, match="Level 3"):
            score.compute_hierarchical_wspl(ds, forecasts, hierarchy)
        with pytest.raises(ValueError, match="one per level"):
            score.compute_hierarchical_wspl(ds, forecasts[:5], hierarchy)


def _scrambled_sample(hierarchy, seed: int = 0) -> np.ndarray:
    """A stand-in for ``sample_submission.csv``: every id of the small hierarchy, in an order that is not ours."""
    ids = [
        f"{series}_{u:.3f}_{window}"
        for window in submission.SUBMISSION_WINDOWS
        for u in score.QUANTILES
        for series in np.concatenate(hierarchy.ids)
    ]
    return np.random.default_rng(seed).permutation(ids)


class TestSubmissionWriter:
    """Criterion 2: the writer lays the forecasts out in the sample's ids and row order and checks them."""

    @pytest.fixture
    def setup(self):
        ds = _make_ds()
        hierarchy = score.build_hierarchy(ds)
        rng = np.random.default_rng(1)
        forecasts = {
            w: [rng.uniform(0, 5, (n, 9, N_STEPS)) for n in hierarchy.n_groups] for w in submission.SUBMISSION_WINDOWS
        }
        return ds, hierarchy, forecasts, _scrambled_sample(hierarchy)

    def test_rows_follow_the_sample_and_carry_the_right_forecast(self, setup):
        _, hierarchy, forecasts, sample_ids = setup
        sub = submission.build_submission(forecasts, hierarchy, sample_ids)
        assert len(sub) == len(sample_ids) == sum(N_GROUPS) * 9 * 2
        assert sub["id"].tolist() == sample_ids.tolist()
        assert list(sub.columns) == ["id", *[f"F{i}" for i in range(1, 29)]]
        row = sub.set_index("id").loc["FOODS_1_001_CA_2_0.835_evaluation"].to_numpy()
        group = list(hierarchy.ids[11]).index("FOODS_1_001_CA_2")
        assert row == pytest.approx(forecasts["evaluation"][11][group, 6])
        row = sub.set_index("id").loc["Total_X_0.005_validation"].to_numpy()
        assert row == pytest.approx(forecasts["validation"][0][0, 0])

    def test_written_file_reads_back_to_the_same_forecasts(self, setup, tmp_path):
        _, hierarchy, forecasts, sample_ids = setup
        sample_path = tmp_path / "sample_submission.csv"
        pd.DataFrame({"id": sample_ids, **{f"F{i}": 0 for i in range(1, 29)}}).to_csv(sample_path, index=False)
        output = tmp_path / "out" / "submission.csv"
        submission.write_submission(forecasts, hierarchy, sample_path, output)

        written = pd.read_csv(output)
        assert written["id"].tolist() == pd.read_csv(sample_path)["id"].tolist()
        for window in submission.SUBMISSION_WINDOWS:
            back = submission.read_forecasts(written, window, hierarchy)
            for got, want in zip(back, forecasts[window], strict=True):
                assert got == pytest.approx(want, rel=1e-5)

    def test_missing_sample_file_points_at_the_fetch(self, tmp_path):
        with pytest.raises(SystemExit, match="m5_fetch.py"):
            submission.read_sample_ids(tmp_path / "sample_submission.csv")

    def test_sample_with_an_unknown_id_is_refused(self, setup):
        _, hierarchy, forecasts, sample_ids = setup
        bad = sample_ids.copy()
        bad[3] = "NOT_A_SERIES_0.500_validation"
        with pytest.raises(submission.SubmissionError, match="no forecast"):
            submission.build_submission(forecasts, hierarchy, bad)

    def test_sample_with_a_missing_or_repeated_row_is_refused(self, setup):
        _, hierarchy, forecasts, sample_ids = setup
        with pytest.raises(submission.SubmissionError, match="rows"):
            submission.build_submission(forecasts, hierarchy, sample_ids[:-1])
        repeated = sample_ids.copy()
        repeated[0] = repeated[1]
        with pytest.raises(submission.SubmissionError, match="unique"):
            submission.build_submission(forecasts, hierarchy, repeated)

    def test_forecasts_that_do_not_fit_are_refused(self, setup):
        _, hierarchy, forecasts, sample_ids = setup
        with pytest.raises(submission.SubmissionError, match="No forecasts for the evaluation"):
            submission.build_submission({"validation": forecasts["validation"]}, hierarchy, sample_ids)
        nan = {w: [f.copy() for f in fs] for w, fs in forecasts.items()}
        nan["validation"][5][0, 0, 0] = np.nan
        with pytest.raises(submission.SubmissionError, match="not all finite"):
            submission.build_submission(nan, hierarchy, sample_ids)
        short = {w: fs[:-1] for w, fs in forecasts.items()}
        with pytest.raises(submission.SubmissionError, match="forecast arrays"):
            submission.build_submission(short, hierarchy, sample_ids)

    def test_check_flags_a_swapped_row_and_a_wrong_count(self, setup):
        _, hierarchy, forecasts, sample_ids = setup
        sub = submission.build_submission(forecasts, hierarchy, sample_ids)
        submission.check_submission(sub, sample_ids)
        swapped = sub.copy()
        swapped.loc[[0, 1], "id"] = swapped.loc[[1, 0], "id"].to_numpy()
        with pytest.raises(submission.SubmissionError, match="Row 0"):
            submission.check_submission(swapped, sample_ids)
        with pytest.raises(submission.SubmissionError, match="rows"):
            submission.check_submission(sub.iloc[:-1], sample_ids)

    def test_reading_a_file_that_lacks_rows_is_refused(self, setup):
        _, hierarchy, forecasts, sample_ids = setup
        sub = submission.build_submission(forecasts, hierarchy, sample_ids)
        with pytest.raises(submission.SubmissionError, match="lacks"):
            submission.read_forecasts(sub[~sub["id"].str.startswith("CA_X_0.250")], "validation", hierarchy)
        with pytest.raises(submission.SubmissionError, match="Unknown window"):
            submission.read_forecasts(sub, "prevalidation", hierarchy)


class TestBaseline:
    """Criterion 2: the cheap baseline forecasts every series of every level, and its local WSPL is scored."""

    def test_forecast_is_the_empirical_quantile_of_the_days_before_the_window(self):
        ds = _make_ds()
        hierarchy = score.build_hierarchy(ds)
        for window, last_day in (("validation", N_TRAIN), ("evaluation", N_TRAIN + N_STEPS)):
            forecasts = baseline.baseline_forecasts(ds, hierarchy, window, history_days=10)
            # Total level: the sum of the four series over the 10 days before the window.
            total = ds["sales"].values[:, last_day - 10 : last_day].sum(axis=0)
            assert forecasts[0].shape == (1, 9, N_STEPS)
            assert forecasts[0][0, :, 0] == pytest.approx(np.quantile(total, score.QUANTILES))
            assert (forecasts[0] == forecasts[0][:, :, :1]).all()  # flat over the 28 days
            assert [f.shape for f in forecasts] == [(n, 9, N_STEPS) for n in N_GROUPS]
            assert all((np.diff(f, axis=1) >= 0).all() for f in forecasts)  # no crossing quantiles

    def test_writes_a_submission_that_matches_the_sample_and_scores_it(self, tmp_path):
        ds = _make_ds()
        hierarchy = score.build_hierarchy(ds)
        sample_ids = _scrambled_sample(hierarchy)
        sample_path = tmp_path / "sample_submission.csv"
        pd.DataFrame({"id": sample_ids, **{f"F{i}": 0 for i in range(1, 29)}}).to_csv(sample_path, index=False)

        windows = list(submission.SUBMISSION_WINDOWS)
        forecasts = {w: baseline.baseline_forecasts(ds, hierarchy, w, history_days=10) for w in windows}
        sub = submission.write_submission(forecasts, hierarchy, sample_path, tmp_path / "baseline.csv")
        assert pd.read_csv(tmp_path / "baseline.csv")["id"].tolist() == sample_ids.tolist()
        assert len(sub) == len(sample_ids)

        for window in windows:
            read_back = submission.read_forecasts(pd.read_csv(tmp_path / "baseline.csv"), window, hierarchy)
            direct = score.compute_hierarchical_wspl(ds, forecasts[window], hierarchy, window=window)
            via_file = score.compute_hierarchical_wspl(ds, read_back, hierarchy, window=window)
            assert via_file["wspl"].to_numpy() == pytest.approx(direct["wspl"].to_numpy())
            assert np.isfinite(direct["wspl"]).all()
            assert (direct["wspl"] > 0).all()

    def test_baseline_scores_a_set_of_bottom_series(self):
        ds = _make_ds()
        hierarchy = score.build_hierarchy(ds, np.array([0, 3]))
        forecasts = baseline.baseline_forecasts(ds, hierarchy, "prevalidation", history_days=5)
        per_level = score.compute_hierarchical_wspl(ds, forecasts, hierarchy, window="prevalidation")
        assert per_level["n_series"].tolist() == [1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        assert np.isfinite(per_level["wspl"]).all()


class _FakeKaggle:
    """Stands in for ``subprocess.run`` of the Kaggle CLI: saves ``<file>.zip`` into the ``-p`` directory."""

    def __init__(self, returncode: int = 0, stderr: str = ""):
        self.calls: list[list[str]] = []
        self.returncode = returncode
        self.stderr = stderr

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        if self.returncode == 0:
            out_dir = Path(command[command.index("-p") + 1])
            name = command[command.index("-f") + 1]
            with zipfile.ZipFile(out_dir / f"{name}.zip", "w") as zf:
                zf.writestr(name, "id,F1\nTotal_X_0.005_validation,0\n")
        return subprocess.CompletedProcess(command, self.returncode, stdout="", stderr=self.stderr)


class TestFetchUncertainty:
    """Criterion 3: the uncertainty sample submission is fetched into its own folder and unzipped."""

    @pytest.fixture(autouse=True)
    def _kaggle_on_path(self, monkeypatch):
        monkeypatch.setattr(fetch.shutil, "which", lambda name: f"/usr/bin/{name}")

    def test_downloads_the_file_unzips_it_and_leaves_the_accuracy_folder_alone(self, tmp_path, monkeypatch):
        accuracy = tmp_path / "m5-forecasting-accuracy"
        accuracy.mkdir()
        (accuracy / "sample_submission.csv").write_text("accuracy sample")
        uncertainty = tmp_path / "m5-forecasting-uncertainty"
        run = _FakeKaggle()
        monkeypatch.setattr(fetch.subprocess, "run", run)

        fetch.download_uncertainty(uncertainty)

        assert sorted(p.name for p in uncertainty.iterdir()) == ["sample_submission.csv"]  # zip deleted
        assert (uncertainty / "sample_submission.csv").read_text().startswith("id,F1")
        assert (accuracy / "sample_submission.csv").read_text() == "accuracy sample"
        (command,) = run.calls
        assert command[command.index("-c") + 1] == "m5-forecasting-uncertainty"
        assert command[command.index("-f") + 1] == "sample_submission.csv"
        assert command[command.index("-p") + 1] == str(uncertainty)

    def test_unaccepted_rules_name_the_uncertainty_competition(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fetch.subprocess, "run", _FakeKaggle(returncode=1, stderr="403 Client Error: Forbidden"))
        with pytest.raises(fetch.FetchError, match="competitions/m5-forecasting-uncertainty/rules"):
            fetch.download_uncertainty(tmp_path)
        assert not list(tmp_path.iterdir())

    def test_accuracy_rules_error_still_names_the_accuracy_competition(self):
        assert "competitions/m5-forecasting-accuracy/rules" in fetch._diagnose("403 Forbidden")

    def test_a_file_that_is_already_there_is_skipped(self, tmp_path, monkeypatch, capsys):
        accuracy, uncertainty = tmp_path / "a", tmp_path / "u"
        present = ((accuracy, fetch.EXPECTED_FILES + fetch.REFERENCE_FILES), (uncertainty, fetch.UNCERTAINTY_FILES))
        for directory, names in present:
            directory.mkdir()
            for name in names:
                (directory / name).write_text("x")

        def _no_calls(*args, **kwargs):  # pragma: no cover - only reached if the skip is broken
            raise AssertionError("a present file must not be downloaded again")

        monkeypatch.setattr(fetch.subprocess, "run", _no_calls)
        monkeypatch.setattr(fetch.urllib.request, "urlretrieve", _no_calls)
        argv = ["m5_fetch.py", "--data-dir", str(accuracy), "--uncertainty-dir", str(uncertainty)]
        monkeypatch.setattr(sys, "argv", argv)
        assert fetch.main() == 0
        assert "nothing to do" in capsys.readouterr().out
