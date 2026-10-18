"""Unit tests for the run log and the forecast review of ``playground/m5_prototype/m5_run.py``.

Every run appends one row to ``output/runs.csv`` (issue #104), and the Kaggle scores of a submitted run are
typed into its row by hand. These tests check that appending keeps such hand-typed values verbatim and that
a column added to the row later does not break an existing log, but lands where the row puts it (issue #125
adds ``wrmsse_prevalidation`` before ``wrmsse_validation``).

``--origins`` (issue #125) fits the pre-validation, validation and evaluation origins by default. These tests
check the choices and that ``both`` is still the two origins a submission needs.

``forecasts.pdf`` (issue #105) draws the total, store and department sums of the fitted series, then series
picked at random, by worst RMSSE or by top dollar-sales weight. These tests check the sums, the RMSSE, bias
and weight in the panel titles against the scorer, and the picking, on a tiny synthetic dataset.

``--round`` (issue #117) clips the forecasts at 0 and rounds them to integers. These tests check that it is off
by default and that it leaves only non-negative integers.

The runner is a playground script, not part of the package, so it is loaded by path as
``tests/test_m5_models.py`` loads the model registry.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")  # headless backend; must precede m5_run's pyplot import

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

    def test_columns_follow_the_latest_row(self, tmp_path):
        # A log written before #125: no pre-validation score, a Kaggle score typed in, and a column later runs
        # no longer write.
        path = tmp_path / "runs.csv"
        m5_run._append_run_log(path, {**_row(kaggle_public="1.08127"), "retired": "x"})
        row: dict[str, object] = {}
        for key, value in _row(run_dir="20261016_210000").items():
            if key == "wrmsse_validation":
                row["wrmsse_prevalidation"] = 0.9
            row[key] = value

        m5_run._append_run_log(path, row)

        log = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert list(log.columns) == [*row, "retired"]
        assert list(log["wrmsse_prevalidation"]) == ["", "0.9"]
        assert list(log["kaggle_public"]) == ["1.08127", ""]
        assert list(log["retired"]) == ["x", ""]


class TestOrigins:
    def test_all_three_origins_by_default(self):
        assert m5_run._parse_args([]).origins == "all"
        assert m5_run.ORIGINS["all"] == ("prevalidation", "validation", "evaluation")

    def test_both_is_the_two_origins_a_submission_needs(self):
        assert m5_run.ORIGINS["both"] == ("validation", "evaluation")

    def test_a_quick_loop_can_fit_prevalidation_alone(self):
        assert m5_run.ORIGINS[m5_run._parse_args(["--origins", "prevalidation"]).origins] == ("prevalidation",)


class TestRounding:
    def test_rounding_is_off_by_default(self):
        assert m5_run._parse_args([]).round is False
        assert m5_run._parse_args(["--round"]).round is True

    def test_rounded_forecasts_are_non_negative_integers(self):
        forecasts = np.array([[-0.7, -0.2, 0.0, 0.4, 0.6], [1.49, 1.51, 2.5, 3.5, 12.2]], dtype=np.float32)

        rounded = m5_run._round_forecasts(forecasts)

        # Negatives clip to 0 before rounding; halves round to even.
        np.testing.assert_array_equal(rounded, [[0, 0, 0, 0, 1], [1, 2, 2, 4, 12]])
        assert rounded.shape == forecasts.shape and rounded.dtype == forecasts.dtype
        assert (rounded >= 0).all() and (rounded == np.floor(rounded)).all()
        # The input is left unrounded.
        assert forecasts[0, 3] == np.float32(0.4)


# Review dataset: 4 series in 2 stores x 2 departments, N_FIT_DAYS fit days, then the 28 validation days.
N_FIT_DAYS = 35
HORIZON = 28
LEVELS = np.array([1.0, 3.0, 6.0, 10.0])
PRICES = np.array([1.0, 2.0, 0.5, 4.0])
# Forecast = actual + OFFSETS. Every series alternates level, level + 1, so its RMSSE scale is 1 and its RMSSE
# is its offset: worst first is 2, 0, 3, 1.
OFFSETS = np.array([2.0, 0.0, 5.0, 1.0])
# Dollar sales over the last 28 fit days: (28 x level + 14) x price = 42, 196, 91, 1176; top first is 3, 1, 2, 0.
DOLLARS = (28 * LEVELS + 14) * PRICES


@pytest.fixture(scope="module")
def review_ds() -> xr.Dataset:
    """M5-shaped dataset whose RMSSE, weights and sums are worked out by hand above."""
    n_dates = N_FIT_DAYS + HORIZON
    sales = (LEVELS[:, None] + np.arange(n_dates) % 2).astype(np.float32)
    return xr.Dataset(
        {
            "sales": (("series_id", "date"), sales),
            "sell_price": (("series_id", "date"), np.repeat(PRICES[:, None], n_dates, axis=1).astype(np.float32)),
        },
        coords={
            "series_id": ["F_1_CA_1", "H_1_CA_1", "F_1_CA_2", "H_1_CA_2"],
            "item_id": ["F_1", "H_1", "F_1", "H_1"],
            "dept_id": ["FOODS_1", "HOBBIES_1", "FOODS_1", "HOBBIES_1"],
            "cat_id": ["FOODS", "HOBBIES", "FOODS", "HOBBIES"],
            "store_id": ["CA_1", "CA_1", "CA_2", "CA_2"],
            "state_id": ["CA"] * 4,
            "date": pd.date_range("2016-01-01", periods=n_dates),
            "split": ("date", np.array(["train"] * N_FIT_DAYS + ["validation"] * HORIZON)),
        },
    )


@pytest.fixture(scope="module")
def panels(review_ds) -> "m5_run._Panels":
    """Validation-origin panels of all four series, fitted in dataset order, with forecast = actual + OFFSETS."""
    fit_idx = np.arange(4)
    forecasts = review_ds["sales"].values[:, N_FIT_DAYS:] + OFFSETS[:, None]
    insample = np.arange(4 * N_FIT_DAYS, dtype=np.float64).reshape(4, N_FIT_DAYS)
    return m5_run._origin_panels(review_ds, fit_idx, "validation", forecasts, insample)


class TestForecastReview:
    def test_series_panels_carry_rmsse_and_weight(self, panels):
        np.testing.assert_allclose(panels.rmsse(), OFFSETS)
        np.testing.assert_allclose(panels.weight, DOLLARS / DOLLARS.sum())
        assert panels.history.shape == (4, N_FIT_DAYS)
        assert panels.actual.shape == panels.forecast.shape == (4, HORIZON)

    def test_sums_are_the_total_then_each_store_and_department(self, review_ds, panels):
        sums = m5_run._sum_panels(panels, review_ds, np.arange(4))

        assert sums.names == ["Total", "store CA_1", "store CA_2", "dept FOODS_1", "dept HOBBIES_1"]
        members = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=float)
        for field in ("history", "insample", "actual", "forecast", "weight"):
            np.testing.assert_allclose(getattr(sums, field), members @ getattr(panels, field), err_msg=field)

    def test_total_panel_matches_the_level_1_score(self, review_ds, panels):
        # The Total panel's RMSSE and bias are the scorer's level-1 WRMSSE and bias of the same forecasts.
        sums = m5_run._sum_panels(panels, review_ds, np.arange(4))
        level_1 = m5_run.compute_hierarchical_wrmsse(
            review_ds, panels.forecast.astype(np.float64), review_ds["series_id"].values, window="validation"
        ).iloc[0]

        assert sums.rmsse()[0] == pytest.approx(level_1["wrmsse"])
        assert sums.bias()[0] == pytest.approx(level_1["bias_pct"] / 100)

    @pytest.mark.parametrize("select, expected", [("worst", [2, 0, 3]), ("top", [3, 1, 2]), ("random", [1, 3, 0])])
    def test_select_series(self, panels, select, expected):
        rows = m5_run._select_series(select, panels, random_order=np.array([1, 3, 0, 2]), n_plot=3)

        assert list(rows) == expected

    def test_without_actuals_there_is_no_rmsse_or_bias_in_the_title(self, panels):
        no_actuals = dataclasses.replace(panels, actual=np.full_like(panels.actual, np.nan))

        assert np.isnan(no_actuals.rmsse()).all() and np.isnan(no_actuals.bias()).all()
        assert m5_run._panel_title("F_1_CA_1", np.nan, np.nan, 0.25) == "F_1_CA_1   weight 25%"
        assert m5_run._panel_title("Total", 0.804, -0.1013, 1.0) == "Total   RMSSE 0.80   bias -10.1%   weight 100%"

    def test_pages_hold_per_page_panels(self, tmp_path, review_ds, panels):
        sums = m5_run._sum_panels(panels, review_ds, np.arange(4))
        with m5_run.PdfPages(tmp_path / "forecasts.pdf") as pdf:
            assert m5_run._draw_pages(pdf, sums, "sums", per_page=2) == 3
            assert m5_run._draw_pages(pdf, dataclasses.replace(panels, insample=None), "series", per_page=5) == 1
            assert pdf.get_pagecount() == 4

    def test_a_misshapen_insample_fit_is_rejected(self):
        forecasts = np.zeros((4, m5_run.HORIZON))
        m5_run._check_model_output("m", forecasts, {m5_run.INSAMPLE_KEY: np.zeros((4, N_FIT_DAYS))}, 4, N_FIT_DAYS)
        with pytest.raises(ValueError, match="in-sample fit"):
            m5_run._check_model_output(
                "m", forecasts, {m5_run.INSAMPLE_KEY: np.zeros((4, N_FIT_DAYS - 1))}, 4, N_FIT_DAYS
            )
