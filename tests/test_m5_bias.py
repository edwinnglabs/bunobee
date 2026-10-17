"""Unit tests for the level-bias diagnostic, ``playground/m5_prototype/m5_bias.py``.

The script asks whether a model's forecast tracks the arithmetic or the geometric mean of ``max(sales, 0.1)``
(issue #113). These tests check its per-series means, the volume terciles and zero-share bins, the group
ratios and the two tracking measures (``position`` and ``slope``) on series small enough to work out by hand,
the share of series with a fitted sigma on a prior bound (issue #115), and that it fits the same random sample
as ``m5_run.py``. None needs the raw M5 data.

The script is a playground script, not part of the package, so it is loaded by path as
``tests/test_m5_run.py`` loads the runner.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("optax", reason="needs the optional 'm5' extra: pip install 'bunobee[m5]'")

M5_DIR = Path(__file__).resolve().parents[1] / "playground" / "m5_prototype"


def _load_script(name: str) -> ModuleType:
    """Import a playground M5 script by path and register it under its own name.

    Registering matters for the scripts ``m5_bias`` imports by name, which only resolve from ``tests/`` once
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
m5_bias = _load_script("m5_bias")


def _stats(**columns: list[float]) -> pd.DataFrame:
    """:func:`m5_bias.series_stats`-shaped rows for the ``fit`` window, with the given columns."""
    n = len(next(iter(columns.values())))
    base = {"forecast_mean": [1.0] * n, "actual_mean": [1.0] * n, "zero_share_fit": [0.0] * n}
    return pd.DataFrame({**base, **columns})


class TestSeriesStats:
    def test_means_of_the_clipped_sales(self):
        # Fit days: 0, 0, 2, 0, 2, 2 -> clipped 0.1, 0.1, 2, 0.1, 2, 2; the last 2 fit days are 2, 2.
        sales_fit = np.array([[0.0, 0.0, 2.0, 0.0, 2.0, 2.0]])
        stats = m5_bias.series_stats(
            sales_fit, actual=np.array([[1.0, 3.0]]), forecasts=np.array([[2.0, 4.0]]), recent_days=2
        )

        row = stats.iloc[0]
        assert row["forecast_mean"] == pytest.approx(3.0)
        assert row["actual_mean"] == pytest.approx(2.0)
        assert row["mean_fit"] == pytest.approx(1.0)
        assert row["amean_fit"] == pytest.approx(6.3 / 6)
        assert row["gmean_fit"] == pytest.approx(np.sqrt(0.2))  # (0.1^3 * 2^3)^(1/6)
        assert row["zero_share_fit"] == pytest.approx(0.5)
        assert row[["amean_last2", "gmean_last2"]].tolist() == pytest.approx([2.0, 2.0])
        assert row["zero_share_last2"] == 0.0
        # From the first sale on (day 3): 2, 0, 2, 2.
        assert row["zero_share_active"] == pytest.approx(0.25)

    def test_a_series_that_never_sells_counts_every_day(self):
        stats = m5_bias.series_stats(np.zeros((1, 4)), np.zeros((1, 2)), np.full((1, 2), 0.5), recent_days=2)

        row = stats.iloc[0]
        assert row["zero_share_active"] == 1.0
        assert row[["amean_fit", "gmean_fit"]].tolist() == pytest.approx([m5_bias.SALES_FLOOR] * 2)

    def test_window_names(self):
        assert m5_bias.window_names(28) == ("fit", "last28")


class TestAssignGroups:
    def test_volume_terciles_split_by_rank(self):
        stats = pd.DataFrame({"mean_fit": [5.0, 1.0, 3.0, 0.0, 9.0, 3.0], "zero_share_active": [0.0] * 6})

        groups = m5_bias.assign_groups(stats)

        # Ranks 4, 1, 2, 0, 5, 3 (ties in row order) -> thirds of two.
        assert list(groups["volume"]) == ["high", "low", "mid", "low", "high", "mid"]
        assert list(groups["volume"].cat.categories) == list(m5_bias.VOLUME_TERCILES)

    def test_zero_share_bins_close_on_the_right(self):
        shares = [0.0, 0.25, 0.26, 0.5, 0.75, 0.9, 1.0]
        stats = pd.DataFrame({"mean_fit": np.arange(len(shares), dtype=float), "zero_share_active": shares})

        groups = m5_bias.assign_groups(stats)

        assert list(groups["zero_bin"]) == ["0-25%", "0-25%", "25-50%", "25-50%", "50-75%", "75-100%", "75-100%"]


class TestGroupTable:
    def test_ratios_are_ratios_of_sums(self):
        stats = _stats(
            forecast_mean=[1.0, 3.0],
            actual_mean=[2.0, 2.0],
            amean_fit=[2.0, 6.0],
            gmean_fit=[0.5, 1.5],
            zero_share_fit=[0.2, 0.4],
        ).assign(dept_id=["A", "A"], volume="low", zero_bin="0-25%")

        table = m5_bias.group_table(stats, "fit")

        assert list(table["grouping"]) == list(m5_bias.GROUPINGS)
        total = table.iloc[0]
        assert total["n"] == 2
        assert total["zero_share"] == pytest.approx(0.3)
        assert total["fc_act"] == pytest.approx(1.0)
        assert total["act_am"] == pytest.approx(0.5)
        assert total["fc_am"] == pytest.approx(0.5)
        assert total["fc_gm"] == pytest.approx(2.0)
        assert total["am_gm"] == pytest.approx(4.0)
        assert total["position"] == pytest.approx(0.5)  # log 2 / log 4

    def test_groups_are_split_and_ordered(self):
        stats = m5_bias.assign_groups(
            _stats(
                forecast_mean=[1.0, 2.0, 3.0],
                amean_fit=[1.0, 2.0, 3.0],
                gmean_fit=[0.5, 1.0, 1.5],
                mean_fit=[3.0, 1.0, 2.0],
                zero_share_active=[0.9, 0.1, 0.1],
            ).assign(dept_id=["H", "F", "H"])
        )

        table = m5_bias.group_table(stats, "fit")

        assert list(zip(table["grouping"], table["group"])) == [
            ("total", "all"),
            ("dept", "F"),
            ("dept", "H"),
            ("volume", "low"),
            ("volume", "mid"),
            ("volume", "high"),
            ("zero_share", "0-25%"),
            ("zero_share", "75-100%"),
        ]
        assert list(table["n"]) == [3, 1, 2, 1, 1, 1, 2, 1]

    @pytest.mark.parametrize("mean, position", [("gmean_fit", 0.0), ("amean_fit", 1.0)])
    def test_position_is_0_at_the_geometric_and_1_at_the_arithmetic_mean(self, mean, position):
        stats = _stats(amean_fit=[2.0, 5.0], gmean_fit=[1.0, 1.0]).assign(dept_id="A", volume="low", zero_bin="0-25%")
        stats["forecast_mean"] = stats[mean]

        assert m5_bias.group_table(stats, "fit")["position"].iloc[0] == pytest.approx(position)

    def test_position_is_nan_without_a_gap(self):
        stats = _stats(amean_fit=[2.0], gmean_fit=[2.0]).assign(dept_id="A", volume="low", zero_bin="0-25%")

        assert np.isnan(m5_bias.group_table(stats, "fit")["position"].iloc[0])

    def test_act_ratio_is_nan_without_actual_sales(self):
        stats = _stats(actual_mean=[0.0], amean_fit=[2.0], gmean_fit=[1.0]).assign(
            dept_id="A", volume="low", zero_bin="0-25%"
        )

        assert np.isnan(m5_bias.group_table(stats, "fit")["fc_act"].iloc[0])


class TestTrackingSlope:
    # Four series with the same geometric mean and growing gaps to the arithmetic mean.
    AMEAN = np.array([1.0, 2.0, 4.0, 8.0])
    GMEAN = np.ones(4)

    def test_slope_0_for_a_constant_multiple_of_the_geometric_mean(self):
        stats = _stats(forecast_mean=list(1.2 * self.GMEAN), amean_fit=list(self.AMEAN), gmean_fit=list(self.GMEAN))

        assert m5_bias.tracking_slope(stats, "fit") == pytest.approx(0.0, abs=1e-12)

    def test_slope_1_for_a_constant_multiple_of_the_arithmetic_mean(self):
        stats = _stats(forecast_mean=list(0.9 * self.AMEAN), amean_fit=list(self.AMEAN), gmean_fit=list(self.GMEAN))

        assert m5_bias.tracking_slope(stats, "fit") == pytest.approx(1.0)

    def test_slope_of_a_geometric_blend(self):
        # forecast = am^0.3 * gm^0.7, so log(forecast / gm) = 0.3 * log(am / gm).
        forecast = self.AMEAN**0.3 * self.GMEAN**0.7
        stats = _stats(forecast_mean=list(forecast), amean_fit=list(self.AMEAN), gmean_fit=list(self.GMEAN))

        assert m5_bias.tracking_slope(stats, "fit") == pytest.approx(0.3)

    def test_slope_is_nan_when_every_gap_is_equal(self):
        stats = _stats(forecast_mean=[1.0, 2.0], amean_fit=[2.0, 4.0], gmean_fit=[1.0, 2.0])

        assert np.isnan(m5_bias.tracking_slope(stats, "fit"))

    def test_series_without_a_positive_forecast_are_left_out(self):
        stats = _stats(
            forecast_mean=[*(0.9 * self.AMEAN), 0.0], amean_fit=[*self.AMEAN, 3.0], gmean_fit=[*self.GMEAN, 1.0]
        )

        assert m5_bias.tracking_slope(stats, "fit") == pytest.approx(1.0)


class TestPriorBoundShares:
    def test_shares_of_series_on_each_bound(self):
        # Priors: sigma_h ~ U(0.1, 0.5), within 0.004 of a bound; sigma_q ~ U(0.01, 0.1), within 0.0009.
        # Series 0 has sigma_h and the first drift on their lower bounds, series 1 no sigma on a bound, series 2
        # sigma_h on its upper bound and series 3 sigma_h past it and the shared drift on its upper bound.
        fitted = {
            "sigma_h": np.array([0.1, 0.3, 0.497, 0.503]),
            "sigma_q": np.array([[0.0105, 0.05, 0.05], [0.05, 0.05, 0.05], [0.05, 0.05, 0.05], [0.05, 0.0995, 0.0995]]),
        }

        shares = m5_bias.prior_bound_shares(fitted)

        assert shares == pytest.approx(
            {
                "sigma_h_low": 0.25,
                "sigma_h_high": 0.5,
                "sigma_q_first_low": 0.25,
                "sigma_q_first_high": 0.0,
                "sigma_q_rest_low": 0.0,
                "sigma_q_rest_high": 0.25,
                "any": 0.75,
            }
        )

    def test_bounds_are_the_fitters_priors(self):
        from bunobee.models.m5 import m5_ssp_optim

        assert m5_bias.SIGMA_H_BOUNDS == m5_ssp_optim.SIGMA_H_BOUNDS == (0.1, 0.5)
        assert m5_bias.SIGMA_Q_BOUNDS == m5_ssp_optim.SIGMA_Q_BOUNDS == (0.01, 0.1)

    def test_empty_without_the_ssp_sigmas(self):
        assert m5_bias.prior_bound_shares({"sigma_h": np.array([0.3])}) == {}
        assert m5_bias.prior_bound_shares({}) == {}


def test_sample_matches_the_runner():
    # m5_run.py draws its random sample with default_rng(seed).choice(n, size=n_demo, replace=False).
    expected = np.random.default_rng(2026).choice(30490, size=3000, replace=False)

    np.testing.assert_array_equal(m5_bias.sample_series(30490, 3000, 2026), expected)
    assert m5_bias.sample_series(10, 0, 1).size == 10
