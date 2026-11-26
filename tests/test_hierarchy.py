"""Tests of :mod:`bunobee.hierarchy`: building levels, summing draws up them, and their quantiles.

The sums are checked against a per-draw ``pandas`` ``groupby`` on a long table, and the quantiles against
``np.quantile``. The last test feeds the dataset ``SSP.predict`` returns straight into :func:`aggregate_draws`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from bunobee import Hierarchy, Level, aggregate_draws, draw_quantiles
from bunobee.models.ssp import SSP

N_SAMPLE, N_TIME = 7, 5
LEVELS = {"total": [], "store": ["store_id"], "dept": ["dept_id"], "store_dept": ["store_id", "dept_id"]}
QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]


@pytest.fixture(scope="module")
def meta() -> pd.DataFrame:
    """Six bottom series over two stores and three departments, in a shuffled order."""
    return pd.DataFrame(
        {
            "store_id": ["S2", "S1", "S1", "S2", "S1", "S2"],
            "dept_id": ["D3", "D1", "D2", "D1", "D3", "D3"],
        }
    )


@pytest.fixture(scope="module")
def draws(meta) -> np.ndarray:
    """``(sample, time, series)`` draws whose series are correlated within a draw."""
    rng = np.random.default_rng(0)
    common = rng.normal(size=(N_SAMPLE, N_TIME, 1))
    return 10.0 + common + rng.normal(size=(N_SAMPLE, N_TIME, len(meta)))


@pytest.fixture(scope="module")
def hierarchy(meta) -> Hierarchy:
    return Hierarchy.from_frame(meta, LEVELS)


def _groupby_sums(draws: np.ndarray, meta: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """Per-draw, per-time ``groupby`` sum over a long table, as ``(sample, time, group)`` with sorted groups."""
    n_sample, n_time, n_series = draws.shape
    long = pd.DataFrame(
        {
            "sample": np.repeat(np.arange(n_sample), n_time * n_series),
            "time": np.tile(np.repeat(np.arange(n_time), n_series), n_sample),
            "value": draws.ravel(),
        }
    )
    for key in keys:
        long[key] = np.tile(meta[key].to_numpy(), n_sample * n_time)
    sums = long.groupby(["sample", "time", *keys], sort=True)["value"].sum()
    return sums.to_numpy().reshape(n_sample, n_time, -1)


class TestFromFrame:
    def test_levels_and_groups(self, hierarchy, meta):
        assert hierarchy.names == list(LEVELS)
        assert hierarchy.n_series == len(meta)
        assert [level.n_groups for level in hierarchy] == [1, 2, 3, 5]
        assert list(hierarchy["total"].labels) == ["total"]
        np.testing.assert_array_equal(hierarchy["total"].group_idx, np.zeros(len(meta)))
        assert list(hierarchy["store"].labels) == ["S1", "S2"]
        np.testing.assert_array_equal(hierarchy["store"].group_idx, [1, 0, 0, 1, 0, 1])
        assert isinstance(hierarchy["store_dept"].labels, pd.MultiIndex)
        assert list(hierarchy["store_dept"].labels.names) == ["store_id", "dept_id"]

    def test_groups_are_the_observed_combinations(self, hierarchy, meta):
        level = hierarchy["store_dept"]
        expected = sorted(set(zip(meta["store_id"], meta["dept_id"], strict=True)))
        assert list(level.labels) == expected
        for series, group in enumerate(level.group_idx):
            assert level.labels[group] == (meta["store_id"][series], meta["dept_id"][series])

    def test_unknown_column(self, meta):
        with pytest.raises(KeyError, match="not in meta"):
            Hierarchy.from_frame(meta, {"region": ["region_id"]})

    def test_unknown_level(self, hierarchy):
        with pytest.raises(KeyError, match="no level 'region'"):
            hierarchy["region"]

    def test_levels_must_group_the_same_series(self):
        with pytest.raises(ValueError, match="same series"):
            Hierarchy((Level("a", np.zeros(3, dtype=int), pd.Index(["x"])), Level("b", np.zeros(4, dtype=int), ["x"])))

    def test_level_names_must_be_distinct(self):
        level = Level("a", np.zeros(3, dtype=int), pd.Index(["x"]))
        with pytest.raises(ValueError, match="distinct"):
            Hierarchy((level, level))

    def test_group_index_out_of_range(self):
        with pytest.raises(ValueError, match="must lie in"):
            Level("a", np.array([0, 1, 2]), pd.Index(["x", "y"]))

    def test_frozen(self, hierarchy):
        with pytest.raises(AttributeError):
            hierarchy.levels = ()


class TestAggregateDraws:
    @pytest.mark.parametrize("name", list(LEVELS))
    def test_matches_a_per_draw_groupby_sum(self, draws, meta, hierarchy, name):
        sums = aggregate_draws(draws, hierarchy, level=name)
        assert sums.shape == (N_SAMPLE, N_TIME, hierarchy[name].n_groups)
        if LEVELS[name]:
            np.testing.assert_allclose(sums, _groupby_sums(draws, meta, LEVELS[name]))

    def test_total_is_the_sum_over_all_series(self, draws, hierarchy):
        total = aggregate_draws(draws, hierarchy, level="total")
        np.testing.assert_allclose(total[..., 0], draws.sum(axis=-1))

    def test_all_levels(self, draws, hierarchy):
        out = aggregate_draws(draws, hierarchy)
        assert list(out) == hierarchy.names
        for name, sums in out.items():
            np.testing.assert_array_equal(sums, aggregate_draws(draws, hierarchy, level=name))

    def test_data_array_keeps_dims_and_labels_the_groups(self, draws, hierarchy):
        da = xr.DataArray(
            draws, dims=("sample", "time", "series"), coords={"time": pd.date_range("2026-01-01", periods=N_TIME)}
        )
        store = aggregate_draws(da.transpose("series", "sample", "time"), hierarchy, level="store")
        assert store.dims == ("sample", "time", "group")
        assert list(store["group"].values) == ["S1", "S2"]
        assert store["time"].equals(da["time"])
        np.testing.assert_allclose(store.values, aggregate_draws(draws, hierarchy, level="store"))

        store_dept = aggregate_draws(da, hierarchy, level="store_dept")
        assert list(store_dept["store_id"].values) == ["S1", "S1", "S1", "S2", "S2"]
        np.testing.assert_array_equal(
            store_dept["store_id"].values, hierarchy["store_dept"].labels.get_level_values("store_id")
        )

    def test_wrong_number_of_series(self, draws, hierarchy):
        with pytest.raises(ValueError, match="6 series"):
            aggregate_draws(draws[..., :-1], hierarchy, level="store")

    def test_dataset_without_forecast_samples(self, hierarchy):
        with pytest.raises(KeyError, match="forecast_samples"):
            aggregate_draws(xr.Dataset({"mu_samples": (("sample",), np.zeros(2))}), hierarchy)


class TestDrawQuantiles:
    def test_matches_np_quantile(self, draws, hierarchy):
        sums = aggregate_draws(draws, hierarchy, level="store_dept")
        np.testing.assert_array_equal(draw_quantiles(sums, QUANTILES), np.quantile(sums, QUANTILES, axis=0))
        np.testing.assert_array_equal(draw_quantiles(sums, 0.5), np.quantile(sums, 0.5, axis=0))

    def test_data_array_quantiles_over_sample(self, draws, hierarchy):
        da = xr.DataArray(draws, dims=("sample", "time", "series"))
        sums = aggregate_draws(da, hierarchy, level="store")
        q = draw_quantiles(sums.transpose("group", "time", "sample"), QUANTILES)
        assert q.dims == ("quantile", "group", "time")
        np.testing.assert_allclose(
            q.transpose("quantile", "time", "group").values, np.quantile(sums.values, QUANTILES, axis=0)
        )


def _ssp_panel(meta: pd.DataFrame, n_steps: int, horizon: int) -> tuple[np.ndarray, np.ndarray, xr.Dataset]:
    """A small level-plus-spend panel with one series per row of ``meta``, and a linear-filter prior for it."""
    n_series = len(meta)
    rng = np.random.default_rng(1)
    spend = rng.uniform(0.5, 1.5, size=(n_steps + horizon, n_series, 1))
    Z = np.concatenate([np.ones((n_steps + horizon, n_series, 1)), spend], axis=-1)
    y = Z @ np.array([1.0, 0.4]) + rng.normal(scale=0.1, size=(n_steps + horizon, n_series))
    prior = xr.Dataset(
        {
            "a0": (("state",), np.array([1.0, 0.3])),
            "P0": (("state",), np.array([0.5, 0.05])),
            "positivity": (("state",), np.array([False, True])),
            "sigma_q_alpha_prior": (("state",), np.full(2, 2.0)),
            "sigma_q_beta_prior": (("state",), np.full(2, 10.0)),
            "sigma_q_scale_prior": (("state",), np.full(2, 0.05)),
        },
        coords={"state": ["level", "spend"], "series": np.arange(n_series)},
    ).assign_attrs(sigma_q_family="beta")
    return y, Z, prior


def test_ssp_predict_goes_through_without_reshaping(meta, hierarchy):
    n_steps, horizon = 30, 4
    y, Z, prior = _ssp_panel(meta, n_steps, horizon)
    mcmc_args = {"num_warmup": 20, "num_samples": 20, "num_chains": 1, "progress_bar": False}
    forecast = SSP(prior, filter="linear").fit(y[:n_steps], Z[:n_steps], mcmc_args=mcmc_args).predict(Z[n_steps:])

    out = aggregate_draws(forecast, hierarchy)
    draws = forecast["forecast_samples"].values
    for name, sums in out.items():
        assert sums.dims == ("sample", "time", "group")
        np.testing.assert_allclose(sums.values, aggregate_draws(draws, hierarchy, level=name))
    np.testing.assert_allclose(out["total"].values[..., 0], draws.sum(axis=-1))

    q = draw_quantiles(out["store"], QUANTILES)
    assert q.dims == ("quantile", "time", "group")
    np.testing.assert_allclose(q.values, np.quantile(out["store"].values, QUANTILES, axis=0))
