"""Tests of :mod:`bunobee.models.count`, the pooled count model ``HierarchicalCount``.

A synthetic panel of negative binomial counts, with a constant level per series and a weekday pattern shared by the
series of a group, is fitted once by SVI: the fit must recover both. The forecast must be counts laid out ``(sample,
time, series)`` that ``aggregate_draws`` and ``draw_quantiles`` take as they are. A smaller panel runs every component
at once, as Orduz's M5 model does, with a short MAP fit. The full M5 check against the previous benchmark (WSPL
within 0.0002 of 0.16302) runs outside this repository.
"""

from __future__ import annotations

import numpy as np
import optax
import pandas as pd
import pytest
import xarray as xr

from bunobee import Hierarchy, aggregate_draws, draw_quantiles
from bunobee.models.base import NotFittedError
from bunobee.models.count import (
    DEFAULT_STEPS,
    DEFAULT_SUBSAMPLE,
    BlockWalk,
    Component,
    DayOfMonth,
    Events,
    Fourier,
    GroupShocks,
    HierarchicalCount,
    Level,
    LinearOneCycle,
    RecentMean,
    Regressor,
    Seasonal,
)
from bunobee.models.inference import MAP, NUTS, SVI
from bunobee.models.obs import NegBin2, Normal, Poisson

N_GROUPS, PER_GROUP, N_OBS, FUTURE = 4, 10, 280, 14
N_SERIES = N_GROUPS * PER_GROUP
CONCENTRATION = 10.0


def _meta(n_series: int, n_groups: int) -> pd.DataFrame:
    group = np.arange(n_series) % n_groups
    return pd.DataFrame({"group": group, "store": group % 2})


def _hierarchy(n_series: int = N_SERIES, n_groups: int = N_GROUPS) -> Hierarchy:
    return Hierarchy.from_frame(_meta(n_series, n_groups), {"group": ["group"], "store": ["store"]})


@pytest.fixture(scope="module")
def panel() -> dict:
    """Counts with a constant log level per series and a zero-sum weekday pattern per group."""
    rng = np.random.default_rng(0)
    group = _hierarchy()["group"].group_idx
    weekday = rng.normal(0.0, 0.4, (N_GROUPS, 7))
    weekday -= weekday.mean(axis=1, keepdims=True)
    level = rng.uniform(np.log(4.0), np.log(20.0), N_SERIES)
    t = np.arange(N_OBS + FUTURE)
    mean = np.exp(level[None, :] + weekday[group][:, t % 7].T)
    y = rng.negative_binomial(CONCENTRATION, CONCENTRATION / (CONCENTRATION + mean)).astype(float)
    return {"y": y[:N_OBS], "y_future": y[N_OBS:], "mean": mean, "level": level, "weekday": weekday}


@pytest.fixture(scope="module")
def fitted(panel) -> HierarchicalCount:
    model = HierarchicalCount(
        level=Level(walk=BlockWalk(block=28, n_blocks=4)),
        components=[Seasonal(7, pool_by="group")],
        hierarchy=_hierarchy(),
        inference=SVI(steps=1500, num_draws=200),
    )
    return model.fit(panel["y"], seed=1)


# ---------------------------------------------------------------------------------------------------------------------
# Recovery and the forecast layout
# ---------------------------------------------------------------------------------------------------------------------


class TestPooledPanel:
    def test_recovers_the_level_at_the_origin(self, fitted, panel):
        # The walk is zero on the last block, so the level at the origin is the anchor plus the intercept.
        intercept = fitted.idata_["level/intercept"].mean(("chain", "draw")).values
        error = np.abs(fitted._fitted["offset"] + intercept - panel["level"])
        assert error.mean() < 0.1
        assert error.max() < 0.3

    def test_recovers_the_weekday_effects(self, fitted, panel):
        weekday = fitted.idata_["seasonal/group"].mean(("chain", "draw")).values
        assert weekday.shape == (N_GROUPS, 7)
        np.testing.assert_allclose(weekday, panel["weekday"], atol=0.1)
        assert np.abs(weekday - panel["weekday"]).mean() < 0.03

    def test_posterior_has_chain_draw_and_series_dims(self, fitted):
        idata = fitted.idata_
        assert idata.attrs["inference"] == "SVI"
        assert idata["level/intercept"].dims == ("chain", "draw", "series")
        assert idata["level/walk/step"].dims[-1] == "series"
        assert idata["obs/concentration"].dims == ("chain", "draw", "series")
        assert idata.sizes == {**idata.sizes, "chain": 1, "draw": 200, "series": N_SERIES}

    def test_predict_returns_counts_by_sample_time_series(self, fitted, panel):
        forecast = fitted.predict(horizon=FUTURE, num_draws=300, chunk=64, seed=3)
        draws = forecast["forecast_samples"]
        assert draws.dims == ("sample", "time", "series")
        assert draws.shape == (300, FUTURE, N_SERIES)
        values = draws.values
        assert (values >= 0).all() and np.all(values == np.round(values))
        # The forecast mean follows the simulated one, day by day across the series.
        mean = values.mean(axis=0)
        assert abs(mean.mean() / panel["mean"][N_OBS:].mean() - 1.0) < 0.1
        assert np.corrcoef(mean.ravel(), panel["mean"][N_OBS:].ravel())[0, 1] > 0.9

    def test_draws_go_through_aggregate_draws_and_draw_quantiles(self, fitted):
        forecast = fitted.predict(horizon=FUTURE, num_draws=100)
        hierarchy = _hierarchy()
        by_store = aggregate_draws(forecast, hierarchy, "store")
        assert by_store.dims == ("sample", "time", "group")
        np.testing.assert_allclose(by_store.sum("group"), forecast["forecast_samples"].sum("series"), rtol=1e-6)
        quantiles = draw_quantiles(by_store, [0.1, 0.5, 0.9])
        assert quantiles.dims == ("quantile", "time", "group")
        assert (quantiles.diff("quantile") >= 0).all()
        assert set(aggregate_draws(forecast, hierarchy)) == {"group", "store"}

    def test_predict_is_reproducible_by_seed(self, fitted):
        first = fitted.predict(horizon=5, num_draws=20, seed=7)["forecast_samples"].values
        again = fitted.predict(horizon=5, num_draws=20, seed=7)["forecast_samples"].values
        other = fitted.predict(horizon=5, num_draws=20, seed=8)["forecast_samples"].values
        np.testing.assert_array_equal(first, again)
        assert not np.array_equal(first, other)

    def test_chunking_does_not_change_the_draw_count_or_posterior_order(self, fitted):
        # More draws than the posterior has reuse its draws in turn.
        forecast = fitted.predict(horizon=3, num_draws=450, chunk=100)
        assert forecast.sizes["sample"] == 450

    def test_gated_future_days_forecast_zero(self, fitted):
        gate = np.ones((FUTURE, N_SERIES))
        gate[:, :5] = 0.0
        draws = fitted.predict(gate_future=gate, num_draws=50)["forecast_samples"].values
        assert (draws[:, :, :5] == 0).all()
        assert draws[:, :, 5:].sum() > 0


# ---------------------------------------------------------------------------------------------------------------------
# Every component at once, as in Orduz's M5 model
# ---------------------------------------------------------------------------------------------------------------------


def _calendar(n_days: int, start: int = 0) -> dict[str, np.ndarray]:
    days = pd.date_range("2020-01-01", periods=start + n_days, freq="D")[start:]
    event = np.zeros(n_days, dtype=int)
    event[(days.month == 12) & (days.day == 24)] = 1
    event[days.day == 15] = 2
    return {
        "weekday": days.dayofweek.to_numpy(),
        "day_of_month": days.day.to_numpy(),
        "event": event,
        "snap": np.asarray(days.day <= 10, dtype=float),
    }


def _orduz_like(n_series: int, inference) -> HierarchicalCount:
    meta = pd.DataFrame({"store": np.arange(n_series) % 2, "dept": np.arange(n_series) % 3})
    hierarchy = Hierarchy.from_frame(meta, {"store": ["store"], "dept": ["dept"], "group": ["store", "dept"]})
    return HierarchicalCount(
        level=Level(
            anchor=RecentMean(days=28, fallback_days=56),
            walk=BlockWalk(block=28, n_blocks=3, scale_by="dept"),
            scale_by="dept",
            shift_by="group",
        ),
        components=[
            GroupShocks("store"),
            Seasonal(7, pool_by="group", deviation="dept", index="weekday"),
            DayOfMonth(pool_by="group"),
            Regressor("snap", pool_by="group"),
            Fourier(365.25, 2, pool_by="dept"),
            Events(pool_by="dept"),
        ],
        obs=NegBin2(dispersion_pool_by=hierarchy["group"], deviation_by=hierarchy["dept"]),
        hierarchy=hierarchy,
        inference=inference,
        log_mean_bounds=(-12.0, float(np.log(500.0))),
    )


class TestAllComponents:
    N_SERIES, N_OBS, FUTURE = 12, 400, 10

    @pytest.fixture(scope="class")
    @classmethod
    def model(cls) -> HierarchicalCount:
        rng = np.random.default_rng(1)
        y = rng.poisson(5.0, (cls.N_OBS, cls.N_SERIES)).astype(float)
        gate = (rng.uniform(size=y.shape) > 0.05).astype(float)
        y[rng.uniform(size=y.shape) < 0.01] = np.nan
        model = _orduz_like(cls.N_SERIES, SVI(subsample=5, steps=200, num_draws=30))
        return model.fit(y, _calendar(cls.N_OBS), gate=gate, seed=0)

    def test_every_component_samples_its_sites(self, model):
        sites = set(model.idata_.data_vars)
        expected = {
            "level/intercept_scale",
            "level/intercept",
            "level/shift",
            "level/walk_scale",
            "level/walk/step",
            "shocks/scale",
            "shocks/shock",
            "seasonal/group",
            "seasonal/deviation_scale",
            "seasonal/deviation",
            "day_of_month/group",
            "snap/coef",
            "fourier/weight",
            "events/scale",
            "events/effect",
            "obs/log_concentration_group",
            "obs/deviation_scale",
            "obs/log_concentration",
            "obs/concentration",
        }
        assert expected <= sites
        # The events are sized from the training covariates: two events, plus "none".
        assert model.idata_["events/effect"].shape[-1] == 2
        assert model.idata_["seasonal/deviation"].dims[2] == "series"

    def test_the_series_plate_is_subsampled(self, model):
        assert model.guide_.prototype_trace["series"]["args"] == (self.N_SERIES, 5)
        assert np.isfinite(model.losses_).all()

    def test_predict_reads_the_future_covariates(self, model):
        future = _calendar(self.FUTURE, start=self.N_OBS)
        forecast = model.predict(future, num_draws=40, chunk=16)
        assert forecast["forecast_samples"].shape == (40, self.FUTURE, self.N_SERIES)
        assert np.isfinite(forecast["forecast_samples"].values).all()

    def test_predict_rejects_an_event_unseen_in_training(self, model):
        future = _calendar(self.FUTURE, start=self.N_OBS)
        future["event"] = np.full(self.FUTURE, 3)
        with pytest.raises(ValueError, match="'event' must hold integers in \\[0, 2\\]"):
            model.predict(future)

    def test_predict_needs_the_covariates_of_fit(self, model):
        future = _calendar(self.FUTURE, start=self.N_OBS)
        del future["snap"]
        with pytest.raises(KeyError, match="snap"):
            model.predict(future)
        future = _calendar(self.FUTURE, start=self.N_OBS)
        future["snap"] = np.ones(self.FUTURE + 1)
        with pytest.raises(ValueError, match="different forecast lengths"):
            model.predict(future)


# ---------------------------------------------------------------------------------------------------------------------
# Configuration, inputs and other inference
# ---------------------------------------------------------------------------------------------------------------------


class TestConfiguration:
    def test_default_inference_is_orduz_svi(self):
        model = HierarchicalCount()
        assert isinstance(model.inference, SVI)
        assert model.inference.subsample == DEFAULT_SUBSAMPLE
        assert model.inference.steps == DEFAULT_STEPS
        assert model.inference.schedule == LinearOneCycle(DEFAULT_STEPS)
        assert isinstance(model.obs, NegBin2) and isinstance(model.level, Level)

    def test_default_inference_fits_a_small_panel_whole(self):
        model = HierarchicalCount()
        model._n_fit_series = 40
        assert model._inference_for(None).subsample is None
        assert model.inference.subsample == DEFAULT_SUBSAMPLE
        model._n_fit_series = DEFAULT_SUBSAMPLE + 1
        assert model._inference_for(None).subsample == DEFAULT_SUBSAMPLE

    def test_linear_one_cycle_is_optax(self):
        expected = optax.linear_onecycle_schedule(
            300, 0.03, pct_start=0.2, pct_final=0.8, div_factor=10.0, final_div_factor=10.0
        )
        for step in (0, 60, 150, 240, 299):
            np.testing.assert_allclose(LinearOneCycle(300)(step), expected(step))

    def test_exported_from_bunobee_models(self):
        import bunobee.models

        assert bunobee.models.HierarchicalCount is HierarchicalCount
        with pytest.raises(AttributeError):
            bunobee.models.NoSuchModel

    def test_components_follow_the_protocol(self):
        for component in (Level(), Seasonal(), DayOfMonth(), Events(), Fourier(), GroupShocks("g"), Regressor("x")):
            assert isinstance(component, Component)

    @pytest.mark.parametrize(
        ("make", "match"),
        [
            (lambda: HierarchicalCount(obs=Normal()), "count support"),
            (lambda: HierarchicalCount(components=[Seasonal(), Seasonal()]), "distinct names"),
            (lambda: HierarchicalCount(components=[Regressor("obs")]), "distinct names"),
            (lambda: HierarchicalCount(hierarchy={"group": []}), "Hierarchy"),
            (lambda: HierarchicalCount(log_mean_bounds=(1.0, 0.0)), "log_mean_bounds"),
            (lambda: Seasonal(deviation="group"), "needs pool_by"),
            (lambda: Seasonal(period=1), "period"),
            (lambda: Fourier(period=7, order=4), "order"),
            (lambda: Events(keys=()), "keys"),
            (lambda: Events(n_events=0), "n_events"),
            (lambda: GroupShocks(None), "by"),
            (lambda: BlockWalk(block=0), "block"),
            (lambda: Level(scale=0.0), "scale"),
            (lambda: Regressor("x", scale=-1.0), "scale"),
        ],
    )
    def test_rejects_bad_config(self, make, match):
        with pytest.raises(ValueError, match=match):
            make()

    def test_predict_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            HierarchicalCount().predict(horizon=3)


def _small(**kwargs) -> HierarchicalCount:
    kwargs.setdefault("level", Level(walk=BlockWalk(block=7, n_blocks=2)))
    kwargs.setdefault("inference", MAP(steps=50))
    return HierarchicalCount(**kwargs)


class TestInputs:
    Y = np.random.default_rng(2).poisson(3.0, (35, 6)).astype(float)

    @pytest.mark.parametrize(
        ("y", "match"),
        [
            (np.ones(10), "n_obs, n_series"),
            (-np.ones((10, 3)), "non-negative integer"),
            (np.full((10, 3), 1.5), "non-negative integer"),
            (np.full((10, 3), np.inf), "non-negative integer"),
        ],
    )
    def test_rejects_bad_counts(self, y, match):
        with pytest.raises(ValueError, match=match):
            _small().fit(y)

    def test_rejects_a_bad_gate(self):
        with pytest.raises(ValueError, match="gate"):
            _small().fit(self.Y, gate=np.ones((35, 5)))
        with pytest.raises(ValueError, match="gate"):
            _small().fit(self.Y, gate=np.full(self.Y.shape, 2.0))

    def test_missing_and_bad_covariates(self):
        with pytest.raises(KeyError, match="snap"):
            _small(components=[Regressor("snap")]).fit(self.Y)
        with pytest.raises(ValueError, match="shape"):
            _small(components=[Regressor("snap")]).fit(self.Y, {"snap": np.ones(34)})
        with pytest.raises(ValueError, match=r"\[1, 31\]"):
            _small(components=[DayOfMonth()]).fit(self.Y, {"day_of_month": np.zeros(35)})
        with pytest.raises(ValueError, match="one index per day"):
            _small(components=[Seasonal(index="weekday")]).fit(self.Y, {"weekday": np.zeros((35, 6))})

    def test_pool_by_needs_the_hierarchy(self):
        with pytest.raises(ValueError, match="no hierarchy"):
            _small(components=[Seasonal(pool_by="group")]).fit(self.Y)
        with pytest.raises(KeyError, match="no level 'dept'"):
            _small(components=[Seasonal(pool_by="dept")], hierarchy=_hierarchy(6, 2)).fit(self.Y)
        with pytest.raises(ValueError, match="groups 40 series but y has 6"):
            _small(hierarchy=_hierarchy()).fit(self.Y)

    def test_nan_is_a_gated_day(self):
        y = self.Y.copy()
        y[:7, 0] = np.nan
        gate = np.ones_like(y)
        gate[:7, 0] = 0.0
        y_zero = np.where(np.isnan(y), 0.0, y)
        by_nan = _small().fit(y, seed=3).idata_
        by_gate = _small().fit(y_zero, gate=gate, seed=3).idata_
        xr.testing.assert_allclose(by_nan, by_gate)

    def test_dataframe_columns_label_the_series(self):
        labels = [f"s{i}" for i in range(6)]
        model = _small().fit(pd.DataFrame(self.Y, columns=labels))
        assert list(model.idata_["level/intercept"].series.values) == labels
        forecast = model.predict(horizon=4, num_draws=3)
        assert list(forecast.series.values) == labels

    def test_predict_needs_a_forecast_length(self):
        model = _small().fit(self.Y)
        with pytest.raises(ValueError, match="horizon="):
            model.predict()
        with pytest.raises(ValueError, match="gate_future"):
            model.predict(gate_future=np.ones((4, 5)))

    def test_poisson_and_nuts(self):
        model = _small(obs=Poisson(), inference=NUTS(num_warmup=30, num_samples=20)).fit(self.Y)
        assert model.idata_.attrs["inference"] == "NUTS"
        forecast = model.predict(horizon=4, num_draws=30)
        assert forecast["forecast_samples"].shape == (30, 4, 6)

    def test_level_alone_has_a_time_axis(self):
        model = HierarchicalCount(level=Level(walk=None), inference=MAP(steps=200, peak_lr=0.1)).fit(self.Y)
        forecast = model.predict(horizon=4, num_draws=400)
        assert forecast["forecast_samples"].shape == (400, 4, 6)
        np.testing.assert_allclose(forecast["forecast_samples"].mean(("sample", "time")), self.Y.mean(0), rtol=0.35)
