"""Tests of the model classes: the ``Forecaster`` contract, ``DLT`` and ``SSP``.

The classes are shells over functions that have their own tests, so the main checks here are equivalences: ``DLT``
must reproduce ``run_dlt_model`` / ``make_inference`` exactly, and ``SSP`` must reproduce the NUTS model the
``ssp_05`` / ``ssp_06`` prototypes write by hand. The fits are tiny on purpose: the point is the wiring. The
inference methods themselves are tested in ``test_inference.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
import pytest
import xarray as xr
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS

from bunobee.models.base import Forecaster, NotFittedError
from bunobee.models.inference import MAP, SVI
from bunobee.models.inference import NUTS as NUTSInference
from bunobee.models.dlt import DLT, make_inference, run_dlt_model
from bunobee.models.ssp import SSP, transform_to_ekf_st
from bunobee.models.ssp.forecast import POSITIVITY_FLOOR
from bunobee.models.ssp.kalman_1d_st_ekf import kalman_filter_1d_ekf_st
from bunobee.regression import RegressionScheme

MCMC_ARGS = {"num_warmup": 40, "num_samples": 40, "num_chains": 1, "progress_bar": False}
N_STEPS, N_SERIES, HORIZON = 40, 2, 5
STATES = ["level", "spend_1", "spend_2"]
SERIES = ["north", "south"]
EXPONENT = 1.0


# ---------------------------------------------------------------------------
# Data and priors
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def panel() -> tuple[np.ndarray, np.ndarray]:
    """A small ``(y, Z)`` panel: a level plus two positive spend effects shared by two series."""
    rng = np.random.default_rng(0)
    spend = rng.uniform(0.5, 1.5, size=(N_STEPS + HORIZON, N_SERIES, 2))
    Z = np.concatenate([np.ones((N_STEPS + HORIZON, N_SERIES, 1)), spend], axis=-1)
    y = Z @ np.array([1.0, 0.4, 0.2]) + rng.normal(scale=0.1, size=(N_STEPS + HORIZON, N_SERIES))
    return y, Z


def _prior(family: str = "beta", compressed: bool = True, anchors: bool = True) -> xr.Dataset:
    """Natural-scale prior over ``STATES``: a free level and two positivity spend states."""
    n_states = len(STATES)
    data = {
        "a0": (("state",), np.array([1.0, 0.3, 0.3])),
        "P0": (("state",), np.array([0.5, 0.05, 0.05])),
        "positivity": (("state",), np.array([False, True, True])),
    }
    sigma_dims, k = (("sigma_q",), 2) if compressed else (("state",), n_states)
    if family == "beta":
        data["sigma_q_alpha_prior"] = (sigma_dims, np.full(k, 2.0))
        data["sigma_q_beta_prior"] = (sigma_dims, np.full(k, 10.0))
        data["sigma_q_scale_prior"] = (sigma_dims, np.full(k, 0.05))
    else:
        data["sigma_q_loc_prior"] = (sigma_dims, np.full(k, 0.01))
        data["sigma_q_scale_prior"] = (sigma_dims, np.full(k, 0.01))
    coords = {"state": STATES, "series": SERIES}
    if compressed:
        coords["sigma_q"] = ["level", "spend"]
    if anchors:
        data["a_obs"] = (("time", "state"), np.zeros((N_STEPS, n_states)))
        data["P_obs"] = (("time", "state"), np.full((N_STEPS, n_states), np.inf))
        coords["time"] = pd.date_range("2026-01-01", periods=N_STEPS)
    return xr.Dataset(data, coords=coords).assign_attrs(sigma_q_family=family)


@pytest.fixture(scope="module")
def ekf_fit(panel) -> SSP:
    y, Z = panel
    return SSP(_prior(), filter="ekf", exponent=EXPONENT).fit(y[:N_STEPS], Z[:N_STEPS], mcmc_args=MCMC_ARGS)


@pytest.fixture(scope="module")
def linear_fit(panel) -> SSP:
    y, Z = panel
    return SSP(_prior(), filter="linear").fit(y[:N_STEPS], Z[:N_STEPS], mcmc_args=MCMC_ARGS)


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------


class TestForecaster:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            Forecaster()

    @pytest.mark.parametrize("make", [lambda: DLT(0.5, 0.5, 0.8), lambda: SSP(_prior())])
    def test_posterior_before_fit_raises(self, make):
        model = make()
        assert not model.is_fitted
        with pytest.raises(NotFittedError, match="fit"):
            model.idata_
        with pytest.raises(NotFittedError, match="fit"):
            model.mcmc_

    def test_failed_refit_keeps_the_previous_fit(self, panel, monkeypatch):
        y, Z = panel
        model = SSP(_prior(), exponent=EXPONENT).fit(y[:N_STEPS], Z[:N_STEPS], mcmc_args=MCMC_ARGS)
        before = model.idata_

        def _broken(*args, **kwargs):
            raise RuntimeError("packaging failed")

        monkeypatch.setattr(model, "_to_idata", _broken)
        with pytest.raises(RuntimeError, match="packaging failed"):
            model.fit(y[:N_STEPS], Z[:N_STEPS], mcmc_args=MCMC_ARGS, seed=1)
        assert model.idata_ is before


# ---------------------------------------------------------------------------
# DLT
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dlt_y() -> np.ndarray:
    rng = np.random.default_rng(42)
    return np.linspace(10.0, 15.0, 20) + rng.normal(scale=0.3, size=20)


@pytest.fixture(scope="module")
def dlt_regression(dlt_y):
    rng = np.random.default_rng(3)
    cov = pd.DataFrame(np.abs(rng.normal(size=(20, 2))), columns=["s1", "ctrl"])
    scheme = RegressionScheme()
    scheme.add_regressors(["s1", "ctrl"], coef_sign=["+", "="], loc_prior=0.0, scale_prior=1.0)
    return dlt_y + 0.5 * cov["s1"].to_numpy(), cov, scheme


class TestDLT:
    def test_fit_matches_run_dlt_model(self, dlt_y):
        model = DLT(0.5, 0.5, 0.8).fit(dlt_y, mcmc_args=MCMC_ARGS, seed=3)
        expected = run_dlt_model(jax.random.PRNGKey(3), 0.5, 0.5, 0.8, dlt_y, mcmc_run_args=MCMC_ARGS)
        xr.testing.assert_identical(model.idata_, expected.assign_attrs(inference="NUTS"))

    def test_nuts_inference_matches_mcmc_args(self, dlt_y):
        by_args = DLT(0.5, 0.5, 0.8).fit(dlt_y, mcmc_args=MCMC_ARGS, seed=3)
        by_inference = DLT(0.5, 0.5, 0.8, inference=NUTSInference(**MCMC_ARGS)).fit(dlt_y, seed=3)
        xr.testing.assert_identical(by_inference.idata_, by_args.idata_)

    @pytest.mark.parametrize("inference", [MAP(steps=200), SVI(steps=200, num_draws=20)], ids=["map", "svi"])
    def test_svi_and_map_fit_and_predict(self, dlt_y, inference):
        model = DLT(0.5, 0.5, 0.8, inference=inference).fit(dlt_y)
        assert model.idata_["yhat"].dims == ("chain", "draw", "time")
        assert model.idata_.sizes["chain"] == 1
        assert model.idata_.attrs["inference"] == inference.name
        forecast = model.predict(end_step=len(dlt_y) + 5)
        assert forecast["forecast_samples"].shape == (inference.num_draws, len(dlt_y) + 5)

    def test_fit_with_regression_matches_run_dlt_model(self, dlt_regression):
        y, cov, scheme = dlt_regression
        model = DLT(0.5, 0.5, 0.8, regression_scheme=scheme, noise_dist="student_t")
        model.fit(y, cov, mcmc_args=MCMC_ARGS)
        expected = run_dlt_model(
            jax.random.PRNGKey(0),
            0.5,
            0.5,
            0.8,
            y,
            mcmc_run_args=MCMC_ARGS,
            regression_scheme=scheme,
            covariates_df=cov,
            noise_dist="student_t",
        )
        xr.testing.assert_identical(model.idata_, expected.assign_attrs(inference="NUTS"))

    def test_predict_matches_make_inference(self, dlt_y):
        model = DLT(0.5, 0.5, 0.8).fit(dlt_y, mcmc_args=MCMC_ARGS)
        forecast = model.predict(end_step=len(dlt_y) + 5, noise_embed=True, seed=7)
        expected = make_inference(
            jax.random.PRNGKey(7), model.idata_, 0.5, 0.5, 0.8, end_step=len(dlt_y) + 5, noise_embed=True
        )
        xr.testing.assert_identical(forecast, expected)
        assert forecast["forecast_samples"].shape == (MCMC_ARGS["num_samples"], len(dlt_y) + 5)

    def test_regression_needs_covariates(self, dlt_regression):
        y, _, scheme = dlt_regression
        with pytest.raises(ValueError, match="covariates_df is required"):
            DLT(0.5, 0.5, 0.8, regression_scheme=scheme).fit(y)

    def test_covariates_need_a_regression_scheme(self, dlt_regression):
        y, cov, _ = dlt_regression
        with pytest.raises(ValueError, match="no regression_scheme"):
            DLT(0.5, 0.5, 0.8).fit(y, cov)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"noise_dist": "cauchy"}, "noise_dist"),
            ({"noise_dist": "student_t", "min_nu": 10.0, "max_nu": 5.0}, "min_nu"),
            ({"pos_coef_sum_scale": 0.0}, "positive"),
        ],
    )
    def test_rejects_bad_config_at_construction(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            DLT(0.5, 0.5, 0.8, **kwargs)


# ---------------------------------------------------------------------------
# SSP
# ---------------------------------------------------------------------------


def _hand_written_ekf_idata(prior: xr.Dataset, y: np.ndarray, Z: np.ndarray) -> dict:
    """The ``ssp_06`` NUTS model, written out by hand as the prototypes do."""
    prior_ekf = transform_to_ekf_st(prior.assign(P0=(("state", "state_dual"), np.diag(prior["P0"].values))), EXPONENT)
    a0 = jnp.asarray(prior_ekf["a0"].values)
    P0 = jnp.asarray(prior_ekf["P0"].values)
    a_obs = jnp.asarray(prior_ekf["a_obs"].values)
    P_obs = jnp.asarray(prior_ekf["P_obs"].values)
    positivity = jnp.asarray(prior_ekf["positivity"].values).astype(bool)
    sdy = jnp.asarray(np.std(y, axis=0))
    alpha = jnp.asarray(prior_ekf["sigma_q_alpha_prior"].values)
    beta = jnp.asarray(prior_ekf["sigma_q_beta_prior"].values)
    scale = jnp.asarray(prior_ekf["sigma_q_scale_prior"].values)
    n_states = a0.shape[0]

    def nuts_fn():
        with numpyro.plate("series", y.shape[1]):
            sigma_h_raw = numpyro.sample("sigma_h_raw", dist.Beta(2.0, 10.0))
        sigma_h = numpyro.deterministic("sigma_h", sigma_h_raw * sdy)
        sigma_q_compressed = numpyro.sample("sigma_q_raw", dist.Beta(alpha, beta)) * scale
        sigma_q = numpyro.deterministic(
            "sigma_q", jnp.concatenate([sigma_q_compressed[:1], jnp.full(n_states - 1, sigma_q_compressed[1])])
        )
        lp, at, *_ = kalman_filter_1d_ekf_st(
            a0=a0,
            P0=P0,
            Z=jnp.asarray(Z),
            sigma_h=sigma_h,
            sigma_q=sigma_q,
            y=jnp.asarray(y),
            logp=True,
            exponent=EXPONENT,
            positivity=positivity,
            a_obs=a_obs,
            P_obs=P_obs,
        )
        numpyro.factor("lp", lp)
        numpyro.deterministic("at", at)

    mcmc = MCMC(NUTS(nuts_fn), **MCMC_ARGS)
    mcmc.run(jax.random.PRNGKey(0))
    return mcmc.get_samples(group_by_chain=True)


class TestSSPFit:
    def test_matches_the_hand_written_prototype_model(self, panel, ekf_fit):
        y, Z = panel
        expected = _hand_written_ekf_idata(_prior(), y[:N_STEPS], Z[:N_STEPS])
        for name in ("sigma_h", "sigma_q", "at"):
            np.testing.assert_allclose(ekf_fit.idata_[name].values, np.asarray(expected[name]), rtol=1e-5)

    def test_posterior_layout(self, ekf_fit):
        idata = ekf_fit.idata_
        assert idata["at"].dims == ("chain", "draw", "time", "state")
        assert idata["at_nat"].dims == ("chain", "draw", "time", "state")
        assert idata["mu"].dims == ("chain", "draw", "time", "series")
        assert idata["sigma_h"].dims == ("chain", "draw", "series")
        assert idata["sigma_q_raw"].dims == ("chain", "draw", "sigma_q_group")
        assert list(idata["state"].values) == STATES
        assert list(idata["series"].values) == SERIES
        assert list(idata["sigma_q_group"].values) == ["level", "spend"]
        assert len(idata["time"]) == N_STEPS
        assert idata.attrs == {
            "exponent": EXPONENT,
            "link": "exp",
            "filter": "kalman_filter_1d_ekf_st",
            "inference": "NUTS",
        }
        np.testing.assert_array_equal(idata["positivity"].values, [False, True, True])

    def test_compressed_sigma_q_is_shared_by_the_later_states(self, ekf_fit):
        sigma_q = ekf_fit.idata_["sigma_q"].values
        np.testing.assert_array_equal(sigma_q[..., 1], sigma_q[..., 2])
        assert not np.allclose(sigma_q[..., 0], sigma_q[..., 1])

    def test_natural_states_are_exp_of_the_positivity_states(self, ekf_fit):
        idata = ekf_fit.idata_
        at, at_nat = idata["at"].values, idata["at_nat"].values
        np.testing.assert_allclose(at_nat[..., 0], at[..., 0])
        np.testing.assert_allclose(at_nat[..., 1:], np.exp(EXPONENT * at[..., 1:]), rtol=1e-6)

    def test_mu_is_the_design_times_the_natural_states(self, panel, ekf_fit):
        _, Z = panel
        idata = ekf_fit.idata_
        expected = np.einsum("tjs,cdts->cdtj", Z[:N_STEPS], idata["at_nat"].values)
        np.testing.assert_allclose(idata["mu"].values, expected, rtol=1e-6)

    def test_linear_filter_floors_the_positivity_states(self, linear_fit):
        idata = linear_fit.idata_
        assert idata.attrs["link"] == "identity"
        assert idata.attrs["filter"] == "kalman_filter_1d_st"
        assert "at_nat" not in idata
        assert (idata["at"].values[..., 1:] >= POSITIVITY_FLOOR).all()

    def test_truncated_normal_sigma_q_per_state(self, panel):
        y, Z = panel
        model = SSP(_prior("truncated_normal", compressed=False), exponent=EXPONENT)
        idata = model.fit(y[:N_STEPS], Z[:N_STEPS], mcmc_args=MCMC_ARGS).idata_
        assert idata["sigma_q_raw"].dims == ("chain", "draw", "state")
        assert (idata["sigma_q"].values > 0).all()

    def test_missing_observations_are_predict_only_steps(self, panel):
        y, Z = panel
        y_gappy = y[:N_STEPS].copy()
        y_gappy[10:15, 0] = np.nan
        idata = SSP(_prior(), exponent=EXPONENT).fit(y_gappy, Z[:N_STEPS], mcmc_args=MCMC_ARGS).idata_
        assert np.isfinite(idata["at"].values).all()
        assert np.isfinite(idata["sigma_h"].values).all()

    def test_one_series(self, panel):
        y, Z = panel
        prior = _prior(anchors=False).drop_vars("series")
        model = SSP(prior, exponent=EXPONENT).fit(y[:N_STEPS, 0], Z[:N_STEPS, 0], mcmc_args=MCMC_ARGS)
        assert model.idata_.sizes["series"] == 1
        forecast = model.predict(Z[N_STEPS:, 0])
        assert forecast["forecast_samples"].shape == (MCMC_ARGS["num_samples"], HORIZON, 1)


class TestSSPPredict:
    @pytest.mark.parametrize("fit_name", ["ekf_fit", "linear_fit"])
    def test_matches_forecast_ssp_on_the_posterior(self, request, panel, fit_name):
        from bunobee.models.ssp import forecast_ssp

        model = request.getfixturevalue(fit_name)
        _, Z = panel
        forecast = model.predict(Z[N_STEPS:], noise_embed=True, seed=5)
        expected = forecast_ssp(model.idata_, Z[N_STEPS:], noise_embed=True, seed=5)
        xr.testing.assert_identical(forecast, expected)
        assert forecast["forecast_samples"].shape == (MCMC_ARGS["num_samples"], HORIZON, N_SERIES)

    def test_predict_before_fit_raises(self, panel):
        _, Z = panel
        with pytest.raises(NotFittedError):
            SSP(_prior()).predict(Z[N_STEPS:])


class TestSSPInference:
    def test_nuts_inference_matches_mcmc_args(self, panel, ekf_fit):
        y, Z = panel
        model = SSP(_prior(), exponent=EXPONENT, inference=NUTSInference(**MCMC_ARGS)).fit(y[:N_STEPS], Z[:N_STEPS])
        xr.testing.assert_identical(model.idata_, ekf_fit.idata_)

    @pytest.mark.parametrize("inference", [MAP(steps=200), SVI(steps=200, num_draws=20)], ids=["map", "svi"])
    @pytest.mark.parametrize("filter", ["ekf", "linear"])
    def test_svi_and_map_fit_and_predict(self, panel, inference, filter):
        y, Z = panel
        model = SSP(_prior(), filter=filter, exponent=EXPONENT, inference=inference).fit(y[:N_STEPS], Z[:N_STEPS])
        idata = model.idata_
        assert idata["at"].dims == ("chain", "draw", "time", "state")
        assert idata["sigma_h"].dims == ("chain", "draw", "series")
        assert (idata.sizes["chain"], idata.sizes["draw"]) == (1, inference.num_draws)
        assert idata.attrs["inference"] == inference.name
        assert np.isfinite(idata["at"].values).all()
        forecast = model.predict(Z[N_STEPS:], noise_embed=True)
        assert forecast["forecast_samples"].shape == (inference.num_draws, HORIZON, N_SERIES)
        assert np.isfinite(forecast["forecast_samples"].values).all()

    def test_svi_with_subsample_raises_at_construction(self):
        with pytest.raises(ValueError, match="no series plate"):
            SSP(_prior(), inference=SVI(subsample=1))

    def test_mcmc_args_under_svi_raise(self, panel):
        y, Z = panel
        with pytest.raises(ValueError, match="only applies to NUTS"):
            SSP(_prior(), inference=MAP(steps=10)).fit(y[:N_STEPS], Z[:N_STEPS], mcmc_args=MCMC_ARGS)

    @pytest.mark.parametrize(
        "make", [lambda inf: SSP(_prior(), inference=inf), lambda inf: DLT(0.5, 0.5, 0.8, inference=inf)]
    )
    def test_unaccepted_method_raises_at_construction(self, make):
        with pytest.raises(ValueError, match="accepts inference by"):
            make(object())


class TestSSPValidation:
    def test_unknown_filter(self):
        with pytest.raises(ValueError, match="filter must be one of"):
            SSP(_prior(), filter="particle")

    def test_prior_without_a_sigma_q_block(self):
        prior = _prior().drop_vars(["sigma_q_alpha_prior", "sigma_q_beta_prior", "sigma_q_scale_prior"])
        with pytest.raises(ValueError, match="sigma_q"):
            SSP(prior)

    def test_sigma_q_block_of_the_wrong_length(self):
        prior = _prior(compressed=False).isel(state=slice(0, 3))
        prior = prior.assign(sigma_q_scale_prior=(("extra",), np.full(4, 0.05)))
        prior = prior.assign(sigma_q_alpha_prior=(("extra",), np.full(4, 2.0)))
        prior = prior.assign(sigma_q_beta_prior=(("extra",), np.full(4, 10.0)))
        with pytest.raises(ValueError, match="shape"):
            SSP(prior, filter="linear")

    def test_design_with_the_wrong_number_of_states(self, panel):
        y, Z = panel
        with pytest.raises(ValueError, match="states"):
            SSP(_prior()).fit(y[:N_STEPS], Z[:N_STEPS, :, :2])

    def test_design_that_does_not_match_y(self, panel):
        y, Z = panel
        with pytest.raises(ValueError, match="matching y"):
            SSP(_prior()).fit(y[:N_STEPS], Z[: N_STEPS - 1])

    def test_anchors_that_do_not_cover_y(self, panel):
        y, Z = panel
        with pytest.raises(ValueError, match="a_obs"):
            SSP(_prior()).fit(y[: N_STEPS - 1], Z[: N_STEPS - 1])

    def test_constant_series(self, panel):
        y, Z = panel
        y_flat = y[:N_STEPS].copy()
        y_flat[:, 1] = 3.0
        with pytest.raises(ValueError, match="constant"):
            SSP(_prior()).fit(y_flat, Z[:N_STEPS])

    def test_non_finite_design_where_y_is_observed(self, panel):
        y, Z = panel
        Z_bad = Z[:N_STEPS].copy()
        Z_bad[3, 0, 1] = np.nan
        with pytest.raises(ValueError, match="finite"):
            SSP(_prior()).fit(y[:N_STEPS], Z_bad)
