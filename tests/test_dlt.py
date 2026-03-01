import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

jax.config.update("jax_enable_x64", True)

from wunkui.models.dlt import dlt_transition_step, run_dlt_model, make_inference

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = jax.random.PRNGKey(0)
LEV_SM = 0.5
SLP_SM = 0.5
THETA = 0.8
N_TRAIN = 20
MCMC_ARGS = {"num_warmup": 50, "num_samples": 50, "num_chains": 1, "progress_bar": False}


@pytest.fixture(scope="module")
def synthetic_y():
    rng = np.random.default_rng(42)
    trend = np.linspace(10.0, 15.0, N_TRAIN)
    noise = rng.normal(scale=0.3, size=N_TRAIN)
    return (trend + noise).astype(np.float64)


@pytest.fixture(scope="module")
def trained_posteriors(synthetic_y):
    return run_dlt_model(RNG, LEV_SM, SLP_SM, THETA, synthetic_y, mcmc_run_args=MCMC_ARGS)


# ---------------------------------------------------------------------------
# Part 1: dlt_transition_step
# ---------------------------------------------------------------------------


class TestDltTransitionStep:
    def _step(self, carry, y_t, eps_t=0.0, lev_sm=LEV_SM, slp_sm=SLP_SM, theta=THETA):
        inputs = (jnp.float64(y_t), jnp.float64(eps_t))
        return dlt_transition_step(carry, inputs, lev_sm, slp_sm, theta)

    def test_forecast_computation(self):
        """dlt_comp_t = lev_prev + theta * slp_prev"""
        carry = (jnp.float64(10.0), jnp.float64(2.0))
        _, (_, _, dlt_comp_t) = self._step(carry, y_t=12.0, theta=0.8)
        assert jnp.allclose(dlt_comp_t, 11.6, atol=1e-10)

    def test_level_update_finite_y(self):
        """new_lev = lev_sm * y_t + (1 - lev_sm) * dlt_comp_t"""
        carry = (jnp.float64(10.0), jnp.float64(2.0))
        (new_lev, _), _ = self._step(carry, y_t=12.0, lev_sm=0.5, theta=0.8)
        # dlt_comp_t = 11.6; new_lev = 0.5 * 12.0 + 0.5 * 11.6 = 11.8
        assert jnp.allclose(new_lev, 11.8, atol=1e-10)

    def test_slope_update_finite_y(self):
        """new_slp = slp_sm * (new_lev - lev_prev) + (1 - slp_sm) * slp_prev"""
        carry = (jnp.float64(10.0), jnp.float64(2.0))
        (new_lev, new_slp), _ = self._step(carry, y_t=12.0, lev_sm=0.5, slp_sm=0.5, theta=0.8)
        # new_lev=11.8; new_slp = 0.5*(11.8-10.0) + 0.5*2.0 = 0.9 + 1.0 = 1.9
        assert jnp.allclose(new_slp, 1.9, atol=1e-10)

    def test_nan_y_oos_uses_forecast_as_obs(self):
        """NaN y_t (OOS) → substituted with dlt_comp_t + eps_t, states update from synthetic obs."""
        carry = (jnp.float64(10.0), jnp.float64(2.0))
        # eps_t=0 → synthetic y_t = dlt_comp_t = 11.6
        (new_lev, new_slp), (_, _, dlt_comp_t) = self._step(carry, y_t=jnp.nan, eps_t=0.0, theta=0.8)
        assert jnp.allclose(dlt_comp_t, 11.6, atol=1e-10), "forecast should be lev + theta * slp"
        # synthetic y_t == dlt_comp_t, so new_lev = lev_sm*11.6 + (1-lev_sm)*11.6 = 11.6 regardless of lev_sm
        assert jnp.allclose(new_lev, 11.6, atol=1e-10), "level should update from synthetic obs"
        # new_slp = slp_sm*(11.6 - 10.0) + (1-slp_sm)*2.0 = 0.5*1.6 + 0.5*2.0 = 1.8
        assert jnp.allclose(new_slp, 1.8, atol=1e-10), "slope should update from synthetic obs"

    def test_lev_sm_one_new_lev_equals_y(self):
        """lev_sm=1.0 → new_lev == y_t exactly."""
        carry = (jnp.float64(10.0), jnp.float64(2.0))
        (new_lev, _), _ = self._step(carry, y_t=99.0, lev_sm=1.0)
        assert jnp.allclose(new_lev, 99.0, atol=1e-10)

    def test_lev_sm_zero_new_lev_equals_forecast(self):
        """lev_sm=0.0 → new_lev == dlt_comp_t (pure forecast, ignores y_t)."""
        carry = (jnp.float64(10.0), jnp.float64(2.0))
        (new_lev, _), (_, _, dlt_comp_t) = self._step(carry, y_t=99.0, lev_sm=0.0, theta=0.8)
        assert jnp.allclose(new_lev, dlt_comp_t, atol=1e-10)

    def test_multistep_scan_matches_manual(self):
        """lax.scan output matches two manual step applications."""
        carry0 = (jnp.float64(10.0), jnp.float64(0.0))
        y_seq = jnp.array([11.0, 12.0])
        eps_seq = jnp.zeros(2)

        # Manual
        carry1, out1 = dlt_transition_step(carry0, (y_seq[0], eps_seq[0]), LEV_SM, SLP_SM, THETA)
        carry2, out2 = dlt_transition_step(carry1, (y_seq[1], eps_seq[1]), LEV_SM, SLP_SM, THETA)

        # Via lax.scan
        _, (levs, slps, comps) = lax.scan(
            lambda c, inp: dlt_transition_step(c, inp, LEV_SM, SLP_SM, THETA),
            carry0,
            (y_seq, eps_seq),
        )

        assert jnp.allclose(levs[0], out1[0], atol=1e-10)
        assert jnp.allclose(levs[1], out2[0], atol=1e-10)
        assert jnp.allclose(comps[0], out1[2], atol=1e-10)
        assert jnp.allclose(comps[1], out2[2], atol=1e-10)


# ---------------------------------------------------------------------------
# Part 2: run_dlt_model
# ---------------------------------------------------------------------------


class TestRunDltModel:
    def test_output_is_xr_dataset(self, trained_posteriors):
        import xarray as xr

        assert isinstance(trained_posteriors, xr.Dataset)

    def test_expected_variables_present(self, trained_posteriors):
        for var in ["dlt_comp", "yhat", "sigma", "last_lev", "last_slp", "resid"]:
            assert var in trained_posteriors, f"missing variable: {var}"

    def test_dlt_comp_shape(self, trained_posteriors):
        # (n_chains=1, n_draws=50, n_time=20)
        assert trained_posteriors["dlt_comp"].shape == (1, 50, N_TRAIN)

    def test_sigma_shape(self, trained_posteriors):
        assert trained_posteriors["sigma"].shape == (1, 50)

    def test_last_lev_last_slp_shape(self, trained_posteriors):
        assert trained_posteriors["last_lev"].shape == (1, 50)
        assert trained_posteriors["last_slp"].shape == (1, 50)

    def test_yhat_p50_close_to_trend(self, trained_posteriors, synthetic_y):
        mae = float(jnp.mean(jnp.abs(jnp.array(trained_posteriors["yhat_p50"].values) - synthetic_y)))
        assert mae < 1.0, f"yhat_p50 MAE too large: {mae:.3f}"

    def test_resid_p50_near_zero_mean(self, trained_posteriors):
        mean_resid = float(jnp.abs(jnp.mean(jnp.array(trained_posteriors["resid_p50"].values))))
        assert mean_resid < 1.0, f"resid_p50 mean not near zero: {mean_resid:.3f}"


# ---------------------------------------------------------------------------
# Part 3: make_inference
# ---------------------------------------------------------------------------


class TestMakeInference:
    def _n_samples(self, trained_posteriors):
        return int(np.prod(trained_posteriors["sigma"].shape))  # chains * draws

    def test_is_only_shape(self, trained_posteriors):
        result = make_inference(RNG, trained_posteriors, LEV_SM, SLP_SM, THETA, end_step=N_TRAIN)
        n_samples = self._n_samples(trained_posteriors)
        assert result["forecast_samples"].shape == (n_samples, N_TRAIN)

    def test_oos_shape(self, trained_posteriors):
        end_step = N_TRAIN + 10
        result = make_inference(RNG, trained_posteriors, LEV_SM, SLP_SM, THETA, end_step=end_step)
        n_samples = self._n_samples(trained_posteriors)
        assert result["forecast_samples"].shape == (n_samples, end_step)
        assert result["dlt_comp_samples"].shape == (n_samples, end_step)

    def test_missing_end_step_raises(self, trained_posteriors):
        with pytest.raises(ValueError):
            make_inference(RNG, trained_posteriors, LEV_SM, SLP_SM, THETA)

    def test_transform_callback_applied(self, trained_posteriors):
        result = make_inference(
            RNG, trained_posteriors, LEV_SM, SLP_SM, THETA, end_step=N_TRAIN, transform_callback=jnp.exp
        )
        assert float(jnp.min(jnp.array(result["forecast_samples"].values))) > 0.0
