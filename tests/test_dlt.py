import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from jax import lax

jax.config.update("jax_enable_x64", True)

from bunobee.models.dlt import (
    dlt_transition_step,
    generate_dlt_components,
    generate_noise_components,
    get_coef_sign_fraction,
    get_forecast_decomposition,
    get_regression_summary,
    make_inference,
    run_dlt_model,
)
from bunobee.regression import RegressionScheme

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


class TestNoFutureState:
    """#78: the predict path only uses the train-end state and noise draws, never states beyond it."""

    N_SAMPLES, H, SIGMA, LAST_LEV, LAST_SLP = 4000, 8, 1.0, 10.0, 0.5

    def _idata(self):
        n = self.N_SAMPLES
        return xr.Dataset(
            {
                "dlt_comp": (["chain", "draw", "time"], np.zeros((1, n, N_TRAIN))),
                "sigma": (["chain", "draw"], np.full((1, n), self.SIGMA)),
                "last_lev": (["chain", "draw"], np.full((1, n), self.LAST_LEV)),
                "last_slp": (["chain", "draw"], np.full((1, n), self.LAST_SLP)),
            }
        )

    def test_oos_states_depend_only_on_train_end_state_and_noise(self):
        idata = self._idata()
        comp = np.asarray(generate_dlt_components(RNG, idata, N_TRAIN + self.H, LEV_SM, SLP_SM, THETA))[:, N_TRAIN:]
        eps = np.asarray(generate_noise_components(RNG, idata, self.H))

        lev = np.full(self.N_SAMPLES, self.LAST_LEV)
        slp = np.full(self.N_SAMPLES, self.LAST_SLP)
        expected = np.empty_like(eps)
        for t in range(self.H):
            expected[:, t] = lev + THETA * slp
            new_lev = LEV_SM * (expected[:, t] + eps[:, t]) + (1 - LEV_SM) * expected[:, t]
            slp = SLP_SM * (new_lev - lev) + (1 - SLP_SM) * slp
            lev = new_lev

        np.testing.assert_allclose(comp, expected, rtol=1e-10)
        np.testing.assert_allclose(comp[:, 0], self.LAST_LEV + THETA * self.LAST_SLP)

    def test_noise_embed_spread_matches_true_process(self):
        idata = self._idata()
        out = make_inference(RNG, idata, LEV_SM, SLP_SM, THETA, end_step=N_TRAIN + self.H, noise_embed=True)
        forecast = np.asarray(out["forecast_samples"])[:, N_TRAIN:]

        # simulate the true generative process from the same origin state
        rng = np.random.default_rng(1)
        lev = np.full(self.N_SAMPLES, self.LAST_LEV)
        slp = np.full(self.N_SAMPLES, self.LAST_SLP)
        truth = np.empty((self.N_SAMPLES, self.H))
        for t in range(self.H):
            comp = lev + THETA * slp
            truth[:, t] = comp + rng.normal(scale=self.SIGMA, size=self.N_SAMPLES)
            new_lev = LEV_SM * truth[:, t] + (1 - LEV_SM) * comp
            slp = SLP_SM * (new_lev - lev) + (1 - SLP_SM) * slp
            lev = new_lev

        np.testing.assert_allclose(forecast.std(axis=0), truth.std(axis=0), rtol=0.1)


class TestConditionalResponse:
    """#79: actuals beyond the train end update the latent states in make_inference."""

    H = 6

    def _idata(self, with_coef=False):
        n = TestNoFutureState.N_SAMPLES
        data = {
            "dlt_comp": (["chain", "draw", "time"], np.zeros((1, n, N_TRAIN))),
            "sigma": (["chain", "draw"], np.full((1, n), 1.0)),
            "last_lev": (["chain", "draw"], np.full((1, n), 10.0)),
            "last_slp": (["chain", "draw"], np.full((1, n), 0.5)),
        }
        coords = {}
        if with_coef:
            data["coef"] = (["chain", "draw", "var_name"], np.full((1, n, 1), 2.0))
            coords["var_name"] = ["x"]
        return xr.Dataset(data, coords=coords)

    def _reference(self, actuals, reg=0.0):
        """Deterministic numpy recursion with NaN-free actuals, from the train-end state."""
        lev, slp = 10.0, 0.5
        comps = []
        for a in actuals:
            comp = lev + THETA * slp
            comps.append(comp)
            new_lev = LEV_SM * (a - reg) + (1 - LEV_SM) * comp
            slp = SLP_SM * (new_lev - lev) + (1 - SLP_SM) * slp
            lev = new_lev
        return np.array(comps)

    def _response(self, actuals):
        return np.concatenate([np.full(N_TRAIN, 99.0), actuals])  # train span must be ignored

    def test_actuals_update_states(self):
        actuals = np.array([14.0, 15.0, 13.0, 16.0, 17.0, 15.5])
        out = make_inference(
            RNG, self._idata(), LEV_SM, SLP_SM, THETA, end_step=N_TRAIN + self.H,
            conditional_response=self._response(actuals),
        )  # fmt: skip
        comp = np.asarray(out["dlt_comp_samples"])[:, N_TRAIN:]
        np.testing.assert_allclose(comp, np.tile(self._reference(actuals), (comp.shape[0], 1)), rtol=1e-10)

    def test_all_nan_equals_unconditioned(self):
        kwargs = dict(end_step=N_TRAIN + self.H)
        base = make_inference(RNG, self._idata(), LEV_SM, SLP_SM, THETA, **kwargs)
        cond = make_inference(
            RNG, self._idata(), LEV_SM, SLP_SM, THETA, conditional_response=np.full(N_TRAIN + self.H, np.nan), **kwargs
        )
        np.testing.assert_allclose(cond["dlt_comp_samples"].values, base["dlt_comp_samples"].values)

    def test_partial_nan_forecasts_after_actuals(self):
        actuals = np.array([14.0, 15.0, np.nan, np.nan, np.nan, np.nan])
        out = make_inference(
            RNG, self._idata(), LEV_SM, SLP_SM, THETA, end_step=N_TRAIN + self.H,
            conditional_response=self._response(actuals),
        )  # fmt: skip
        comp = np.asarray(out["dlt_comp_samples"])[:, N_TRAIN:]
        # steps 0-2 are deterministic given the actuals (step 2 sees both); later steps spread out via noise
        np.testing.assert_allclose(comp[:, :3].std(axis=0), 0.0, atol=1e-10)
        expected = self._reference([14.0, 15.0, 0.0])[:3]  # the third actual is never seen by comp[2]
        np.testing.assert_allclose(comp[:, :3], np.tile(expected, (comp.shape[0], 1)), rtol=1e-10)
        assert comp[:, 4].std() > 0.0

    def test_regression_removed_before_update(self):
        n_total = N_TRAIN + self.H
        idata = self._idata(with_coef=True)
        cov = pd.DataFrame({"x": np.full(n_total, 3.0)})  # reg_comp = 2 * 3 = 6 at every step
        actuals = np.array([20.0, 21.0, 19.0, 22.0, 23.0, 21.5])
        out = make_inference(
            RNG, idata, LEV_SM, SLP_SM, THETA, covariates_df=cov, conditional_response=self._response(actuals)
        )
        comp = np.asarray(out["dlt_comp_samples"])[:, N_TRAIN:]
        np.testing.assert_allclose(comp, np.tile(self._reference(actuals, reg=6.0), (comp.shape[0], 1)), rtol=1e-10)

    @pytest.mark.parametrize(
        "response",
        [np.zeros(N_TRAIN + 5), np.zeros((N_TRAIN + 6, 1))],
        ids=["wrong_length", "not_1d"],
    )
    def test_bad_shape_raises(self, response):
        with pytest.raises(ValueError):
            make_inference(
                RNG, self._idata(), LEV_SM, SLP_SM, THETA, end_step=N_TRAIN + self.H, conditional_response=response
            )

    def test_no_steps_beyond_train_raises(self):
        with pytest.raises(ValueError, match="beyond the train end"):
            make_inference(
                RNG, self._idata(), LEV_SM, SLP_SM, THETA, end_step=N_TRAIN, conditional_response=np.zeros(N_TRAIN)
            )


class TestRegressionSummary:
    """#83: coefficient sign fraction and regression summary dataframe."""

    def _idata(self):
        # x1 all positive, x2 three negative / one zero, x3 split 2 positive / 2 negative
        coef = np.array([[[1.0, -1.0, 2.0], [2.0, -2.0, 1.0], [3.0, -3.0, -1.0], [4.0, 0.0, -2.0]]])
        return xr.Dataset(
            {"coef": (["chain", "draw", "var_name"], coef)},
            coords={"var_name": ["x1", "x2", "x3"]},
        )

    def test_sign_fraction(self):
        out = get_coef_sign_fraction(self._idata())
        assert out.index.name == "regressor"
        assert out.index.tolist() == ["x1", "x2", "x3"]
        np.testing.assert_allclose(out["frac_pos"], [1.0, 0.0, 0.5])
        np.testing.assert_allclose(out["frac_neg"], [0.0, 0.75, 0.5])

    def test_sign_fraction_pools_chains(self):
        idata = xr.Dataset(
            {"coef": (["chain", "draw", "var_name"], np.array([[[1.0], [1.0]], [[-1.0], [1.0]]]))},
            coords={"var_name": ["x"]},
        )
        out = get_coef_sign_fraction(idata)
        assert out.loc["x", "frac_pos"] == 0.75
        assert out.loc["x", "frac_neg"] == 0.25

    def test_no_coef_raises(self, trained_posteriors):
        with pytest.raises(ValueError, match="no 'coef'"):
            get_coef_sign_fraction(trained_posteriors)
        with pytest.raises(ValueError, match="no 'coef'"):
            get_regression_summary(trained_posteriors)

    def test_summary_quantiles(self):
        idata = self._idata()
        out = get_regression_summary(idata)
        coef = idata["coef"].to_numpy()[0]
        assert out.columns.tolist() == ["coef_p50", "coef_p05", "coef_p95", "frac_pos", "frac_neg"]
        np.testing.assert_allclose(out["coef_p50"], np.median(coef, axis=0))
        np.testing.assert_allclose(out["coef_p05"], np.quantile(coef, 0.05, axis=0))
        np.testing.assert_allclose(out["coef_p95"], np.quantile(coef, 0.95, axis=0))

    def test_summary_custom_interval(self):
        out = get_regression_summary(self._idata(), lower=0.025, upper=0.975)
        assert {"coef_p2.5", "coef_p97.5"}.issubset(out.columns)

    @pytest.mark.parametrize("lower, upper", [(0.9, 0.1), (-0.1, 0.9), (0.1, 1.1)])
    def test_summary_bad_interval_raises(self, lower, upper):
        with pytest.raises(ValueError, match="lower < upper"):
            get_regression_summary(self._idata(), lower=lower, upper=upper)

    def test_summary_with_fitted_scheme(self, synthetic_y):
        rng = np.random.default_rng(0)
        cov = pd.DataFrame({"neu": rng.normal(size=N_TRAIN), "pos": rng.normal(size=N_TRAIN)})
        scheme = RegressionScheme()
        scheme.add_regressors(["neu", "pos"], coef_sign=["=", "+"], loc_prior=0.0, scale_prior=[1.0, 0.5])
        idata = run_dlt_model(
            RNG, LEV_SM, SLP_SM, THETA, synthetic_y, MCMC_ARGS, regression_scheme=scheme, covariates_df=cov
        )
        out = get_regression_summary(idata, regression_scheme=scheme)
        assert out.index.tolist() == ["neu", "pos"]
        assert out.loc["pos", "frac_pos"] == 1.0  # truncated at zero
        assert out.loc["pos", "frac_neg"] == 0.0
        assert out["sign"].tolist() == ["=", "+"]
        np.testing.assert_allclose(out["scale_prior"], [1.0, 0.5])
        np.testing.assert_allclose(out["coef_p50"], idata["coef_p50"].to_numpy())


class TestForecastDecomposition:
    """#80: forecast decomposed into trend and regression components."""

    N_SAMPLES = 40
    H = 4

    def _inference(self):
        """make_inference output with regressors: two weekday dummies and a price covariate."""
        rng = np.random.default_rng(1)
        n_total = N_TRAIN + self.H
        idata = xr.Dataset(
            {
                "dlt_comp": (
                    ["chain", "draw", "time"],
                    np.tile(np.linspace(10.0, 12.0, N_TRAIN), (1, self.N_SAMPLES, 1)),
                ),
                "sigma": (["chain", "draw"], np.full((1, self.N_SAMPLES), 0.5)),
                "last_lev": (["chain", "draw"], np.full((1, self.N_SAMPLES), 12.0)),
                "last_slp": (["chain", "draw"], np.full((1, self.N_SAMPLES), 0.1)),
                "coef": (["chain", "draw", "var_name"], rng.normal([1.0, -1.0, 0.5], 0.1, size=(1, self.N_SAMPLES, 3))),
            },
            coords={"var_name": ["dow_1", "price", "dow_2"]},
        )
        cov = pd.DataFrame(
            {
                "dow_1": np.arange(n_total) % 2 == 0,
                "price": rng.normal(size=n_total),
                "dow_2": np.arange(n_total) % 2 == 1,
            }
        ).astype(float)
        return make_inference(RNG, idata, LEV_SM, SLP_SM, THETA, covariates_df=cov)

    def test_components_without_groups(self):
        inference = self._inference()
        out = get_forecast_decomposition(inference)
        components = [c[: -len("_p50")] for c in out.columns if c.endswith("_p50")]
        assert components == ["trend", "dow_1", "price", "dow_2", "regression", "yhat"]
        assert out.index.name == "time"
        assert len(out) == N_TRAIN + self.H
        trend = inference["dlt_comp_samples"].to_numpy()
        reg = inference["reg_comp_samples"].to_numpy()
        np.testing.assert_allclose(out["trend_p50"], np.median(trend, axis=0))
        np.testing.assert_allclose(out["price_p05"], np.quantile(reg[..., 1], 0.05, axis=0))
        np.testing.assert_allclose(out["regression_p95"], np.quantile(reg.sum(axis=-1), 0.95, axis=0))
        # yhat is summarised per sample, and matches forecast_samples without noise or transform
        np.testing.assert_allclose(out["yhat_p50"], np.median(inference["forecast_samples"].to_numpy(), axis=0))

    def test_groups_sum_regressors(self):
        inference = self._inference()
        out = get_forecast_decomposition(inference, groups={"seasonality": ["dow_1", "dow_2"]})
        components = [c[: -len("_p50")] for c in out.columns if c.endswith("_p50")]
        assert components == ["trend", "seasonality", "price", "regression", "yhat"]
        reg = inference["reg_comp_samples"].to_numpy()
        np.testing.assert_allclose(out["seasonality_p50"], np.median(reg[..., [0, 2]].sum(axis=-1), axis=0))

    def test_without_regression(self, trained_posteriors):
        inference = make_inference(RNG, trained_posteriors, LEV_SM, SLP_SM, THETA, end_step=N_TRAIN + self.H)
        out = get_forecast_decomposition(inference, lower=0.1, upper=0.9)
        assert out.columns.tolist() == ["trend_p50", "trend_p10", "trend_p90", "yhat_p50", "yhat_p10", "yhat_p90"]
        np.testing.assert_allclose(out["yhat_p50"], out["trend_p50"])
        assert (out["trend_p10"] <= out["trend_p90"]).all()

    @pytest.mark.parametrize(
        "groups, match",
        [
            ({"seasonality": ["dow_1", "dow_9"]}, "unknown regressors"),
            ({"a": ["dow_1"], "b": ["dow_1", "dow_2"]}, "more than one group"),
            ({"price": ["dow_1", "dow_2"]}, "clash"),
            ({"trend": ["dow_1"]}, "clash"),
        ],
        ids=["unknown", "shared", "clash_regressor", "clash_reserved"],
    )
    def test_bad_groups_raise(self, groups, match):
        with pytest.raises(ValueError, match=match):
            get_forecast_decomposition(self._inference(), groups=groups)

    def test_groups_without_regression_raises(self, trained_posteriors):
        inference = make_inference(RNG, trained_posteriors, LEV_SM, SLP_SM, THETA, end_step=N_TRAIN)
        with pytest.raises(ValueError, match="no 'reg_comp_samples'"):
            get_forecast_decomposition(inference, groups={"seasonality": ["x"]})

    def test_bad_interval_raises(self):
        with pytest.raises(ValueError, match="lower < upper"):
            get_forecast_decomposition(self._inference(), lower=0.9, upper=0.1)
