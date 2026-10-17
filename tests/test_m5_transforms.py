"""Unit tests for the M5 target transforms, ``bunobee.models.m5.transforms`` (issue #114).

The MAP fitters and predictors of ``m5_ssp_optim`` used to hard-code the log target: fit on
``log(max(sales, 0.1) / mean)``, forecast ``exp(mu + 0.5 · var) · mean``. They now take a ``Transform``, by
default ``LogTransform``. These tests check that ``LogTransform`` is that target bit for bit, and that the
default fit and forecast reproduce the pre-refactor implementation, kept verbatim below as the reference,
exactly on a fixed set of series.

``PowerTransform`` (issue #115) fits a Box-Cox or Yeo-Johnson λ per series with scikit-learn's
``PowerTransformer``. These tests check its forward map against scikit-learn's, its λ bounds against a bounded
maximum-likelihood fit, and its inverse against the mean of the censored sales computed by adaptive quadrature;
and that its forecasts are never negative.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit, value_and_grad
from numpyro import distributions as dist
from scipy import integrate, optimize, special, stats

from bunobee.models.m5.transforms import LogTransform, PowerTransform
from bunobee.regression import make_periodic_dummies

try:
    import optax

    from bunobee.models.m5 import m5_ssp_optim
except ModuleNotFoundError:  # the optional 'm5' extra (optax, tqdm) is not installed
    m5_ssp_optim = None

requires_m5_extra = pytest.mark.skipif(
    m5_ssp_optim is None,
    reason="needs the optional 'm5' extra: pip install 'bunobee[m5]'",
)

requires_sklearn = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="needs scikit-learn, part of the optional 'm5' extra: pip install 'bunobee[m5]'",
)

# Fixed series: Poisson sales from intermittent to dense, so the 0.1 floor bites on most days of the first.
N_FIT = 140
HORIZON = 28
LEVELS = np.array([0.2, 0.7, 1.5, 4.0, 15.0])


@pytest.fixture(scope="module")
def sales() -> np.ndarray:
    """Daily sales of the fixed series over the fit and forecast days, shape (n_series, N_FIT + HORIZON)."""
    rng = np.random.default_rng(114)
    return rng.poisson(LEVELS[:, None], size=(LEVELS.size, N_FIT + HORIZON)).astype(np.float32)


def _assert_identical(actual: np.ndarray, expected: np.ndarray) -> None:
    """Assert two arrays have the same dtype and the same values, bit for bit."""
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


class TestLogTransform:
    """``LogTransform`` is the log target the fitters hard-coded, bit for bit, for a batch and one series."""

    def test_batch_forward_is_the_legacy_target(self, sales):
        transform = LogTransform()
        params = transform.fit(sales)
        clipped = np.clip(sales, 1e-1, None).astype(np.float32)
        response_norm = clipped.mean(axis=1)
        _assert_identical(params["response_norm"], response_norm)
        _assert_identical(transform.forward(sales, params), np.log(clipped / response_norm[:, None]))

    def test_single_series_forward_is_the_legacy_target(self, sales):
        transform = LogTransform()
        params = transform.fit(sales[0])
        clipped = np.clip(sales[0], 1e-1, None).astype(np.float32)
        response_norm = float(clipped.mean())
        assert float(params["response_norm"]) == response_norm
        _assert_identical(transform.forward(sales[0], params), np.log(clipped / response_norm))

    def test_inverse_is_the_legacy_lognormal_mean(self):
        # The single-series fit used to return response_norm as a Python float, so check that type too.
        rng = np.random.default_rng(0)
        mu = rng.normal(size=(3, HORIZON)).astype(np.float32)
        var = rng.uniform(0.01, 0.3, size=(3, HORIZON)).astype(np.float32)
        response_norm = rng.uniform(0.5, 5.0, size=3).astype(np.float32)
        params = {"response_norm": response_norm}
        out = LogTransform().inverse(mu, var, params)
        _assert_identical(out, np.exp(mu + 0.5 * var) * response_norm[:, None])
        for value in (response_norm[0], float(response_norm[0])):
            single = LogTransform().inverse(mu[0], var[0], {"response_norm": value})
            _assert_identical(single, np.exp(mu[0] + 0.5 * var[0]) * value)

    def test_floor_sets_the_value_of_zero_days(self, sales):
        # With floor=1 a zero day counts as one unit: it adds 1 to the mean and maps to log(1 / mean).
        transform = LogTransform(floor=1.0)
        params = transform.fit(sales)
        np.testing.assert_allclose(params["response_norm"], np.maximum(sales, 1.0).mean(axis=1), rtol=1e-6)
        y = transform.forward(sales, params)
        zero_day_value = np.broadcast_to(-np.log(params["response_norm"])[:, None], y.shape)
        np.testing.assert_allclose(y[sales == 0], zero_day_value[sales == 0], rtol=1e-6)


def _softplus(x):
    return m5_ssp_optim._softplus(x)


def _softplus_inv(v):
    return m5_ssp_optim._softplus_inv(v)


def _legacy_fit_batch_series_opt(sales_matrix, n_iter, lr=3e-2, Z=None, chunk_size=4096):
    """The pre-refactor ``fit_batch_series_opt``, kept verbatim but for logging, as the numerical reference."""
    B = sales_matrix.shape[0]

    sales_clipped = np.clip(sales_matrix, 1e-1, None).astype(np.float32)
    response_norm = sales_clipped.mean(axis=1)  # (B,)
    y = jnp.array(np.log(sales_clipped / response_norm[:, None]))  # (B, n_steps)

    n_states = Z.shape[1]
    a0 = jnp.zeros(n_states)
    P0 = jnp.ones(n_states)

    init_params = jnp.tile(jnp.array([_softplus_inv(0.30), _softplus_inv(0.055), _softplus_inv(0.055)]), (B, 1))

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(init_params)

    def _loss_fn(params):
        sigma_h = _softplus(params[:, 0])
        sigma_q_level = _softplus(params[:, 1])
        sigma_q_seas = _softplus(params[:, 2])
        sigma_q = jnp.concatenate(
            [sigma_q_level[:, None], jnp.repeat(sigma_q_seas[:, None], n_states - 1, axis=1)], axis=1
        )
        log_p, _, _, _, _, _ = m5_ssp_optim.kalman_filter_1d_batch(
            a0=a0, P0=P0, Z=Z, sigma_h=sigma_h, sigma_q=sigma_q, y=y, logp=True, chunk_size=chunk_size
        )
        log_prior = (
            dist.Uniform(0.1, 0.5).log_prob(sigma_h)
            + dist.Uniform(0.01, 0.1).log_prob(sigma_q_level)
            + dist.Uniform(0.01, 0.1).log_prob(sigma_q_seas)
        )
        per_series = -(log_p + log_prior)
        return jnp.sum(per_series), per_series

    @jit
    def _step(params, opt_state, best_params, best_loss):
        (_, per_series_loss), grads = value_and_grad(_loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        improved = per_series_loss < best_loss
        new_best_loss = jnp.where(improved, per_series_loss, best_loss)
        new_best_params = jnp.where(improved[:, None], new_params, best_params)
        return new_params, new_opt_state, new_best_params, new_best_loss, per_series_loss

    best_params = init_params
    best_loss = jnp.full(B, float("inf"))
    params = init_params
    for _ in range(n_iter):
        params, opt_state, best_params, best_loss, _ = _step(params, opt_state, best_params, best_loss)

    sigma_h = jnp.asarray(_softplus(best_params[:, 0]))
    sigma_q_level = _softplus(best_params[:, 1])
    sigma_q_seas = _softplus(best_params[:, 2])
    sigma_q = jnp.concatenate([sigma_q_level[:, None], jnp.repeat(sigma_q_seas[:, None], n_states - 1, axis=1)], axis=1)
    _, at, Pt, _, _, _ = m5_ssp_optim.kalman_filter_1d_batch(
        a0=a0, P0=P0, Z=Z, sigma_h=sigma_h, sigma_q=sigma_q, y=y, logp=False, chunk_size=chunk_size
    )
    return {
        "at": np.array(at),
        "Pt": np.array(Pt),
        "sigma_h": np.array(sigma_h),
        "sigma_q": np.array(sigma_q),
        "response_norm": response_norm,
        "final_loss": np.array(best_loss),
    }


def _legacy_predict_batch_series_opt(fit_result, Z_future):
    """The pre-refactor ``predict_batch_series_opt``, kept verbatim as the numerical reference."""
    at = fit_result["at"]
    Pt = fit_result["Pt"]
    sigma_h = np.asarray(fit_result["sigma_h"])
    response_norm = fit_result["response_norm"]
    Z_future = np.asarray(Z_future)
    a_last = at[:, -1, :]
    P_last = Pt[:, -1, :]
    mu_future = a_last @ Z_future.T
    Z_sq = Z_future**2
    var_future = P_last @ Z_sq.T + (sigma_h**2)[:, None]
    return np.exp(mu_future + 0.5 * var_future) * response_norm[:, None]


def _legacy_fit_one_series_opt(sales, n_iter, lr=3e-2, Z=None, patience=50, tol=1e-5):
    """The pre-refactor ``fit_one_series_opt``, kept verbatim but for logging, as the numerical reference."""
    sales_clipped = np.clip(sales, 1e-1, None).astype(np.float32)
    response_norm = float(sales_clipped.mean())
    y = jnp.array(np.log(sales_clipped / response_norm))

    n_states = Z.shape[1]
    a0 = jnp.zeros(n_states)
    P0 = jnp.ones(n_states)

    def neg_log_posterior(params):
        sigma_h = _softplus(params[0])
        sigma_q_level = _softplus(params[1])
        sigma_q_seas = _softplus(params[2])
        sigma_q = jnp.concatenate([sigma_q_level[None], jnp.repeat(sigma_q_seas[None], n_states - 1)])
        lp, _, _, _, _, _ = m5_ssp_optim.kalman_filter_1d(
            a0=a0, P0=P0, sigma_h=sigma_h, sigma_q=sigma_q, y=y, Z=Z, logp=True
        )
        log_prior = (
            dist.Uniform(0.1, 0.5).log_prob(sigma_h)
            + dist.Uniform(0.01, 0.1).log_prob(sigma_q_level)
            + dist.Uniform(0.01, 0.1).log_prob(sigma_q_seas)
        )
        return -(lp + log_prior)

    params = jnp.array([_softplus_inv(0.30), _softplus_inv(0.055), _softplus_inv(0.055)])
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jit
    def _step(params, opt_state):
        loss, grads = value_and_grad(neg_log_posterior)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), new_opt_state, loss

    best_loss = float("inf")
    no_improve = 0
    for _ in range(n_iter):
        params, opt_state, loss = _step(params, opt_state)
        loss_val = float(loss)
        if patience is not None:
            if best_loss - loss_val > tol:
                best_loss = loss_val
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

    sigma_h = float(_softplus(params[0]))
    sigma_q_level = float(_softplus(params[1]))
    sigma_q_seas = float(_softplus(params[2]))
    sigma_q = jnp.concatenate([jnp.array([sigma_q_level]), jnp.repeat(jnp.array([sigma_q_seas]), n_states - 1)])
    _, at, Pt, _, _, _ = m5_ssp_optim.kalman_filter_1d(
        a0=a0, P0=P0, sigma_h=sigma_h, sigma_q=sigma_q, y=y, Z=Z, logp=False
    )
    return {"at": np.array(at), "Pt": np.array(Pt), "sigma_h": sigma_h, "response_norm": response_norm}


def _legacy_predict_one_series_opt(fit_result, Z_future):
    """The pre-refactor ``predict_one_series_opt``, kept verbatim as the numerical reference."""
    at = fit_result["at"]
    Pt = fit_result["Pt"]
    sigma_h = float(fit_result["sigma_h"])
    response_norm = fit_result["response_norm"]
    Z_future = np.asarray(Z_future)
    a_last = at[-1]
    P_last = Pt[-1]
    mu_future = Z_future @ a_last
    Z_sq = Z_future**2
    var_future = Z_sq @ P_last + sigma_h**2
    return np.exp(mu_future + 0.5 * var_future) * response_norm


def _designs(intercept: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Weekday dummies over the fit and forecast days, phased as M5's d_1 (a Saturday), with or without a level."""
    weekly = make_periodic_dummies(N_FIT + HORIZON, period=7, phase=5, intercept=intercept)
    return jnp.asarray(weekly[:N_FIT]), jnp.asarray(weekly[N_FIT:])


@requires_m5_extra
class TestReproducesLegacy:
    """With the default ``LogTransform`` the fitters and predictors return today's forecasts exactly."""

    @pytest.mark.parametrize("intercept", [False, True], ids=["ssp_dow", "ssp_level_dow"])
    def test_batch_forecasts_unchanged(self, sales, intercept):
        Z, Z_future = _designs(intercept)
        fit_sales = sales[:, :N_FIT]

        result = m5_ssp_optim.fit_batch_series_opt(fit_sales, n_iter=60, Z=Z)
        legacy = _legacy_fit_batch_series_opt(fit_sales, n_iter=60, Z=Z)

        assert result["transform"] == LogTransform()
        for key in legacy:
            np.testing.assert_array_equal(result[key], legacy[key], err_msg=key)
        _assert_identical(
            m5_ssp_optim.predict_batch_series_opt(result, Z_future=Z_future),
            _legacy_predict_batch_series_opt(legacy, Z_future),
        )

    def test_single_series_forecasts_unchanged(self, sales):
        Z, Z_future = _designs(intercept=True)
        for series in sales[:, :N_FIT]:
            result = m5_ssp_optim.fit_one_series_opt(series, n_iter=60, Z=Z)
            legacy = _legacy_fit_one_series_opt(series, n_iter=60, Z=Z)
            assert float(result["response_norm"]) == legacy["response_norm"]
            _assert_identical(
                m5_ssp_optim.predict_one_series_opt(result, Z_future=Z_future),
                _legacy_predict_one_series_opt(legacy, Z_future),
            )


# The registered power transforms (m5_models.py): zero-sales days 0.5, 0.1 and 1 unit above zero before the power.
POWER_TRANSFORMS = {
    "boxcox_shift": PowerTransform("box-cox", shift=0.5),
    "boxcox_floor": PowerTransform("box-cox", floor=0.1),
    "yeojohnson": PowerTransform("yeo-johnson"),
}


def _censored_mean(transform: PowerTransform, mu: float, var: float, params: dict) -> float:
    """Mean of the sales a Gaussian forecast implies, by adaptive quadrature of the Gaussian density.

    An independent reference for ``PowerTransform.inverse``, which integrates the survival function instead: here the
    Gaussian ``z`` on the original power scale is mapped back by the inverse power, censored to the admissible
    sales from 0 to ``sales_max``, and averaged under its density.
    """
    offset = 1.0 if transform.method == "yeo-johnson" else 0.0
    lmbda, loc, scale = (float(params[key]) for key in ("lmbda", "loc", "scale"))
    w_low = transform.floor + transform.shift + offset
    w_high = max(float(params["sales_max"]), transform.floor) + transform.shift + offset
    z_low, z_high = special.boxcox(w_low, lmbda), special.boxcox(w_high, lmbda)
    centre, spread = loc + scale * mu, scale * np.sqrt(var)
    a, b = (z_low - centre) / spread, (z_high - centre) / spread
    inside = integrate.quad(
        lambda t: special.inv_boxcox(centre + spread * t, lmbda) * stats.norm.pdf(t),
        a,
        b,
        epsabs=1e-13,
        epsrel=1e-12,
        limit=500,
    )[0]
    mean = w_low * stats.norm.cdf(a) + inside + w_high * stats.norm.sf(b)
    return mean - offset - transform.shift


@requires_sklearn
class TestPowerTransform:
    """``PowerTransform`` is scikit-learn's power transform, with λ bounds and a mean-preserving inverse."""

    @pytest.mark.parametrize("name", list(POWER_TRANSFORMS))
    def test_forward_is_sklearns_standardized_power(self, sales, name):
        from sklearn.preprocessing import PowerTransformer

        transform = replace(POWER_TRANSFORMS[name], lmbda_bounds=None)
        # PowerTransform fits in float64; scikit-learn fits in its input's dtype.
        x = np.maximum(sales.astype(np.float64), transform.floor) + transform.shift
        reference = PowerTransformer(method=transform.method, standardize=True).fit(x.T)
        params = transform.fit(sales)
        np.testing.assert_allclose(params["lmbda"], reference.lambdas_, rtol=1e-9)
        np.testing.assert_allclose(transform.forward(sales, params), reference.transform(x.T).T, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(params["sales_max"], sales.max(axis=1))

    def test_unstandardized_forward_is_the_power(self, sales):
        transform = PowerTransform("box-cox", floor=0.1, standardize=False)
        params = transform.fit(sales)
        np.testing.assert_array_equal(params["loc"], 0.0)
        np.testing.assert_array_equal(params["scale"], 1.0)
        expected = special.boxcox(np.maximum(sales, 0.1).astype(np.float64), params["lmbda"][:, None])
        np.testing.assert_allclose(transform.forward(sales, params), expected, rtol=1e-6)

    def test_yeo_johnson_is_box_cox_of_sales_plus_one(self, sales):
        # On sales >= 0 Yeo-Johnson is the Box-Cox power of sales + 1, with the same maximum-likelihood λ.
        yeo_johnson, box_cox = PowerTransform("yeo-johnson"), PowerTransform("box-cox", shift=1.0)
        params_yj, params_bc = yeo_johnson.fit(sales), box_cox.fit(sales)
        np.testing.assert_allclose(params_yj["lmbda"], params_bc["lmbda"], rtol=1e-5)
        np.testing.assert_allclose(yeo_johnson.forward(sales, params_yj), box_cox.forward(sales, params_bc), atol=1e-4)

    @pytest.mark.parametrize("name", list(POWER_TRANSFORMS))
    def test_bounded_lmbda_is_the_bounded_maximum_likelihood(self, sales, name):
        # The intermittent first series has its unbounded λ below -1, so the bound binds there.
        transform = POWER_TRANSFORMS[name]
        offset = 1.0 if transform.method == "yeo-johnson" else 0.0
        unbounded = replace(transform, lmbda_bounds=None).fit(sales)["lmbda"]
        bounded = transform.fit(sales)["lmbda"]
        assert unbounded[0] < -1 and bounded[0] == -1.0
        for series, lmbda in zip(sales, bounded):
            w = np.maximum(series, transform.floor).astype(np.float64) + transform.shift + offset
            best = optimize.minimize_scalar(
                lambda lm: -stats.boxcox_llf(lm, w), bounds=transform.lmbda_bounds, method="bounded"
            )
            assert lmbda == pytest.approx(best.x, abs=1e-4)

    def test_constant_series_keeps_lmbda_one(self, sales):
        # scipy's Box-Cox raises on a constant series; it keeps λ = 1 and is forecast as its constant sales.
        transform = POWER_TRANSFORMS["boxcox_shift"]
        with_constant = np.vstack([sales, np.zeros((1, sales.shape[1]), dtype=np.float32)])
        params = transform.fit(with_constant)
        assert params["lmbda"][-1] == 1.0 and params["scale"][-1] == 1.0
        np.testing.assert_array_equal(transform.forward(with_constant, params)[-1], 0.0)
        forecast = transform.inverse(np.zeros((6, 3)), np.ones((6, 3)), params)
        np.testing.assert_array_equal(forecast[-1], 0.0)

    @pytest.mark.parametrize("name", list(POWER_TRANSFORMS))
    def test_inverse_is_the_mean_of_the_censored_sales(self, sales, name):
        transform = POWER_TRANSFORMS[name]
        params = transform.fit(sales)
        mu = np.array([-2.0, -0.5, 0.0, 0.7, 2.5])
        var = np.array([0.02, 0.25, 1.0, 0.4, 3.0])
        forecast = transform.inverse(np.tile(mu, (LEVELS.size, 1)), np.tile(var, (LEVELS.size, 1)), params)
        for i in range(LEVELS.size):
            series = {key: value[i] for key, value in params.items()}
            expected = [_censored_mean(transform, m, v, series) for m, v in zip(mu, var)]
            np.testing.assert_allclose(forecast[i], expected, rtol=1e-6, atol=1e-12, err_msg=f"series {i}")

    @pytest.mark.parametrize("name", list(POWER_TRANSFORMS))
    def test_inverse_of_a_certain_forecast_is_the_inverse_power(self, sales, name):
        # With no variance the forecast is its median, so forward then inverse returns the floored sales.
        transform = POWER_TRANSFORMS[name]
        params = transform.fit(sales)
        y = transform.forward(sales, params).astype(np.float64)
        np.testing.assert_allclose(
            transform.inverse(y, np.zeros_like(y), params), np.maximum(sales, transform.floor), rtol=1e-5, atol=1e-6
        )

    @pytest.mark.parametrize("name", list(POWER_TRANSFORMS))
    def test_forecasts_stay_within_the_admissible_sales(self, sales, name):
        # However far the Gaussian forecast reaches, the forecast is finite, never negative nor below the floor,
        # and never above the largest fit-day sales.
        transform = POWER_TRANSFORMS[name]
        params = transform.fit(sales)
        mu, var = np.meshgrid([-1e3, -20.0, -3.0, 0.0, 3.0, 20.0, 1e3], [0.0, 1e-8, 0.3, 30.0, 1e4])
        for i in range(LEVELS.size):
            series = {key: value[i] for key, value in params.items()}
            forecast = transform.inverse(mu, var, series)
            assert np.isfinite(forecast).all()
            assert (forecast >= transform.floor).all()
            assert (forecast <= max(series["sales_max"], transform.floor) * (1 + 1e-9)).all()

    def test_single_series_matches_its_batch_row(self, sales):
        transform = POWER_TRANSFORMS["yeojohnson"]
        batch, single = transform.fit(sales), transform.fit(sales[2])
        for key, value in single.items():
            assert np.shape(value) == ()
            assert float(value) == pytest.approx(batch[key][2], rel=1e-9)
        np.testing.assert_allclose(transform.forward(sales[2], single), transform.forward(sales, batch)[2], rtol=1e-6)
        mu, var = np.linspace(-1, 1, HORIZON), np.full(HORIZON, 0.3)
        np.testing.assert_allclose(
            transform.inverse(mu, var, single),
            transform.inverse(np.tile(mu, (LEVELS.size, 1)), np.tile(var, (LEVELS.size, 1)), batch)[2],
            rtol=1e-9,
        )

    def test_inverse_keeps_float32(self, sales):
        transform = POWER_TRANSFORMS["boxcox_floor"]
        params = transform.fit(sales)
        mu = np.zeros((LEVELS.size, HORIZON), dtype=np.float32)
        assert transform.inverse(mu, np.ones_like(mu), params).dtype == np.float32

    @pytest.mark.parametrize(
        "kwargs",
        [{"method": "log"}, {"method": "box-cox"}, {"method": "box-cox", "shift": -0.5, "floor": 1.0}],
        ids=["unknown method", "box-cox of zero", "negative shift"],
    )
    def test_invalid_settings_raise(self, kwargs):
        with pytest.raises(ValueError):
            PowerTransform(**kwargs)


@requires_m5_extra
@requires_sklearn
class TestPowerTransformFit:
    """The MAP fitters fit a power transform as any other: they return its parameters and invert its forecasts."""

    @pytest.mark.parametrize("name", list(POWER_TRANSFORMS))
    def test_batch_fit_returns_its_parameters_and_nonnegative_forecasts(self, sales, name):
        transform = POWER_TRANSFORMS[name]
        Z, Z_future = _designs(intercept=False)
        result = m5_ssp_optim.fit_batch_series_opt(sales[:, :N_FIT], n_iter=30, Z=Z, transform=transform)
        assert result["transform"] == transform
        expected = transform.fit(sales[:, :N_FIT])
        for key, value in expected.items():
            np.testing.assert_array_equal(result[key], value, err_msg=key)
        forecasts = m5_ssp_optim.predict_batch_series_opt(result, Z_future=Z_future)
        assert forecasts.shape == (LEVELS.size, HORIZON)
        assert np.isfinite(forecasts).all() and (forecasts >= 0).all()
