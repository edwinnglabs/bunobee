"""Unit tests for the M5 target transforms, ``bunobee.models.m5.transforms`` (issue #114).

The MAP fitters and predictors of ``m5_ssp_optim`` used to hard-code the log target: fit on
``log(max(sales, 0.1) / mean)``, forecast ``exp(mu + 0.5 · var) · mean``. They now take a ``Transform``, by
default ``LogTransform``. These tests check that ``LogTransform`` is that target bit for bit, and that the
default fit and forecast reproduce the pre-refactor implementation, kept verbatim below as the reference,
exactly on a fixed set of series.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit, value_and_grad
from numpyro import distributions as dist

from bunobee.models.m5.transforms import LogTransform
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
