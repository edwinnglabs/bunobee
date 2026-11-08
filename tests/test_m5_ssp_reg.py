"""Unit tests for the NUTS fitter with a static regression, ``m5_ssp_reg``.

The model is ``level_t + dow_t · alpha_t + x_t · beta`` with the level and weekday offsets as the filter's states and
``beta`` a parameter of the series with no time dimension. These tests check that the filter sees the series less
``X beta`` (the log density of the model against the filter's own), that the sampler recovers a known ``beta`` and keeps
the series apart, and that the forecast is linear in the future ``X`` on the target scale.

Every test fits a tiny synthetic dataset, on the identity transform so that the model scale is the data scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import distributions as dist
from numpyro.infer.util import log_density

from bunobee.models.ssp.kalman_1d import kalman_filter_1d
from bunobee.regression import make_periodic_dummies

m5_ssp_reg = pytest.importorskip(
    "bunobee.models.m5.m5_ssp_reg", reason="needs the optional 'm5' extra: pip install 'bunobee[m5]'"
)
SigmaPrior = m5_ssp_reg.SigmaPrior

N_FIT = 200
HORIZON = 28
N_SERIES = 3
SIGMA_PRIORS = {
    "sigma_h_prior": SigmaPrior("half_normal", (1.0,)),
    "sigma_q_prior": SigmaPrior("half_normal", (0.05,)),
}
BETA = np.array([[1.5, -1.0], [0.0, 0.5], [-0.8, 0.0]])  # True coefficients, one row per series.


@dataclass(frozen=True)
class Identity:
    """The identity transform: the model scale is the data scale. It has no parameter."""

    def fit(self, sales: np.ndarray) -> dict[str, np.ndarray]:
        return {}

    def forward(self, sales: np.ndarray, params) -> np.ndarray:
        return np.asarray(sales, dtype=np.float32)

    def inverse(self, mu: np.ndarray, var: np.ndarray, params) -> np.ndarray:
        return mu


@pytest.fixture(scope="module")
def design() -> tuple[np.ndarray, np.ndarray]:
    """The filter design ``Z`` and the regression design ``X`` over the fit and forecast days."""
    rng = np.random.default_rng(3)
    n = N_FIT + HORIZON
    Z = make_periodic_dummies(n, period=7, phase=0, intercept=True)
    X = np.zeros((n, 2), dtype=np.float32)
    X[rng.choice(N_FIT, 30, replace=False), 0] = 1.0
    X[rng.choice(N_FIT, 25, replace=False), 1] = 1.0
    X[N_FIT:, 0] = np.tile([1.0, 0.0, 0.0, 0.0], HORIZON // 4)
    return Z, X


@pytest.fixture(scope="module")
def sales(design) -> np.ndarray:
    """Series with a weekly pattern, a slow level and the effects ``BETA`` on the model scale, ``(N_SERIES, N_FIT)``."""
    Z, X = design
    rng = np.random.default_rng(4)
    level = 10 + np.cumsum(rng.normal(0, 0.03, N_FIT))
    weekly = Z[:N_FIT, 1:] @ np.array([0.5, 1.0, 0.3, -0.4, -0.8, 0.2])
    noise = rng.normal(0, 0.3, (N_SERIES, N_FIT))
    return level + weekly + BETA @ X[:N_FIT].T + noise


@pytest.fixture(scope="module")
def fit(design, sales) -> dict:
    Z, X = design
    return m5_ssp_reg.fit_batch_series_reg(
        sales,
        Z[:N_FIT],
        X[:N_FIT],
        tau=1.0,
        transform=Identity(),
        **SIGMA_PRIORS,
        num_warmup=150,
        num_samples=150,
        num_chains=2,
    )


class TestModel:
    """The model's log density is the filter's likelihood of the series less ``X beta``, plus the priors."""

    def test_log_density_is_the_filter_on_the_series_less_x_beta_plus_the_priors(self, design, sales):
        Z, X = design
        Zf, Xf = jnp.asarray(Z[:N_FIT], dtype=jnp.float32), jnp.asarray(X[:N_FIT])
        y = jnp.asarray(sales, dtype=jnp.float32)
        tau = 0.5
        params = {
            "sigma_h": jnp.array([0.3, 0.4, 0.5]),
            "sigma_q_level": jnp.array([0.02, 0.03, 0.04]),
            "sigma_q_dow": jnp.array([0.01, 0.015, 0.02]),
            "beta": jnp.asarray(BETA, dtype=jnp.float32),
        }
        got, _ = log_density(m5_ssp_reg.ssp_reg_model, (y, Zf, Xf, tau, *SIGMA_PRIORS.values()), {}, params)

        expected = 0.0
        for i in range(N_SERIES):
            sigma_q = jnp.array([params["sigma_q_level"][i], *[params["sigma_q_dow"][i]] * 6])
            log_p, *_ = kalman_filter_1d(
                a0=jnp.zeros(7),
                P0=jnp.ones(7),
                Z=Zf,
                sigma_h=params["sigma_h"][i],
                sigma_q=sigma_q,
                y=y[i] - Xf @ params["beta"][i],
                logp=True,
            )
            expected += log_p
            expected += SIGMA_PRIORS["sigma_h_prior"].distribution.log_prob(params["sigma_h"][i])
            expected += SIGMA_PRIORS["sigma_q_prior"].distribution.log_prob(params["sigma_q_level"][i])
            expected += SIGMA_PRIORS["sigma_q_prior"].distribution.log_prob(params["sigma_q_dow"][i])
            expected += dist.Normal(0.0, tau).log_prob(params["beta"][i]).sum()
        np.testing.assert_allclose(got, expected, rtol=1e-4)

    def test_without_covariates_there_is_no_beta_site(self, design, sales):
        Z, _ = design
        result = m5_ssp_reg.fit_batch_series_reg(
            sales, Z[:N_FIT], None, transform=Identity(), **SIGMA_PRIORS, num_warmup=20, num_samples=20, num_chains=1
        )
        assert "beta" not in result["samples"]
        assert result["X"].shape == (N_FIT, 0)


class TestFit:
    """The sampler recovers a known ``beta``, keeps each series' own, and reports its diagnostics per series."""

    def test_posterior_mean_of_beta_is_near_the_truth(self, fit):
        np.testing.assert_allclose(fit["samples"]["beta"].mean(axis=0), BETA, atol=0.25)

    def test_sample_shapes(self, fit):
        n_draws = 2 * 150
        assert fit["samples"]["beta"].shape == (n_draws, N_SERIES, 2)
        for site in m5_ssp_reg.SIGMA_SITES:
            assert fit["samples"][site].shape == (n_draws, N_SERIES)
        assert fit["y"].shape == (N_SERIES, N_FIT)

    def test_diagnostics_are_per_series(self, fit):
        assert fit["r_hat_max"].shape == fit["n_eff_min"].shape == (N_SERIES,)
        assert (fit["r_hat_max"] < 1.1).all()
        assert (fit["n_eff_min"] > 50).all()

    def test_one_chain_has_no_r_hat(self, design, sales):
        Z, X = design
        result = m5_ssp_reg.fit_batch_series_reg(
            sales,
            Z[:N_FIT],
            X[:N_FIT],
            transform=Identity(),
            **SIGMA_PRIORS,
            num_warmup=100,
            num_samples=100,
            num_chains=1,
        )
        assert np.isnan(result["r_hat_max"]).all()
        assert np.isfinite(result["n_eff_min"]).all()

    def test_same_seed_same_samples(self, design, sales):
        Z, X = design
        kwargs = dict(transform=Identity(), **SIGMA_PRIORS, num_warmup=20, num_samples=20, num_chains=1, seed=7)
        first = m5_ssp_reg.fit_batch_series_reg(sales, Z[:N_FIT], X[:N_FIT], **kwargs)
        second = m5_ssp_reg.fit_batch_series_reg(sales, Z[:N_FIT], X[:N_FIT], **kwargs)
        np.testing.assert_array_equal(first["samples"]["beta"], second["samples"]["beta"])

    @pytest.mark.parametrize(
        ("z_cols", "x_rows", "match"),
        [(1, N_FIT, "level column"), (7, N_FIT - 1, "differ in steps")],
    )
    def test_bad_shapes_are_rejected(self, sales, z_cols, x_rows, match):
        with pytest.raises(ValueError, match=match):
            m5_ssp_reg.fit_batch_series_reg(sales, np.ones((N_FIT, z_cols)), np.ones((x_rows, 2)), transform=Identity())


class TestPredict:
    """The forecast is the draw-averaged back-transform of the filter's mean and variance."""

    def test_forecast_shape_and_finite(self, fit, design):
        Z, X = design
        forecast = m5_ssp_reg.predict_batch_series_reg(fit, Z[N_FIT:], X[N_FIT:], n_draws=40)
        assert forecast.shape == (N_SERIES, HORIZON)
        assert np.isfinite(forecast).all()

    def test_forecast_is_linear_in_the_future_x_on_an_identity_transform(self, fit, design):
        Z, X = design
        at = lambda scale: m5_ssp_reg.predict_batch_series_reg(
            fit, Z[N_FIT:], scale * X[N_FIT:], n_draws=40
        )  # noqa: E731
        np.testing.assert_allclose(at(2.0) - at(1.0), at(1.0) - at(0.0), atol=1e-4)
        assert np.abs(at(1.0) - at(0.0)).max() > 0.1  # and the covariate moves the forecast

    def test_forecast_moves_with_the_event_in_the_direction_of_beta(self, fit, design):
        Z, X = design
        on = m5_ssp_reg.predict_batch_series_reg(fit, Z[N_FIT:], X[N_FIT:], n_draws=40)
        off_x = X[N_FIT:].copy()
        off_x[:, 0] = 0.0
        off = m5_ssp_reg.predict_batch_series_reg(fit, Z[N_FIT:], off_x, n_draws=40)
        event_days = X[N_FIT:, 0] == 1
        assert (on[0, event_days] - off[0, event_days] > 1.0).all()  # Series 0 has beta = +1.5 on that event.
        assert (on[2, event_days] - off[2, event_days] < -0.3).all()  # Series 2 has -0.8.

    def test_series_order_does_not_matter(self, design, sales):
        Z, X = design
        kwargs = dict(transform=Identity(), **SIGMA_PRIORS, num_warmup=100, num_samples=100, num_chains=2)
        forward = m5_ssp_reg.fit_batch_series_reg(sales[[0, 2]], Z[:N_FIT], X[:N_FIT], **kwargs)
        reverse = m5_ssp_reg.fit_batch_series_reg(sales[[2, 0]], Z[:N_FIT], X[:N_FIT], **kwargs)
        # The two runs share a step size and a seed but not a series order, so the posteriors agree, not the draws.
        np.testing.assert_allclose(
            forward["samples"]["beta"].mean(axis=0), reverse["samples"]["beta"].mean(axis=0)[::-1], atol=0.25
        )

    def test_missing_x_future_is_rejected(self, fit, design):
        Z, _ = design
        with pytest.raises(ValueError, match="covariates"):
            m5_ssp_reg.predict_batch_series_reg(fit, Z[N_FIT:], None)
