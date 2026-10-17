"""Unit tests for the sigma priors and the best-parameter tracking of the MAP fitters, ``m5_ssp_optim`` (issue #116).

Each Adam step computes the loss of the current parameters, then updates them. The fitters used to keep the updated
parameters as the best, one step after the loss they were credited with, so a sigma its uniform prior's bound held
was returned just past the bound. These tests check that both fitters return the parameters that achieved the
lowest loss, by recomputing the loss at the returned sigmas, and that none lies outside its prior's support, on
series whose noise and drifts the default priors hold at a bound.

The priors are a fitter argument, ``SigmaPrior``: the defaults are today's uniform bounds, and a half-normal or
log-normal prior has no bound. The fit starts each sigma at its prior's mean.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import distributions as dist

from bunobee.models.m5.transforms import LogTransform
from bunobee.regression import make_periodic_dummies

m5_ssp_optim = pytest.importorskip(
    "bunobee.models.m5.m5_ssp_optim", reason="needs the optional 'm5' extra: pip install 'bunobee[m5]'"
)
SigmaPrior = m5_ssp_optim.SigmaPrior

# Poisson sales from intermittent to dense. On the log target the default priors hold sigma_h at 0.5 for the first
# four series and both drifts at 0.1 for the first three.
N_FIT = 140
LEVELS = np.array([0.2, 0.7, 1.5, 4.0, 15.0])


@pytest.fixture(scope="module")
def sales() -> np.ndarray:
    """Daily sales of the fixed series over the fit days, shape (n_series, N_FIT)."""
    rng = np.random.default_rng(114)
    return rng.poisson(LEVELS[:, None], size=(LEVELS.size, N_FIT)).astype(np.float32)


@pytest.fixture(scope="module")
def Z() -> jnp.ndarray:
    """Weekday dummies with the level, phased as M5's d_1 (a Saturday), shape (N_FIT, 7)."""
    return jnp.asarray(make_periodic_dummies(N_FIT, period=7, phase=5, intercept=True))


def _loss(sales, Z, sigma_h, sigma_q, sigma_h_prior, sigma_q_prior) -> np.ndarray:
    """Negative log posterior of each series at the given sigmas, as the fitters define it, with the log target."""
    transform = LogTransform()
    y = jnp.asarray(transform.forward(sales, transform.fit(sales)))
    n_states = Z.shape[1]
    log_p, *_ = m5_ssp_optim.kalman_filter_1d_batch(
        a0=jnp.zeros(n_states),
        P0=jnp.ones(n_states),
        Z=Z,
        sigma_h=jnp.asarray(sigma_h),
        sigma_q=jnp.asarray(sigma_q),
        y=y,
        logp=True,
        chunk_size=4096,
    )
    log_prior = (
        sigma_h_prior.distribution.log_prob(jnp.asarray(sigma_h))
        + sigma_q_prior.distribution.log_prob(jnp.asarray(sigma_q[:, 0]))
        + sigma_q_prior.distribution.log_prob(jnp.asarray(sigma_q[:, 1]))
    )
    return np.asarray(-(log_p + log_prior))


def _assert_inside(values: np.ndarray, prior: SigmaPrior) -> None:
    """Assert every value lies in the support of a uniform prior."""
    low, high = prior.bounds
    assert np.all((values >= low) & (values <= high)), (values, prior.bounds)


def _step(bound: float, lr: float = 3e-2) -> float:
    """About how far one Adam step moves a sigma at ``bound``: ``lr`` times the softplus' slope there."""
    return lr * -math.expm1(-bound)


class TestBestParams:
    """The fitters return the parameters of the lowest loss, which lie inside the uniform priors' bounds."""

    def test_batch_returns_the_parameters_of_its_best_loss(self, sales, Z):
        result = m5_ssp_optim.fit_batch_series_opt(sales, n_iter=100, Z=Z)

        # The upper bounds hold sigma_h of the first four series and the drifts of the first three: each stops within
        # one Adam step of its bound, where the next step would take it past.
        sigma_h_high, sigma_q_high = m5_ssp_optim.SIGMA_H_BOUNDS[1], m5_ssp_optim.SIGMA_Q_BOUNDS[1]
        assert np.all(result["sigma_h"][:4] > sigma_h_high - _step(sigma_h_high))
        assert np.all(result["sigma_q"][:3] > sigma_q_high - _step(sigma_q_high))
        _assert_inside(result["sigma_h"], m5_ssp_optim.SIGMA_H_PRIOR)
        _assert_inside(result["sigma_q"], m5_ssp_optim.SIGMA_Q_PRIOR)

        loss = _loss(
            sales, Z, result["sigma_h"], result["sigma_q"], m5_ssp_optim.SIGMA_H_PRIOR, m5_ssp_optim.SIGMA_Q_PRIOR
        )
        assert np.all(np.isfinite(result["final_loss"]))
        np.testing.assert_allclose(loss, result["final_loss"], rtol=1e-6)

    @pytest.mark.parametrize("patience", [50, None], ids=["early_stop", "all_steps"])
    def test_single_series_returns_the_parameters_of_its_best_loss(self, sales, Z, patience):
        for series in sales:
            result = m5_ssp_optim.fit_one_series_opt(series, n_iter=100, Z=Z, patience=patience)

            assert result["final_loss"] == min(result["losses"])
            assert np.isfinite(result["final_loss"])
            _assert_inside(np.array([result["sigma_h"]]), m5_ssp_optim.SIGMA_H_PRIOR)
            _assert_inside(result["sigma_q"], m5_ssp_optim.SIGMA_Q_PRIOR)
            loss = _loss(
                series[None],
                Z,
                np.array([result["sigma_h"]], dtype=np.float32),
                result["sigma_q"][None],
                m5_ssp_optim.SIGMA_H_PRIOR,
                m5_ssp_optim.SIGMA_Q_PRIOR,
            )
            np.testing.assert_allclose(loss, [result["final_loss"]], rtol=1e-6)

    def test_batch_with_soft_priors_returns_the_parameters_of_its_best_loss(self, sales, Z):
        priors = {
            "sigma_h_prior": SigmaPrior("half_normal", (1.0,)),
            "sigma_q_prior": SigmaPrior("half_normal", (0.05,)),
        }
        result = m5_ssp_optim.fit_batch_series_opt(sales, n_iter=100, Z=Z, **priors)

        loss = _loss(sales, Z, result["sigma_h"], result["sigma_q"], *priors.values())
        np.testing.assert_allclose(loss, result["final_loss"], rtol=1e-6)


class TestPriors:
    """The fitters take their sigma priors as arguments, by default today's uniform bounds."""

    def test_default_priors_are_the_legacy_bounds_and_start(self):
        assert m5_ssp_optim.SIGMA_H_PRIOR == SigmaPrior("uniform", (0.1, 0.5))
        assert m5_ssp_optim.SIGMA_Q_PRIOR == SigmaPrior("uniform", (0.01, 0.1))
        # The fit used to start at these midpoints, hard-coded.
        assert m5_ssp_optim.SIGMA_H_PRIOR.init == 0.30
        assert m5_ssp_optim.SIGMA_Q_PRIOR.init == 0.055

    @pytest.mark.parametrize(
        ("prior", "expected"),
        [
            (SigmaPrior("uniform", (0.05, 1.5)), dist.Uniform(0.05, 1.5)),
            (SigmaPrior("half_normal", (0.05,)), dist.HalfNormal(0.05)),
            (SigmaPrior("log_normal", (math.log(0.02), 1.0)), dist.LogNormal(math.log(0.02), 1.0)),
        ],
        ids=["uniform", "half_normal", "log_normal"],
    )
    def test_prior_is_its_numpyro_distribution_started_at_its_mean(self, prior, expected):
        values = jnp.array([0.06, 0.2, 1.2])
        np.testing.assert_allclose(prior.distribution.log_prob(values), expected.log_prob(values), rtol=1e-6)
        assert prior.init == pytest.approx(float(expected.mean), rel=1e-6)

    def test_only_a_uniform_prior_has_bounds(self):
        assert SigmaPrior("uniform", (0.05, 1.5)).bounds == (0.05, 1.5)
        assert SigmaPrior("half_normal", (1.0,)).bounds is None
        assert SigmaPrior("log_normal", (0.0, 1.0)).bounds is None

    def test_unknown_family_is_rejected(self):
        with pytest.raises(ValueError, match="family"):
            SigmaPrior("gamma", (2.0, 1.0))

    def test_bounds_hold_the_sigmas_the_fit_returns(self, sales, Z):
        # Narrower bounds than the defaults hold every sigma inside them.
        priors = {
            "sigma_h_prior": SigmaPrior("uniform", (0.2, 0.3)),
            "sigma_q_prior": SigmaPrior("uniform", (0.02, 0.04)),
        }
        result = m5_ssp_optim.fit_batch_series_opt(sales, n_iter=100, Z=Z, **priors)

        _assert_inside(result["sigma_h"], priors["sigma_h_prior"])
        _assert_inside(result["sigma_q"], priors["sigma_q_prior"])

    def test_soft_priors_free_the_sigmas_the_default_bounds_hold(self, sales, Z):
        # Without the cap, the noise of the four series it held goes past 0.5. The states no longer absorb the noise
        # the cap left over, so the drifts the upper bound held fall well below it.
        priors = {
            "sigma_h_prior": SigmaPrior("half_normal", (1.0,)),
            "sigma_q_prior": SigmaPrior("half_normal", (0.05,)),
        }
        result = m5_ssp_optim.fit_batch_series_opt(sales, n_iter=100, Z=Z, **priors)

        assert np.all(result["sigma_h"][:4] > m5_ssp_optim.SIGMA_H_BOUNDS[1])
        assert np.all(result["sigma_q"][:3] < m5_ssp_optim.SIGMA_Q_BOUNDS[1] / 2)

    def test_single_series_fit_takes_the_priors(self, sales, Z):
        priors = {
            "sigma_h_prior": SigmaPrior("uniform", (0.2, 0.3)),
            "sigma_q_prior": SigmaPrior("uniform", (0.02, 0.04)),
        }
        result = m5_ssp_optim.fit_one_series_opt(sales[0], n_iter=100, Z=Z, **priors)

        _assert_inside(np.array([result["sigma_h"]]), priors["sigma_h_prior"])
        _assert_inside(result["sigma_q"], priors["sigma_q_prior"])
