"""Tests of the observation families in ``bunobee.models.obs``.

The families' NumPy samplers must agree with the NumPyro distributions they stand for, their priors must register
under the names the models read back, and a gated step must score as a zero under the floored mean. That ``DLT``
fits the same posterior through ``obs=`` as through ``noise_dist=`` is tested in ``test_models.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import pytest
from numpyro import handlers

from bunobee.hierarchy import Level
from bunobee.models.obs import (
    GATE_FLOOR,
    NegBin2,
    Normal,
    ObsFamily,
    Poisson,
    RealObsFamily,
    StudentT,
    gate_mean,
)

N_DRAWS = 400_000


def _trace(fn, seed: int = 0) -> dict:
    return handlers.trace(handlers.seed(fn, jax.random.PRNGKey(seed))).get_trace()


def _level() -> Level:
    """Five series in two groups."""
    return Level("store", np.array([0, 0, 1, 1, 1]), pd.Index(["a", "b"]))


class TestProtocol:
    @pytest.mark.parametrize(
        ("family", "support"),
        [(Normal(), "real"), (StudentT(), "real"), (Poisson(), "count"), (NegBin2(), "count")],
    )
    def test_families_implement_the_protocol(self, family, support):
        assert isinstance(family, ObsFamily)
        assert family.support == support
        assert isinstance(family, RealObsFamily) == (support == "real")

    @pytest.mark.parametrize(
        ("family", "params"),
        [
            (Normal(), {"sigma": 1.0}),
            (StudentT(), {"sigma": 1.0, "nu": 7.0}),
            (Poisson(), {}),
            (NegBin2(), {"concentration": 2.0}),
        ],
    )
    def test_dist_is_the_numpyro_distribution(self, family, params):
        expected = {
            Normal: dist.Normal,
            StudentT: dist.StudentT,
            Poisson: dist.Poisson,
            NegBin2: dist.NegativeBinomial2,
        }[type(family)]
        mean = jnp.array([1.0, 2.0, 3.0])
        assert isinstance(family.dist(mean, params), expected)
        np.testing.assert_allclose(family.dist(mean, params).mean, mean)

    @pytest.mark.parametrize(
        ("make", "match"),
        [
            (lambda: Normal(sigma_scale=0.0), "sigma_scale"),
            (lambda: StudentT(min_nu=10.0, max_nu=5.0), "min_nu"),
            (lambda: StudentT(min_nu=0.0), "min_nu"),
            (lambda: NegBin2(log_concentration_scale=-1.0), "log_concentration_scale"),
            (lambda: NegBin2(dispersion_pool_by="store"), "Level"),
            (lambda: NegBin2(deviation_by=_level()), "needs dispersion_pool_by"),
            (lambda: NegBin2(dispersion_pool_by=_level(), deviation_by="dept"), "Level"),
            (lambda: NegBin2(dispersion_pool_by=_level(), deviation_by=_level(), deviation_scale=0.0), "deviation"),
        ],
    )
    def test_rejects_bad_config(self, make, match):
        with pytest.raises(ValueError, match=match):
            make()


class TestSampleParams:
    @pytest.mark.parametrize(
        ("family", "sites"),
        [
            (Normal(), ["sigma"]),
            (StudentT(), ["sigma", "nu"]),
            (Poisson(), []),
            (NegBin2(), ["log_concentration", "concentration"]),
        ],
    )
    @pytest.mark.parametrize("name", [None, "obs"])
    def test_site_names(self, family, sites, name):
        trace = _trace(lambda: family.sample_params(name))
        prefix = "" if name is None else f"{name}/"
        assert list(trace) == [prefix + site for site in sites]

    def test_returns_the_sampled_parameters(self):
        def model():
            return StudentT(min_nu=3.0, max_nu=30.0, sigma_scale=2.0).sample_params(None)

        trace = _trace(model)
        params = handlers.seed(model, jax.random.PRNGKey(0))()
        assert set(params) == set(StudentT.param_names)
        assert trace["sigma"]["fn"].scale == 2.0
        assert (trace["nu"]["fn"].low, trace["nu"]["fn"].high) == (3.0, 30.0)
        np.testing.assert_allclose(params["nu"], trace["nu"]["value"])

    def test_negbin2_one_concentration_without_a_series_plate(self):
        trace = _trace(lambda: NegBin2().sample_params("obs"))
        assert trace["obs/concentration"]["value"].shape == ()
        np.testing.assert_allclose(
            trace["obs/concentration"]["value"], jnp.exp(trace["obs/log_concentration"]["value"])
        )

    def test_negbin2_one_concentration_per_series(self):
        def model():
            return NegBin2().sample_params("obs", series=numpyro.plate("series", 5))

        trace = _trace(model)
        assert trace["obs/log_concentration"]["value"].shape == (5,)
        assert trace["obs/concentration"]["value"].shape == (5,)

    def test_negbin2_pooled_by_level_shares_the_group_dispersion(self):
        level = _level()
        trace = _trace(lambda: NegBin2(dispersion_pool_by=level).sample_params("obs"))
        log_group = trace["obs/log_concentration"]["value"]
        assert log_group.shape == (level.n_groups,)
        assert "obs/dispersion_groups" in trace
        np.testing.assert_allclose(trace["obs/concentration"]["value"], jnp.exp(log_group)[level.group_idx])

    def test_negbin2_pooled_follows_a_subsampled_series_plate(self):
        level = _level()

        def model():
            series = numpyro.plate("series", level.n_series, subsample_size=3)
            with series as idx:
                pass
            return NegBin2(dispersion_pool_by=level).sample_params("obs", series=series), idx

        params, idx = handlers.seed(model, jax.random.PRNGKey(1))()
        trace = _trace(model, seed=1)
        log_group = trace["obs/log_concentration"]["value"]
        assert params["concentration"].shape == (3,)
        np.testing.assert_allclose(params["concentration"], jnp.exp(log_group)[level.group_idx[np.asarray(idx)]])

    def test_negbin2_deviation_by_draws_each_series_around_its_group(self):
        group = _level()
        spread = Level("dept", np.array([0, 1, 0, 1, 1]), pd.Index(["x", "y"]))
        family = NegBin2(dispersion_pool_by=group, deviation_by=spread, deviation_scale=0.5)

        def model():
            series = numpyro.plate("series", group.n_series, subsample_size=3)
            return family.sample_params("obs", series=series), series._indices

        (params, idx), trace = handlers.seed(model, jax.random.PRNGKey(2))(), _trace(model, seed=2)
        idx = np.asarray(idx)
        site = trace["obs/log_concentration"]
        assert trace["obs/log_concentration_group"]["value"].shape == (group.n_groups,)
        assert trace["obs/deviation_scale"]["value"].shape == (spread.n_groups,)
        assert site["value"].shape == (3,)
        assert [frame.name for frame in site["cond_indep_stack"]] == ["series"]
        # Each series is Normal(its group's log-concentration, its department's spread).
        loc = trace["obs/log_concentration_group"]["value"][group.group_idx[idx]]
        scale = trace["obs/deviation_scale"]["value"][spread.group_idx[idx]]
        np.testing.assert_allclose(site["fn"].loc, loc)
        np.testing.assert_allclose(site["fn"].scale, scale)
        np.testing.assert_allclose(params["concentration"], jnp.exp(site["value"]))

    def test_negbin2_deviation_by_needs_the_series_plate(self):
        family = NegBin2(dispersion_pool_by=_level(), deviation_by=_level())
        with pytest.raises(ValueError, match="series="):
            _trace(lambda: family.sample_params("obs"))


class TestSamplers:
    @pytest.mark.parametrize(
        ("family", "params", "expected"),
        [
            (Poisson(), {}, lambda mean, p: dist.Poisson(mean)),
            (NegBin2(), {"concentration": 0.7}, lambda mean, p: dist.NegativeBinomial2(mean, p["concentration"])),
            (NegBin2(), {"concentration": 25.0}, lambda mean, p: dist.NegativeBinomial2(mean, p["concentration"])),
            (Normal(), {"sigma": 1.5}, lambda mean, p: dist.Normal(mean, p["sigma"])),
            (StudentT(), {"sigma": 1.5, "nu": 8.0}, lambda mean, p: dist.StudentT(p["nu"], mean, p["sigma"])),
        ],
    )
    @pytest.mark.parametrize("mean", [0.3, 4.0, 60.0])
    def test_matches_numpyro_mean_and_variance(self, family, params, expected, mean):
        draws = family.sample(np.random.default_rng(0), np.full(N_DRAWS, mean), params)
        reference = expected(jnp.asarray(mean), params)
        target_mean, target_var = float(reference.mean), float(reference.variance)
        assert draws.shape == (N_DRAWS,) and draws.dtype == np.float64
        # 6 standard errors of the sample mean and, loosely, of the sample variance
        assert abs(draws.mean() - target_mean) < 6 * np.sqrt(target_var / N_DRAWS)
        np.testing.assert_allclose(draws.var(), target_var, rtol=0.05)

    @pytest.mark.parametrize("family", [Poisson(), NegBin2()])
    def test_count_draws_are_non_negative_integers(self, family):
        params = {} if isinstance(family, Poisson) else {"concentration": 1.0}
        draws = family.sample(np.random.default_rng(0), np.full(1000, 2.0), params)
        assert (draws >= 0).all() and (draws == np.round(draws)).all()

    def test_broadcasts_posterior_parameters_over_steps(self):
        # (n_samples, 1) concentration against an (n_samples, n_steps) mean
        concentration = np.array([[0.5], [50.0]])
        draws = NegBin2().sample(np.random.default_rng(0), np.full((2, 20_000), 5.0), {"concentration": concentration})
        assert draws.shape == (2, 20_000)
        assert draws[0].var() > 3 * draws[1].var()

    def test_noise_matches_the_old_dlt_draws(self):
        key = jax.random.PRNGKey(3)
        sigma, nu = np.array([[1.0], [2.0]]), np.array([[4.0], [9.0]])
        np.testing.assert_array_equal(
            Normal().noise(key, {"sigma": sigma}, (2, 5)), jax.random.normal(key, shape=(2, 5)) * sigma
        )
        np.testing.assert_array_equal(
            StudentT().noise(key, {"sigma": sigma, "nu": nu}, (2, 5)),
            jax.random.t(key, df=nu, shape=(2, 5)) * sigma,
        )


class TestGate:
    def test_gate_mean(self):
        mean = jnp.array([2.0, 3.0])
        assert gate_mean(mean) is mean
        np.testing.assert_allclose(gate_mean(mean, jnp.array([1.0, 0.0])), [2.0 + GATE_FLOOR, GATE_FLOOR])

    @pytest.mark.parametrize(
        ("family", "params"), [(Poisson(), {}), (NegBin2(), {"concentration": 1.5})], ids=["poisson", "negbin2"]
    )
    def test_gated_step_scores_a_zero_under_the_floored_mean(self, family, params):
        mean = jnp.array([4.0, 4.0, 7.0])
        gate = jnp.array([1.0, 0.0, 0.0])
        y = jnp.array([3.0, 0.0, 0.0])
        log_lik = family.dist(mean, params, gate=gate).log_prob(y)
        zero_under_floor = family.dist(jnp.full(2, GATE_FLOOR), params).log_prob(jnp.zeros(2))
        np.testing.assert_allclose(log_lik[1:], zero_under_floor)
        np.testing.assert_allclose(log_lik[1:], 0.0, atol=1e-5)  # a zero is all but certain
        # the listed step keeps its own mean, up to the floor
        np.testing.assert_allclose(log_lik[0], family.dist(mean[0] + GATE_FLOOR, params).log_prob(3.0))

    def test_gated_likelihood_in_a_model(self):
        y = jnp.array([2.0, 0.0, 5.0])
        gate = jnp.array([1.0, 0.0, 1.0])

        def model():
            params = NegBin2().sample_params("obs")
            numpyro.sample("y", NegBin2().dist(jnp.full(3, 3.0), params, gate=gate), obs=y)

        site = _trace(model)["y"]
        concentration = _trace(model)["obs/concentration"]["value"]
        expected = dist.NegativeBinomial2(GATE_FLOOR, concentration).log_prob(0.0)
        np.testing.assert_allclose(site["fn"].log_prob(y)[1], expected)

    @pytest.mark.parametrize(
        ("family", "params"), [(Poisson(), {}), (NegBin2(), {"concentration": 1.5})], ids=["poisson", "negbin2"]
    )
    def test_gated_forecasts_are_zero(self, family, params):
        mean = np.full((1000, 2), 8.0)
        draws = family.sample(np.random.default_rng(0), mean, params, gate=np.array([1.0, 0.0]))
        assert (draws[:, 1] == 0).all()
        assert draws[:, 0].mean() > 5
