"""Tests of :mod:`bunobee.models.inference`: NUTS, SVI and MAP behind ``Forecaster``.

``Pooled`` is a toy model with a series plate: a shared mean ``mu`` and one offset ``alpha`` per series. It checks the
parts no real model exercises yet, subsampling the series plate above all. The fits are tiny on purpose.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
import xarray as xr
from numpyro import distributions as dist
from numpyro.infer import MCMC
from numpyro.infer import NUTS as NUTSKernel

from bunobee.models.base import Forecaster, NotFittedError
from bunobee.models.inference import MAP, NUTS, SVI, Inference, Posterior

N_STEPS, N_SERIES = 30, 12
NUTS_ARGS = {"num_warmup": 30, "num_samples": 30}


class Pooled(Forecaster):
    """``y[t, j] ~ N(mu + alpha[j], 0.5)`` with ``alpha[j] ~ N(0, 1)``, over a subsamplable ``"series"`` plate."""

    inference_methods = (NUTS, SVI, MAP)
    series_plate = "series"

    def __init__(self, inference: Inference | None = None) -> None:
        super().__init__(inference)

    def model(self, y: jnp.ndarray) -> None:
        mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
        with numpyro.plate(self.series_plate, y.shape[1]) as idx:
            alpha = numpyro.sample("alpha", dist.Normal(0.0, 1.0))
            numpyro.deterministic("level", mu + alpha)
            with numpyro.plate("time", y.shape[0], dim=-2):
                numpyro.sample("obs", dist.Normal(mu + alpha, 0.5), obs=y[:, idx])

    def fit(self, y: np.ndarray, *, mcmc_args=None, seed: int = 0) -> Pooled:
        return self._fit({"y": jnp.asarray(y)}, mcmc_args=mcmc_args, seed=seed)

    def _to_idata(self, samples, inputs, context) -> xr.Dataset:
        dims = {"mu": ("chain", "draw"), "alpha": ("chain", "draw", "series"), "level": ("chain", "draw", "series")}
        return xr.Dataset({name: (dims[name], np.asarray(value)) for name, value in samples.items()})

    def predict(self) -> xr.Dataset:
        return self.idata_[["level"]]


class NoPlate(Pooled):
    """``Pooled`` without a declared series plate, so it cannot be subsampled."""

    series_plate = None

    def model(self, y: jnp.ndarray) -> None:
        mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
        numpyro.sample("obs", dist.Normal(mu, 0.5), obs=y)


@pytest.fixture(scope="module")
def pooled_y() -> np.ndarray:
    rng = np.random.default_rng(0)
    alpha = np.linspace(-1.0, 1.0, N_SERIES)
    return 2.0 + alpha + rng.normal(scale=0.5, size=(N_STEPS, N_SERIES))


class TestNUTS:
    def test_defaults_match_the_old_mcmc_args(self):
        assert NUTS().mcmc_args == {"num_warmup": 500, "num_samples": 500, "num_chains": 1, "progress_bar": False}

    def test_updated_puts_mcmc_args_on_top(self):
        nuts = NUTS(num_chains=2, thinning=2, kernel_args={"target_accept_prob": 0.9}).updated({"num_samples": 10})
        assert nuts.mcmc_args == {
            "num_warmup": 500,
            "num_samples": 10,
            "num_chains": 2,
            "progress_bar": False,
            "thinning": 2,
        }
        assert nuts.kernel_args == {"target_accept_prob": 0.9}

    def test_run_is_mcmc_with_the_model_init_strategy(self, pooled_y):
        model = Pooled(NUTS(**NUTS_ARGS)).fit(pooled_y, seed=4)
        kernel = NUTSKernel(model.model, init_strategy=model._init_strategy())
        mcmc = MCMC(kernel, **NUTS(**NUTS_ARGS).mcmc_args)
        mcmc.run(jax.random.PRNGKey(4), y=jnp.asarray(pooled_y))
        expected = mcmc.get_samples(group_by_chain=True)
        for name in ("mu", "alpha", "level"):
            np.testing.assert_array_equal(model.idata_[name].values, np.asarray(expected[name]))
        assert model.idata_.attrs["inference"] == "NUTS"
        assert isinstance(model.mcmc_, MCMC)


class TestSVI:
    @pytest.mark.parametrize("inference", [SVI(steps=300, num_draws=20), MAP(steps=300)], ids=["svi", "map"])
    def test_fit_keeps_chain_and_draw_dims(self, pooled_y, inference):
        model = Pooled(inference).fit(pooled_y)
        idata = model.idata_
        assert idata["alpha"].dims == ("chain", "draw", "series")
        assert idata.sizes["chain"] == 1
        assert idata.sizes["draw"] == inference.num_draws
        assert idata.attrs["inference"] == inference.name
        assert model.losses_.shape == (300,)
        assert set(model.params_) and callable(model.guide_)
        # The deterministic sites are computed at the guide's draws.
        np.testing.assert_allclose(idata["level"].values, idata["mu"].values[..., None] + idata["alpha"].values)

    def test_recovers_the_series_means(self, pooled_y):
        idata = Pooled(MAP(steps=1000, peak_lr=0.1)).fit(pooled_y).idata_
        np.testing.assert_allclose(idata["level"].values[0, 0], pooled_y.mean(axis=0), atol=0.1)

    def test_subsample_fits_more_series_than_k(self, pooled_y):
        k = 4
        model = Pooled(SVI(subsample=k, steps=1500, peak_lr=0.1, num_draws=50)).fit(pooled_y)
        idata = model.idata_
        assert idata.sizes["series"] == N_SERIES > k
        # Each step saw k series: the guide was traced on a subsample of the plate.
        assert model.guide_.prototype_trace["alpha"]["value"].shape == (k,)
        assert model.guide_.prototype_trace["series"]["args"] == (N_SERIES, k)
        assert np.isfinite(model.losses_).all()
        # Every series is fitted, not only the last subsample.
        level = idata["level"].mean(("chain", "draw")).values
        np.testing.assert_allclose(level, pooled_y.mean(axis=0), atol=0.3)
        assert model.params_["alpha_auto_loc"].shape == (N_SERIES,)

    def test_subsample_larger_than_the_plate_raises(self, pooled_y):
        with pytest.raises(ValueError, match="more than"):
            Pooled(SVI(subsample=N_SERIES + 1, steps=10)).fit(pooled_y)

    def test_mcmc_args_are_for_nuts_only(self, pooled_y):
        with pytest.raises(ValueError, match="only applies to NUTS"):
            Pooled(MAP(steps=10)).fit(pooled_y, mcmc_args=NUTS_ARGS)

    def test_extras_of_the_other_method_raise(self, pooled_y):
        model = Pooled(MAP(steps=10)).fit(pooled_y)
        with pytest.raises(AttributeError, match="fitted by MAP"):
            model.mcmc_
        nuts = Pooled(NUTS(**NUTS_ARGS)).fit(pooled_y)
        with pytest.raises(AttributeError, match="fitted by NUTS"):
            nuts.guide_

    @pytest.mark.parametrize("name", ["guide_", "params_", "losses_"])
    def test_extras_before_fit_raise(self, name):
        with pytest.raises(NotFittedError):
            getattr(Pooled(SVI()), name)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"guide": "flow"}, "guide"),
            ({"subsample": 0}, "subsample"),
            ({"steps": 0}, "steps"),
            ({"num_draws": 0}, "num_draws"),
            ({"peak_lr": 0.0}, "peak_lr"),
            ({"clip_norm": -1.0}, "clip_norm"),
        ],
    )
    def test_rejects_bad_settings(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            SVI(**kwargs)


class TestAccepts:
    def test_default_is_nuts(self):
        assert isinstance(Pooled().inference, NUTS)

    def test_methods_follow_the_protocol(self):
        for inference in (NUTS(), SVI(), MAP()):
            assert isinstance(inference, Inference)

    def test_unaccepted_method_raises_at_construction(self):
        class NutsOnly(Pooled):
            inference_methods = (NUTS,)

        with pytest.raises(ValueError, match="accepts inference by NUTS"):
            NutsOnly(SVI())

    def test_subsample_without_a_series_plate_raises_at_construction(self):
        with pytest.raises(ValueError, match="no series plate"):
            NoPlate(SVI(subsample=4))
        assert isinstance(NoPlate(SVI()).inference, SVI)

    def test_posterior_is_frozen(self):
        posterior = Posterior({}, "NUTS")
        with pytest.raises(AttributeError):
            posterior.method = "SVI"
