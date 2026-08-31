<div align="center">
  <img src="https://raw.githubusercontent.com/edwinnglabs/bunobee/main/bunobee.png" alt="Bunobee Logo" width="320" />
</div>

# Bunobee

[![PyPI version](https://img.shields.io/pypi/v/bunobee.svg)](https://pypi.org/project/bunobee/)
[![CI](https://github.com/edwinnglabs/bunobee/actions/workflows/ci.yaml/badge.svg)](https://github.com/edwinnglabs/bunobee/actions/workflows/ci.yaml)
[![License: MIT](https://img.shields.io/pypi/l/bunobee.svg)](https://github.com/edwinnglabs/bunobee/blob/main/LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/bunobee.svg)](https://pypi.org/project/bunobee/)

Bunobee is an experimental Bayesian time-series library built on [JAX](https://github.com/google/jax) and [NumPyro](https://github.com/pyro-ppl/numpyro). It currently supports the following model families:

- **Damped Local Trend (DLT)** — an exponential smoothing model with a damped slope component
- **State Space Models (SSP)** — structured state space models with Kalman filtering and smoothing

**Fun fact:** "bunobee" is a word invented by my kids during their childhood to describe the action of the role they were playing.

## Installation

```bash
pip install bunobee
```

### JAX install caveat

Bunobee depends on [JAX](https://github.com/google/jax). `pip install bunobee` pulls in the **CPU** build of JAX, which is enough to get started. If you want GPU/TPU acceleration you must install the matching JAX build yourself, since the wheels differ by platform and accelerator:

```bash
# Example: CUDA 12 GPU build
pip install "jax[cuda12]"
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for the build that matches your hardware.

## Quickstart

### DLT forecast

Fit a Damped Local Trend (DLT) model to a univariate series and forecast:

```python
import jax
import numpy as np
from bunobee.models.dlt import run_dlt_model, make_inference

jax.config.update("jax_enable_x64", True)

# Synthetic upward-trending series
y = np.linspace(10.0, 15.0, 20) + np.random.default_rng(0).normal(scale=0.3, size=20)

rng_key = jax.random.PRNGKey(0)
mcmc_args = {"num_warmup": 200, "num_samples": 200, "num_chains": 1}

# Fit: returns an xarray.Dataset of posterior samples (chain, draw, time)
idata = run_dlt_model(rng_key, lev_sm=0.5, slp_sm=0.5, theta=0.8, y=y, mcmc_run_args=mcmc_args)

# Forecast 10 steps beyond the training horizon
result = make_inference(rng_key, idata, lev_sm=0.5, slp_sm=0.5, theta=0.8, end_step=len(y) + 10)
print(result["forecast_samples"].shape)  # (n_samples, 30)
```

### SSP forecast

Fit a multi-series state space model with the extended Kalman filter, then push it past the end of the sample with `forecast_ssp`. The forecast is a posterior replay: it launches from the last filtered state and walks a driftless random walk forward, one process-noise draw per step and per posterior sample, so the predictive interval widens with the horizon:

```python
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import xarray as xr
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS

from bunobee.models.ssp import build_forecast_design, forecast_ssp
from bunobee.models.ssp.kalman_1d_st_ekf import kalman_filter_1d_ekf_st

n_steps, n_series, horizon = 60, 3, 10
positivity = np.array([False, True])  # the level is free; the spend coefficient is kept positive
rng = np.random.default_rng(0)

# Design (n_steps, n_series, n_states): an intercept plus one covariate per series
spend = rng.uniform(0.5, 1.5, size=(n_steps, n_series, 1))
Z = np.concatenate([np.ones((n_steps, n_series, 1)), spend], axis=-1)
y = (Z * np.array([1.0, 0.5])).sum(-1) + rng.normal(scale=0.1, size=(n_steps, n_series))

def model():
    sigma_h = numpyro.sample("sigma_h", dist.HalfNormal(0.5 * jnp.ones(n_series)))
    sigma_q = numpyro.sample("sigma_q", dist.HalfNormal(0.05 * jnp.ones(2)))
    lp, at, *_ = kalman_filter_1d_ekf_st(
        a0=jnp.zeros(2), P0=jnp.eye(2), Z=jnp.asarray(Z), sigma_h=sigma_h, sigma_q=sigma_q,
        y=jnp.asarray(y), logp=True, exponent=1.0, positivity=jnp.asarray(positivity),
    )
    numpyro.factor("lp", lp)
    numpyro.deterministic("at", at)

mcmc = MCMC(NUTS(model), num_warmup=200, num_samples=200, num_chains=1, progress_bar=False)
mcmc.run(jax.random.PRNGKey(0))
samples = mcmc.get_samples(group_by_chain=True)

# forecast_ssp reads the filtered state path plus the process and observation noise
idata = xr.Dataset(
    {
        "at": (("chain", "draw", "time", "state"), np.asarray(samples["at"])),
        "sigma_q": (("chain", "draw", "state"), np.asarray(samples["sigma_q"])),
        "sigma_h": (("chain", "draw", "series"), np.asarray(samples["sigma_h"])),
        "positivity": (("state",), positivity),
    }
).assign_attrs(exponent=1.0, link="exp")  # link="exp": positivity states are EKF log-intensities

# Continue the design past the sample: the intercept carries forward, future spend is supplied
Z_future = build_forecast_design(Z, horizon, covariates_future=rng.uniform(0.5, 1.5, size=(horizon, n_series, 1)))
forecast = forecast_ssp(idata, Z_future, noise_embed=True, seed=0)
print(forecast["forecast_samples"].shape)  # (200, 10, 3) = (n_sample, horizon, n_series)
```

A worked example on a real multi-series panel — fan chart, state path, and per-series small multiples — is in [`playground/statespace_prototype/ssp_v6_forecast.ipynb`](playground/statespace_prototype/ssp_v6_forecast.ipynb).

## License

Bunobee is released under the [MIT License](LICENSE).
