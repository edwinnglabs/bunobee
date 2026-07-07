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

## License

Bunobee is released under the [MIT License](LICENSE).
