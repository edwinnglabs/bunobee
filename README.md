<div align="center">
  <img src="https://raw.githubusercontent.com/edwinnglabs/bunobee/main/bunobee.png" alt="Bunobee Logo" width="320" />
</div>

# Bunobee

[![PyPI version](https://img.shields.io/pypi/v/bunobee.svg)](https://pypi.org/project/bunobee/)
[![Publish](https://github.com/edwinnglabs/bunobee/actions/workflows/publish.yaml/badge.svg)](https://github.com/edwinnglabs/bunobee/actions/workflows/publish.yaml)
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

`get_forecast_decomposition` breaks the forecast into components for diagnostics: a DataFrame indexed by `time`
with `<component>_p50`, `_p05` and `_p95` columns for `trend`, each regressor, `regression` and `yhat`. Pass
`groups` to sum regressors into one component, e.g. `groups={"seasonality": ["dow_1", "dow_2"]}`:

```python
from bunobee.models.dlt import get_forecast_decomposition

decomp = get_forecast_decomposition(result)
print(decomp[["trend_p50", "yhat_p50"]].tail())
```

`pos_coef_sum_scale` puts a total scale constraint on the positive (`"+"`) regressors, e.g. the spend channels of a
marketing mix model, in a single fit. A `HalfNormal(pos_coef_sum_scale)` factor on the sum of their coefficients is
added on top of each coefficient's own scheme prior, so the total is shrunk while the split across regressors is
learned from the data, which removes the need for a first fit to weight each channel's prior:

```python
idata = run_dlt_model(
    rng_key, lev_sm=0.5, slp_sm=0.5, theta=0.8, y=y, mcmc_run_args=mcmc_args,
    regression_scheme=scheme, covariates_df=covariates_df, pos_coef_sum_scale=0.3,
)
```

`noise_dist="student_t"` is a heavier-tailed backup to the default Normal noise, as in Orbit's DLT, for series whose
outliers pull the Normal fit around. Its degrees of freedom `nu` get a `Uniform(min_nu, max_nu)` prior (default 5 to
40) and are sampled with the rest of the model; `idata["nu"]` holds the posterior, and `make_inference` then draws
Student-t forecast noise:

```python
idata = run_dlt_model(
    rng_key, lev_sm=0.5, slp_sm=0.5, theta=0.8, y=y, mcmc_run_args=mcmc_args, noise_dist="student_t",
)
```

### DLT hyper-parameter tuning

`hyper_tuning_dlt_with_wbic` picks `lev_sm`, `slp_sm` and `theta` by Bayesian optimization (`pip install bunobee[tune]`).
Pass `search_space` as `{name: (low, high)}` or `{name: (low, high, prior)}` with prior `"log-uniform"` or
`"uniform"`; a name left out keeps its default (`lev_sm` in [0.0001, 0.1] and `slp_sm` in [0.001, 0.1]
log-uniform, `theta` in [0, 1] uniform), and an unknown name raises `ValueError`:

```python
from bunobee.hyper_tune import hyper_tuning_dlt_with_wbic

tuned = hyper_tuning_dlt_with_wbic(y, h=4, n_calls=15, search_space={"theta": (0.5, 1.0), "lev_sm": (0.01, 0.5)})
print(tuned["lev_sm"], tuned["slp_sm"], tuned["theta"])
```

The score is an **approximation** of WBIC: it samples the ordinary posterior, not the posterior tempered at
`beta = 1 / log(n)` that WBIC is defined on, and averages the log likelihood over observations. It is good
enough to rank candidates; a true tempered WBIC is tracked in
[#135](https://github.com/edwinnglabs/bunobee/issues/135).

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

A worked example on a real multi-series panel — fan chart, state path, and per-series small multiples — is in [`playground/statespace_prototype/ssp_06_multi_series_forecast.ipynb`](playground/statespace_prototype/ssp_06_multi_series_forecast.ipynb).

### Model classes

`SSP` and `DLT` wrap the same models behind one shape: configure, `fit`, `predict`. `SSP` writes the NumPyro model
above for you from a natural-scale prior. It puts `Beta(2, 10) * sdy` on each series' `sigma_h` and reads `sigma_q`'s
prior from the prior's `sigma_q` block. The filter name fixes the positivity mechanism and the matching
`forecast_ssp` link. `DLT` gives the same posterior as `run_dlt_model` and forecasts with the hyperparameters it was
fitted with. Both `predict`s return `forecast_samples` and `mu_samples` (plus `eps_samples` with `noise_embed`) as
`(sample, time, series)` over the horizon; a `DLT` forecast is one series, and `in_sample=True` adds the training
steps. The posterior is on `idata_` and the sampler on `mcmc_`:

```python
import numpy as np
import xarray as xr

from bunobee.models.dlt import DLT
from bunobee.models.ssp import SSP, build_forecast_design

# Natural-scale prior: a free level and a spend effect kept positive, with Beta(2, 10) * scale on each drift
prior = xr.Dataset(
    {
        "a0": (("state",), np.array([1.0, 0.3])),
        "P0": (("state",), np.array([0.5, 0.1])),
        "positivity": (("state",), np.array([False, True])),
        "sigma_q_alpha_prior": (("state",), np.array([2.0, 2.0])),
        "sigma_q_beta_prior": (("state",), np.array([10.0, 10.0])),
        "sigma_q_scale_prior": (("state",), np.array([0.05, 0.02])),
    },
    coords={"state": ["level", "spend"]},
).assign_attrs(sigma_q_family="beta")

# y, Z, horizon and rng as in the SSP example above
model = SSP(prior, filter="ekf", exponent=1.0).fit(y, Z, mcmc_args={"num_warmup": 200, "num_samples": 200})
Z_future = build_forecast_design(Z, horizon, covariates_future=rng.uniform(0.5, 1.5, size=(horizon, n_series, 1)))
forecast = model.predict(Z_future, noise_embed=True)
print(forecast["forecast_samples"].shape)  # (200, 10, 3) = (n_sample, horizon, n_series)

dlt = DLT(lev_sm=0.5, slp_sm=0.5, theta=0.8).fit(y_dlt, mcmc_args={"num_warmup": 200, "num_samples": 200})
print(dlt.predict(end_step=len(y_dlt) + 10)["forecast_samples"].shape)  # (200, 10, 1) = (n_sample, horizon, 1)
print(dlt.predict(end_step=len(y_dlt) + 10, in_sample=True)["forecast_samples"].shape)  # (200, 30, 1) for 20 steps
```

`filter="linear"` uses `kalman_filter_1d_st` instead, which keeps the positivity states on the natural scale and
floors them. NaN entries of `y` become predict-only steps.

Both models fit by NUTS unless told otherwise. `inference=` swaps in variational inference or a point estimate from
`bunobee.models.inference` (they need the `optim` extra). The posterior keeps its `(chain, draw)` dims either way, so
`predict` works unchanged:

```python
from bunobee.models.inference import MAP, SVI

svi = SSP(prior, inference=SVI(steps=3000, num_draws=200)).fit(y, Z)  # AutoNormal guide, 200 draws as one chain
map_fit = DLT(lev_sm=0.5, slp_sm=0.5, theta=0.8, inference=MAP()).fit(y_dlt)  # one draw at the mode
print(svi.idata_.attrs["inference"], svi.losses_[-1])
```

### Pooled count model for many series

`HierarchicalCount` fits one negative binomial model to thousands of count series at once, by SVI on batches of
series. It is the general form of Juan Orduz's hierarchical M5 state space model. Each series' log mean is its
recent mean plus terms that are pooled over the levels of a `Hierarchy`: an intercept and a walk over 28-day blocks,
a weekday pattern, shocks shared by a store, a covariate such as SNAP, and others. A `gate` of `0` marks the days a
series cannot sell. `predict` returns `(sample, time, series)` count draws that sum up the hierarchy with
`aggregate_draws`. It needs the `optim` extra.

```python
import numpy as np
import pandas as pd

from bunobee import Hierarchy, aggregate_draws, draw_quantiles
from bunobee.models.count import BlockWalk, GroupShocks, HierarchicalCount, Level, Regressor, Seasonal
from bunobee.models.inference import SVI

rng = np.random.default_rng(0)
n_obs, horizon, n_series = 364, 28, 60
meta = pd.DataFrame({"store_id": np.arange(n_series) % 3, "dept_id": np.arange(n_series) % 4})
hierarchy = Hierarchy.from_frame(meta, {"store": ["store_id"], "group": ["store_id", "dept_id"]})
y = rng.poisson(4.0, size=(n_obs, n_series))  # counts, (n_obs, n_series)
snap = (np.arange(n_obs + horizon) % 30 < 10).astype(float)  # 1 on the first 10 days of each month

model = HierarchicalCount(
    level=Level(walk=BlockWalk(block=28, n_blocks=6), shift_by="group"),
    components=[Seasonal(7, pool_by="group"), GroupShocks("store"), Regressor("snap", pool_by="group")],
    hierarchy=hierarchy,
    inference=SVI(subsample=20, steps=1000),  # 20 series per step
)
model.fit(y, {"snap": snap[:n_obs]})
forecast = model.predict({"snap": snap[n_obs:]}, num_draws=200)  # forecast_samples (200, 28, 60)
store_quantiles = draw_quantiles(aggregate_draws(forecast, hierarchy, "store"), [0.05, 0.5, 0.95])
```

The other terms are `DayOfMonth`, `Events` and `Fourier`. `pool_by=None`, the default, gives each series its own
effect. Orduz's M5 model, rebuilt on this class, scores a mean WSPL of 0.16228 (sd 0.0014) on the M5 evaluation
window over seeds 42 to 46. The previous benchmark scores 0.16272 (sd 0.0006).

### Fast MAP fits for many series

For hundreds or thousands of non-negative series (sales, counts, demand), `fit_map_batch` fits the linear state-space
model to all of them in parallel by MAP instead of NUTS, in seconds and without posterior uncertainty. The target is
fitted on a transform scale (`LogTransform` by default, or a Box-Cox / Yeo-Johnson `PowerTransform` with a λ per
series) and mapped back to units. It needs the `optim` extra: `pip install "bunobee[optim]"`.

```python
import jax.numpy as jnp
import numpy as np

from bunobee.models.ssp.map_fit import fit_map_batch, predict_map_batch
from bunobee.regression import make_periodic_dummies
from bunobee.target_transforms import PowerTransform

n_fit, horizon = 200, 28
rng = np.random.default_rng(0)
values = rng.poisson(5.0, size=(100, n_fit)).astype(np.float32)  # 100 non-negative series

# One design for all series: a level plus weekday dummies, built over the fit and forecast days.
Z = np.asarray(make_periodic_dummies(n_fit + horizon, period=7, intercept=True), dtype=np.float32)

fit = fit_map_batch(values, Z=jnp.asarray(Z[:n_fit]), transform=PowerTransform("yeo-johnson"))
forecast = predict_map_batch(fit, Z_future=Z[n_fit:])  # (100, 28), in the units of `values`
```

The recipe was developed on the [M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
data. `bunobee.datasets.load_m5_aggregate()` returns ten aggregated M5 series, and
[`playground/m5_prototype/ssp_m5_aggregate_forecast_seasonality.ipynb`](playground/m5_prototype/ssp_m5_aggregate_forecast_seasonality.ipynb)
fits one with annual and event covariates.

## Development

Notebooks under `playground/` are committed **without outputs**, so the repo stays small and diffs stay
readable. Two layers keep that true:

```bash
pip install nbstripout pre-commit
nbstripout --install     # git clean filter, strips outputs on commit (per clone, run once)
pre-commit install       # same hook, as a pre-commit safety net
```

CI enforces it regardless of local setup: the `notebooks` job runs `nbstripout --verify` on every
tracked `.ipynb` and caps each notebook at 1.5 MiB. A notebook whose rendered plots *are* the point
opts out with `"keep_output": true` in its notebook metadata — `ssp_06_multi_series_forecast.ipynb` is
the only one that does, and the size cap is what keeps that opt-out honest.

## License

Bunobee is released under the [MIT License](LICENSE).
