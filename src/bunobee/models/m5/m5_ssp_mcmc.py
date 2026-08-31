"""M5 single-series SSP model fitted with NUTS.

The forecast side is a thin wrapper over the shared SSP core: ``predict_one_series``
delegates the state propagation and the ``Z_future`` contraction to
:func:`bunobee.models.ssp.forecast.forecast_ssp` and only keeps the m5-specific
back-transform (``exp`` on the log scale, rescaled by ``response_norm``).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro
import xarray as xr
from jax import random
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS

from bunobee.models.ssp.forecast import build_forecast_design, forecast_ssp
from bunobee.models.ssp.kalman_1d import kalman_filter_1d
from bunobee.regression import make_peridoic_dummies

#: Default forecast horizon for the M5 competition (28 days).
HORIZON = 28


def fit_one_series(
    sales: np.ndarray,
    num_warmup: int = 100,
    num_samples: int = 100,
    num_chains: int = 4,
    seed: int = 0,
    Z: jnp.ndarray | None = None,
) -> dict:
    """Fit a local-level + weekly-seasonality state-space model to one series.

    Parameters
    ----------
    sales : np.ndarray
        1-D array of daily unit sales (n_steps,).
    num_warmup : int
        NUTS warmup iterations per chain.
    num_samples : int
        NUTS posterior samples per chain.
    num_chains : int
        Number of MCMC chains.
    seed : int
        PRNG seed for reproducibility.
    Z : jnp.ndarray | None
        (n_steps, n_states) pre-built design matrix. When provided the internal
        dummy build is skipped. Pass ``Z_shared`` to avoid redundant work when
        fitting many series of the same length.

    Returns
    -------
    dict
        Keys: ``posterior_dict`` (MCMC samples), ``response_norm`` (float),
        ``Z`` (design matrix), ``a0``, ``P0``.
    """
    sales_clipped = np.clip(sales, 1e-1, None).astype(np.float32)
    response_norm = float(sales_clipped.mean())
    y = jnp.array(np.log(sales_clipped / response_norm))

    n_steps = len(y)

    if Z is None:
        weekly_dummies = make_peridoic_dummies(n_steps, period=7, drop_first=True)
        Z = jnp.concatenate([jnp.ones((n_steps, 1)), weekly_dummies], axis=1)
    n_states = Z.shape[1]

    a0 = jnp.zeros(n_states)
    P0 = jnp.ones(n_states)

    sigma_q_loc_prior = jnp.array([0.05, 0.01])
    sigma_q_scale_prior = jnp.array([0.05, 0.01])

    def _nuts_fn(a0, P0):
        sigma_h = numpyro.sample(
            "sigma_h",
            dist.TruncatedNormal(0.1, 1.0, high=1.0, low=1e-5),
        )
        sigma_q_raw = numpyro.sample(
            "sigma_q",
            dist.TruncatedNormal(
                sigma_q_loc_prior,
                sigma_q_scale_prior,
                high=0.1,
                low=1e-5,
            ),
        )
        n_seas = n_states - 1
        sigma_q = jnp.concatenate([sigma_q_raw[:1], jnp.repeat(sigma_q_raw[1:], n_seas)])

        lp, at, _, _, _, _ = kalman_filter_1d(
            a0=a0,
            P0=P0,
            sigma_h=sigma_h,
            sigma_q=sigma_q,
            y=y,
            Z=Z,
            logp=True,
        )
        numpyro.factor("lp", lp)
        numpyro.deterministic("at", at)
        numpyro.deterministic("mu", jnp.sum(Z * at, -1))

    rng_key = random.PRNGKey(seed)
    mcmc = MCMC(NUTS(_nuts_fn), num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(random.split(rng_key, 1)[0], a0, P0)

    return {
        "posterior_dict": mcmc.get_samples(),
        "response_norm": response_norm,
        "Z": Z,
        "a0": a0,
        "P0": P0,
    }


def _to_forecast_idata(at_samples: np.ndarray, sigma_h_samples: np.ndarray) -> xr.Dataset:
    """Wrap flat NUTS draws in the ``(chain, draw, ...)`` layout ``forecast_ssp`` reads.

    ``numpyro.MCMC.get_samples()`` returns chain-flattened draws, so a singleton
    ``chain`` axis is prepended. ``sigma_q`` is set to zero: the m5 point forecast
    launches from the terminal filtered state and holds it flat over the horizon
    rather than re-injecting process noise, so a zero process scale makes the
    shared core reproduce that behavior exactly.

    Parameters
    ----------
    at_samples : np.ndarray, shape (n_samples, n_steps, n_states)
        Filtered state draws from the ``at`` deterministic site.
    sigma_h_samples : np.ndarray, shape (n_samples,)
        Observation-noise scale draws.

    Returns
    -------
    xr.Dataset
        Dataset carrying ``at``, ``sigma_q`` and ``sigma_h`` with a leading
        ``(chain, draw)`` pair and a single series.
    """
    at = np.asarray(at_samples, dtype=float)[None]  # (1, n_samples, n_steps, n_states)
    n_samples, n_states = at.shape[1], at.shape[3]
    sigma_h = np.asarray(sigma_h_samples, dtype=float).reshape(1, n_samples, 1)
    sigma_q = np.zeros((1, n_samples, n_states), dtype=float)

    return xr.Dataset(
        data_vars={
            "at": (["chain", "draw", "step", "state"], at),
            "sigma_q": (["chain", "draw", "state"], sigma_q),
            "sigma_h": (["chain", "draw", "series"], sigma_h),
        },
        coords={"chain": np.arange(1), "draw": np.arange(n_samples)},
    )


def predict_one_series(
    fit_result: dict,
    horizon: int = HORIZON,
    Z_future: np.ndarray | None = None,
) -> np.ndarray:
    """Generate point forecast (median) from a fitted single-series model.

    A thin wrapper over :func:`bunobee.models.ssp.forecast.forecast_ssp`: the state
    launch, the ``Z_future`` contraction and the observation-noise draw all come from
    the shared core, and this function only supplies the m5 back-transform
    (``exp`` on the log scale, rescaled by ``response_norm``) and the median reduction.

    The model is linear-Gaussian on ``log(sales / response_norm)``, so ``positivity``
    is all-``False`` and the ``exp(exponent * a)`` state map is bypassed; ``sigma_q``
    is zeroed so the terminal filtered state is held flat over the horizon, matching
    the historical m5 behavior. Draws are not bitwise identical to the pre-refactor
    implementation because ``forecast_ssp`` owns the RNG stream, but the sampling
    distribution — and hence the returned median — is unchanged.

    Parameters
    ----------
    fit_result : dict
        Output of ``fit_one_series()``.
    horizon : int
        Number of future steps to forecast. Ignored when ``Z_future`` is given.
    Z_future : np.ndarray | None
        (horizon, n_states) pre-built future design matrix. When provided the
        internal dummy continuation is skipped. Pass ``Z_future_shared`` to
        avoid redundant work when forecasting many series.

    Returns
    -------
    np.ndarray
        Point forecasts of shape (horizon,).
    """
    posterior_dict = fit_result["posterior_dict"]
    response_norm = fit_result["response_norm"]

    at_samples = np.array(posterior_dict["at"])
    sigma_h_samples = np.array(posterior_dict["sigma_h"])

    n_steps, n_states = at_samples.shape[1], at_samples.shape[2]

    if Z_future is None:
        Z_train = np.asarray(fit_result["Z"], dtype=float)[:n_steps]
        Z_future = build_forecast_design(
            Z_train,
            horizon,
            periodic={"columns": slice(1, None), "period": 7, "drop_first": True},
        )[:, 0, :]

    idata = _to_forecast_idata(at_samples, sigma_h_samples)
    forecast = forecast_ssp(
        idata,
        np.asarray(Z_future, dtype=float)[:, None, :],  # (horizon, n_series=1, n_states)
        positivity=np.zeros(n_states, dtype=bool),
        noise_embed=True,
        transform_callback=lambda samples: np.exp(samples) * response_norm,
        seed=42,
    )

    yhat_samples = forecast["forecast_samples"].to_numpy()[:, :, 0]  # (n_samples, horizon)

    return np.median(yhat_samples, axis=0)
