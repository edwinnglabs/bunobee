import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
import logging
import xarray as xr

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .utils import flatten_front_dim
from .models.dlt import run_dlt_model
from .regression import RegressionScheme

logger = logging.getLogger("bunobee")


def slice_trend_single(trend_comp: jnp.ndarray, h: int) -> jnp.ndarray:
    """Slices a single trend into overlapping windows of size h.

    Args
    ----
    trend_comp: array representing the trend component with shape (n_steps, )
    h: Size of the window to slice the trend into.

    Returns
    -------
    A 2D array of shape (n_windows, h) where each row is a window of size h where
    num_windows = n_steps - h + 1.
    """
    n_steps = trend_comp.shape[0]
    num_windows = n_steps - h + 1

    def get_window(t_idx):
        return jax.lax.dynamic_slice(trend_comp, start_indices=(t_idx,), slice_sizes=(h,))

    t_indices = jnp.arange(num_windows)
    # (n_windows, h)
    return vmap(get_window)(t_indices)


def slice_trend(trend_comp, h):
    """Slices multiple trends into overlapping windows of size h.
    Args
    ----
    trend_comp: array representing multiple trend components with shape (N_samples, n_steps)
    h: Size of the window to slice the trends into.

    Returns
    -------
    A 3D array of shape (N_samples, n_windows, h) where each row is a window of size h
    and num_windows = n_steps - h + 1 for each trend.
    """
    # trends: shape (N_samples, T)
    return vmap(lambda trend_comp: slice_trend_single(trend_comp, h))(trend_comp)


def generate_forecast_span_samples(idata: xr.Dataset, h: int) -> jnp.ndarray:
    # how to do a sliding window prediction?
    dlt_comp = flatten_front_dim(idata["dlt_comp"].to_numpy(), n=2)
    reg_comp = flatten_front_dim(idata["reg_comp"].to_numpy(), n=2)

    logger.debug("dlt_comp shape: %s", dlt_comp.shape)
    logger.debug("reg_comp shape: %s", reg_comp.shape)

    dlt_comp_slice = dlt_comp[:, : -(h - 1), None]
    reg_comp_slice = slice_trend(reg_comp, h=h)

    # (n_samples, n_windows, h)
    yhat_span = dlt_comp_slice + reg_comp_slice
    return yhat_span


def compute_log_likelihood(yhat_span, sigma, y):
    """Computes the log likelihood of the forecasted span given the observations.

    Args
    ----
    yhat_span: Forecasted span of shape (n_samples, n_windows, h).
    sigma: Standard deviation of the noise, shape (n_samples,)
    y_slice: Observations for the forecasted span, shape (n_windows, h).

    Returns
    -------
    A 3D array of log probabilities of shape (n_samples, n_windows, h).
    """
    h = yhat_span.shape[-1]
    # reshape for broadcasting
    # (n_samples, 1, 1)
    sigma_broadcast = sigma[:, None, None]
    # (n_windos, h)
    y_slice = slice_trend_single(y, h=h)
    # (1, n_windows, h)
    y_slice_broadcast = y_slice[None, :, :]

    # compute squared error
    sq_error = (y_slice_broadcast - yhat_span) ** 2

    # compute log likelihood per (s, t, h)
    log_prob = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(sigma_broadcast) - 0.5 * sq_error / (sigma_broadcast**2)

    return log_prob


def compute_wbic(loglk: jnp.ndarray) -> float:
    """Compute an approximate Widely applicable Bayesian Information Criterion (WBIC) from log likelihood.

    WBIC is the expected negative log likelihood under the posterior tempered at ``beta = 1 / log(n)``.
    The draws here come from the ordinary (untempered, ``beta = 1``) posterior, so the score is an
    approximation of WBIC, good enough to rank hyper-param candidates; a true tempered WBIC is tracked in
    https://github.com/edwinnglabs/bunobee/issues/135. It also averages rather than sums the log likelihood
    over observations, so scores stay comparable across series lengths.

    Args
    ----
    loglk: array-like, log likelihood per samples per obs; should be with values of shape (n_samples, n_steps, h)
    """
    # in original paper, they use sum but it leads to unstable scale due to different size of datasets
    # we use mean instead
    loglk_per_sample = jnp.nanmean(loglk, axis=(-1, -2))
    # (n_steps * h, )
    nobs = loglk.shape[-1] * loglk.shape[-2]
    beta = 1.0 / jnp.log(nobs)
    wbic = -(1.0 / beta) * jnp.nanmean(loglk_per_sample)
    wbic = float(wbic)
    return wbic


DEFAULT_MCMC_RUN_ARGS = {"num_warmup": 200, "num_samples": 200, "num_chains": 1, "progress_bar": False}


def run_dlt_model_and_compute_wbic(
    params: Tuple,
    y: np.ndarray,
    rng_key: jnp.ndarray,
    mcmc_run_args: Dict[str, Any],
    regression_scheme: Optional[RegressionScheme] = None,
    covariates_df: Optional[pd.DataFrame] = None,
    h: int = 12,
) -> float:
    """Fit the DLT model with the given hyper-params and return its WBIC over ``h``-step windows.

    The score samples the ordinary posterior, so it is an approximation of WBIC (see ``compute_wbic``).

    Args
    ----
    params: ``(lev_sm, slp_sm, theta)``.
    y: Observations, shape ``(n_steps,)``.
    rng_key: JAX random key forwarded to ``run_dlt_model``.
    mcmc_run_args: Keyword arguments forwarded to ``numpyro.infer.MCMC``.
    regression_scheme: Optional ``RegressionScheme`` for the regressors.
    covariates_df: Optional covariates, required when ``regression_scheme`` is provided.
    h: Forecast horizon of the sliding windows.
    """
    lev_sm, slp_sm, theta = params
    logger.info(f"Trying lev_sm={lev_sm:.4f}, slp_sm={slp_sm:.4f}, theta={theta:.4f}")

    idata = run_dlt_model(
        rng_key=rng_key,
        lev_sm=lev_sm,
        slp_sm=slp_sm,
        theta=theta,
        y=y,
        mcmc_run_args=mcmc_run_args,
        regression_scheme=regression_scheme,
        covariates_df=covariates_df,
    )

    yhat_span = generate_forecast_span_samples(idata=idata, h=h)
    loglk = compute_log_likelihood(
        yhat_span=yhat_span,
        sigma=flatten_front_dim(idata["sigma"].to_numpy(), n=2),
        y=y,
    )
    wbic = compute_wbic(loglk)

    logger.info(f"WBIC: {wbic:.4f}")
    return wbic


# name -> (low, high, prior); the order is the order of ``params`` in ``run_dlt_model_and_compute_wbic``
DEFAULT_SEARCH_SPACE = {
    "lev_sm": (0.0001, 0.1, "log-uniform"),
    "slp_sm": (0.001, 0.1, "log-uniform"),
    "theta": (0.0, 1.0, "uniform"),
}
SEARCH_SPACE_PRIORS = ("log-uniform", "uniform")


def resolve_search_space(
    search_space: Optional[Dict[str, Tuple]] = None,
) -> Dict[str, Tuple[float, float, str]]:
    """Merge a user search space into ``DEFAULT_SEARCH_SPACE``.

    Args
    ----
    search_space: ``{name: (low, high)}`` or ``{name: (low, high, prior)}`` with ``prior`` in
        ``SEARCH_SPACE_PRIORS``. A ``(low, high)`` entry keeps the parameter's default prior; a name left out
        keeps its default range and prior. ``None`` returns the defaults.

    Returns
    -------
    Dict of ``name -> (low, high, prior)`` for every DLT hyper-param, in ``DEFAULT_SEARCH_SPACE`` order.

    Raises
    ------
    ValueError: On an unknown parameter name, an entry that is not a 2- or 3-tuple, or an unknown prior.
    """
    search_space = search_space or {}
    unknown = sorted(set(search_space) - set(DEFAULT_SEARCH_SPACE))
    if unknown:
        raise ValueError(f"Unknown hyper-param(s) {unknown} in search_space; expected {list(DEFAULT_SEARCH_SPACE)}.")

    resolved = {}
    for name, (low, high, prior) in DEFAULT_SEARCH_SPACE.items():
        if name in search_space:
            spec = tuple(search_space[name])
            if len(spec) == 2:
                low, high = spec
            elif len(spec) == 3:
                low, high, prior = spec
            else:
                raise ValueError(
                    f"search_space[{name!r}] must be (low, high) or (low, high, prior), got {search_space[name]!r}."
                )
            if prior not in SEARCH_SPACE_PRIORS:
                raise ValueError(f"search_space[{name!r}] prior must be one of {SEARCH_SPACE_PRIORS}, got {prior!r}.")
        resolved[name] = (low, high, prior)
    return resolved


def hyper_tuning_dlt_with_wbic(
    y: np.ndarray,
    regression_scheme: Optional[RegressionScheme] = None,
    covariates_df: Optional[pd.DataFrame] = None,
    mcmc_run_args: Optional[Dict[str, Any]] = None,
    # forecast horizon
    h: int = 12,
    n_calls: int = 15,
    random_state: int = 42,
    search_space: Optional[Dict[str, Tuple]] = None,
) -> Dict[str, Any]:
    """Tune the DLT hyper-params (``lev_sm``, ``slp_sm``, ``theta``) by minimizing WBIC with Bayesian optimization.

    Requires ``scikit-optimize`` (``pip install bunobee[tune]``).

    The score samples the ordinary posterior rather than the posterior tempered at ``beta = 1 / log(n)``, so
    it is an approximation of WBIC; it is used only to rank the candidates. A true tempered WBIC is tracked in
    https://github.com/edwinnglabs/bunobee/issues/135.

    Args
    ----
    y: Observations, shape ``(n_steps,)``.
    regression_scheme: Optional ``RegressionScheme`` for the regressors.
    covariates_df: Optional covariates, required when ``regression_scheme`` is provided.
    mcmc_run_args: Keyword arguments forwarded to ``numpyro.infer.MCMC``; defaults to ``DEFAULT_MCMC_RUN_ARGS``.
    h: Forecast horizon of the sliding windows used to score each candidate.
    n_calls: Number of hyper-param candidates evaluated (``skopt`` requires at least 10).
    random_state: Seed for both the optimizer and the MCMC ``rng_key``.
    search_space: Optional ``{name: (low, high)}`` or ``{name: (low, high, prior)}`` with ``prior``
        ``"log-uniform"`` or ``"uniform"``. A name left out keeps its default from ``DEFAULT_SEARCH_SPACE``
        (``lev_sm`` in [0.0001, 0.1] and ``slp_sm`` in [0.001, 0.1] log-uniform, ``theta`` in [0, 1] uniform),
        as does a prior left out; ``None`` tunes over the defaults.

    Returns
    -------
    Dict with ``lev_sm``, ``slp_sm``, ``theta`` and ``result`` (the raw ``skopt`` result).

    Raises
    ------
    ValueError: If ``search_space`` names an unknown hyper-param or has a malformed entry.
    """
    resolved_space = resolve_search_space(search_space)
    logger.info(f"Starting hyperparameter tuning for DLT model using WBIC (h={h}, n_calls={n_calls})")
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError as e:
        raise ImportError(
            "scikit-optimize is required for hyperparameter tuning. Install it with `pip install bunobee[tune]`."
        ) from e

    if mcmc_run_args is None:
        mcmc_run_args = DEFAULT_MCMC_RUN_ARGS
    rng_key = jax.random.PRNGKey(random_state)

    # Define the hyperparam space
    dimensions = [Real(low, high, prior=prior, name=name) for name, (low, high, prior) in resolved_space.items()]

    # Run Bayesian Optimization
    result = gp_minimize(
        func=lambda params: run_dlt_model_and_compute_wbic(
            params=params,
            y=y,
            rng_key=rng_key,
            mcmc_run_args=mcmc_run_args,
            regression_scheme=regression_scheme,
            covariates_df=covariates_df,
            h=h,
        ),
        dimensions=dimensions,
        n_calls=n_calls,
        random_state=random_state,
    )

    lev_sm, slp_sm, theta = result.x
    return {"lev_sm": lev_sm, "slp_sm": slp_sm, "theta": theta, "result": result}
