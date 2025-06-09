import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
import logging
import xarray as xr

from typing import Dict, Tuple

from .utils import flatten_front_dim
from .models.dlt import run_dlt_model


logger = logging.getLogger("wunku")

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

def generate_forecast_span_samples(posteriors: xr.Dataset, h: int) -> jnp.ndarray:
    # how to do a sliding window prediction?
    dlt_comp = flatten_front_dim(posteriors["dlt_comp"].to_numpy(), n=2)
    reg_comp = flatten_front_dim(posteriors["reg_comp"].to_numpy(), n=2)

    logger.debug("dlt_comp shape:", dlt_comp.shape)
    logger.debug("reg_comp shape:", reg_comp.shape)

    dlt_comp_slice = dlt_comp[:, :-(h-1), None]
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
    log_prob = -0.5 * jnp.log(2 * jnp.pi) \
               - jnp.log(sigma_broadcast) \
               - 0.5 * sq_error / (sigma_broadcast ** 2)

    return log_prob

def compute_wbic(loglk: jnp.ndarray) -> float:
    """Compute Weighted Bayesian Information Criterion (WBIC) from log likelihood.
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
    wbic = - (1.0 / beta) * jnp.nanmean(loglk_per_sample)
    wbic = float(wbic)
    return wbic


def run_dlt_model_and_compute_wbic(
    params: Tuple, 
    data: Dict[str, np.ndarray],
    h: int = 12
) -> float:
    lev_sm, slp_sm, theta = params
    y = data["y"]
    x_seas = data["x_seas"]
    x_glb_trend = data["x_glb_trend"]
    logger.info(f"Trying lev_sm={lev_sm:.4f}, slp_sm={slp_sm:.4f}, theta={theta:.4f}")

    posteriors = run_dlt_model(
        lev_sm=lev_sm,
        slp_sm=slp_sm,
        theta=theta,
        x_seas=x_seas,
        x_glb_trend=x_glb_trend,
        y=y,
    )
    
    yhat_span = generate_forecast_span_samples(posteriors=posteriors, h=h)
    loglk = compute_log_likelihood(
        yhat_span=yhat_span,
        sigma=flatten_front_dim(posteriors["sigma"].to_numpy(), n=2),
        y=y,
    )
    wbic = compute_wbic(loglk)

    print(f"WBIC: {wbic:.4f}")
    return wbic

def hyper_tuning_dlt_with_wbic(
    data: Dict[str, np.ndarray],
    # forecast horizon
    h: int = 12,  
    n_calls: int = 15,                   
    random_state: int = 42,
):
    print("Starting hyperparameter tuning for DLT model using WBIC...")
    # print args
    print(f"h: {h}, n_calls: {n_calls}, random_state: {random_state}")
    # try import skopt
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError:
        raise ImportError("Please install scikit-optimize to run hyperparameter tuning.")

    # Define the hyperparam space
    search_space = [
        Real(0.0001, 0.1, prior='log-uniform', name='lev_sm'),
        Real(0.001, 0.1, prior='log-uniform', name='slp_sm'),
        Real(0, 1., prior='uniform', name='theta'),
    ]

    # Run Bayesian Optimization
    result = gp_minimize(
        func=lambda params: run_dlt_model_and_compute_wbic(params=params, data=data, h=h),  
        dimensions=search_space,
        n_calls=n_calls,                   
        random_state=random_state
    )

    return result