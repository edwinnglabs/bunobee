import jax
import jax.numpy as jnp
from jax import vmap
import logging

from typing import Dict

from models.dlt import run_dlt_model

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

def generate_forecast_span_samples(posteriors_dict: Dict[str, jnp.ndarray], h) -> jnp.ndarray:
    # how to do a sliding window prediction?
    dlt_comp = posteriors_dict["dlt_comp"]
    reg_comp = posteriors_dict["reg_comp"]

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

def compute_wbic(loglk):
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
    return wbic


def run_dlt_model_and_compute_wbic(params):
    lev_sm, slp_sm, theta = params
    print(f"Trying lev_sm={lev_sm:.4f}, slp_sm={slp_sm:.4f}, theta={theta:.4f}")

    posteriors_dict = run_dlt_model(
        lev_sm=lev_sm,
        slp_sm=slp_sm,
        theta=theta,
        x_seas=params["x_seas"],
        x_glb_trend=params["x_glb_trend"],
        y=params["y"],
    )
    
    dlt_comp = posteriors_dict["dlt_comp"]
    beta_glb_trend = posteriors_dict["beta_glb_trend"]
    beta_seas = posteriors_dict["beta_seas"]
    sigma = posteriors_dict["sigma"]

    reg_comp = np.sum(x_seas * jnp.expand_dims(beta_seas, -2), axis=-1) + x_glb_trend * jnp.expand_dims(beta_glb_trend, -1)

    dlt_comp_slice = dlt_comp[:, :-(h-1), None]
    reg_comp_slice = slice_trend(reg_comp, h=h)
    yhat_span = dlt_comp_slice + reg_comp_slice
    y_slice = slice_trend_single(y, h=h)

    loglk = compute_log_likelihood(yhat_span, sigma, y_slice)
    wbic = compute_wbic(loglk)
    wbic = float(wbic)

    print("WBIC:", wbic)
    return wbic