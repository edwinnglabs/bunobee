import time
import arviz as az
import xarray as xr
import pandas as pd
import jax.numpy as jnp
import numpy as np

def generate_seed():
    return int(time.time())

def summarize_posteriors(posteriors: xr.Dataset) -> pd.DataFrame:
    """
    """
    # idata = az.from_xarray(posterior=posteriors)
    summary_df = az.summary(posteriors)
    return summary_df

def flatten_front_dim(x: jnp.array, n: int) -> jnp.array:
    new_x = x.reshape(-1, *x.shape[n:])
    return new_x


# def make_fourier_series(t: np.array, period: float, order: int) -> np.array:
#     """
#     Args
#     ----
#     t: array-like, time points at which to evaluate the Fourier series
#     period: float, the period of the seasonality; can be a fractional value
#     order: int, the number of Fourier terms to include
#     """
#     sin_terms = np.array([np.sin(2 * np.pi * i * t / period) for i in range(1, order + 1)])
#     cos_terms = np.array([np.cos(2 * np.pi * i * t / period) for i in range(1, order + 1)])
#     return np.concatenate((sin_terms, cos_terms), axis=0).transpose(1, 0)