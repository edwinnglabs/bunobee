import time
import arviz as az
import xarray as xr
import pandas as pd
import jax.numpy as jnp
import numpy as np


def generate_seed():
    return int(time.time())


def summarize_idata(idata: xr.Dataset) -> pd.DataFrame:
    """Summarize an inference dataset with ArviZ.

    Parameters
    ----------
    idata : xr.Dataset
        Inference dataset, with the usual ``(chain, draw, ...)`` layout.

    Returns
    -------
    pd.DataFrame
        ArviZ summary table (mean, sd, HDI, diagnostics) for each variable.
    """
    # idata = az.from_xarray(posterior=idata)
    summary_df = az.summary(idata)
    return summary_df


def flatten_front_dim(x: jnp.array, n: int) -> jnp.array:
    new_x = x.reshape(-1, *x.shape[n:])
    return new_x
