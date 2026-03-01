import time
import arviz as az
import xarray as xr
import pandas as pd
import jax.numpy as jnp


def generate_seed():
    return int(time.time())


def summarize_posteriors(posteriors: xr.Dataset) -> pd.DataFrame:
    """ """
    # idata = az.from_xarray(posterior=posteriors)
    summary_df = az.summary(posteriors)
    return summary_df


def flatten_front_dim(x: jnp.array, n: int) -> jnp.array:
    new_x = x.reshape(-1, *x.shape[n:])
    return new_x
