import time
import arviz as az
import xarray as xr
import pandas as pd

def generate_seed():
    return int(time.time())

def summarize_posteriors(posteriors: xr.Dataset) -> pd.DataFrame:
    """
    """
    idata = az.from_dict(posterior=posteriors)
    summary_df = az.summary(idata)
    return summary_df