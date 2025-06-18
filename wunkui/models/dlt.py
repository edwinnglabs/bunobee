import logging
import jax.numpy as jnp
import numpy as np
from jax import lax
import jax
import pandas as pd
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median
import xarray as xr
from typing import Optional, Dict, Tuple

from ..utils import generate_seed, flatten_front_dim
from ..regression import RegressionScheme

logger = logging.getLogger("wunkui")

def dlt_transition_step(carry, inputs, lev_sm: float, slp_sm: float, theta: float, oos: bool = False):
    """Damped Local Trend (DLT) transition function
    Args
    ----
    carry: tuple containing the previous level and bias
    inputs: tuple containing the current observation, growth, trend, and seasonality
    lev_sm: level smoothing factor
    slp_sm: slope smoothing factor
    theta: damping factor where each step is computed as:
        dlt_comp_t = lev_prev + theta * slp_prev
    oos: boolean indicating whether the function is used for out-of-sample forecasting

    
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jax import lax
    >>> from wunku.models import dlt_transition_step

    _, res = lax.scan(lambda carry, inputs: dlt_transition_step(carry, inputs, lev_sm, slp_sm, theta), (y[0], 0), y)
    levs, slps, dlt_comp = res
    """
    lev_prev, slp_prev = carry

    # forecast
    dlt_comp_t = lev_prev + theta * slp_prev

    if oos:
        eps_t = inputs
        y_t = dlt_comp_t + eps_t
    else:
        y_t = inputs

    new_lev = jnp.where(
        jnp.isfinite(y_t),
        lev_sm * y_t + (1 - lev_sm) * (dlt_comp_t),
        lev_prev,
    )
    new_slp = jnp.where(
        jnp.isfinite(y_t),
        slp_sm * (new_lev - lev_prev) + (1 - slp_sm) * slp_prev,
        slp_prev,
    )

    return (new_lev, new_slp), (new_lev, new_slp, dlt_comp_t)


def dlt_model(lev_sm, slp_sm, theta, y, covariates=None):
    """Damped Local Trend (DLT) model for time series forecasting.
    Args
    ----
    lev_sm: Level smoothing factor (scalar).
    slp_sm: Slope smoothing factor (scalar).
    theta: Damping factor (scalar).
    y: Observations (1D array with shape (n_steps,))
    """

    sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))
    if covariates is not None:
        coef = numpyro.sample("coef", dist.Normal(0, 0.3).expand([covariates.shape[1]]))
        # (n_steps, )
        reg_comp = jnp.sum(covariates * coef, axis=-1)
    else:
        reg_comp = 0

    # scan with the partial function
    final_states, all_states = lax.scan(
        lambda carry, inputs: dlt_transition_step(carry, inputs, lev_sm, slp_sm, theta), 
        (y[0] - reg_comp[0], 0.), y - reg_comp
    )
    _, _, dlt_comp = all_states
    last_lev, last_slp = final_states
    # mid point estimation
    mu = dlt_comp + reg_comp

    numpyro.deterministic("mu", mu)
    # use for in-sample
    numpyro.deterministic("dlt_comp", dlt_comp)
    numpyro.deterministic("reg_comp", reg_comp)
    # use for out-of-sample
    numpyro.deterministic("last_lev", last_lev)
    numpyro.deterministic("last_slp", last_slp)

    # likelihood
    numpyro.sample("observations", dist.Normal(loc=mu, scale=sigma), obs=y, obs_mask=jnp.isfinite(y))



def run_dlt_model(
    rng_key: jnp.ndarray,
    lev_sm, 
    slp_sm, 
    theta, 
    y, 
    mcmc_run_args: Dict[str, any],
    regression_scheme: RegressionScheme,
    covariates_df: pd.DataFrame = None,
    # seed: Optional[int] = None,
) -> xr.Dataset:
    """Run the DLT model with the provided parameters and data.

    Args
    ----
    lev_sm: Level smoothing factor (scalar).
    slp_sm: Slope smoothing factor (scalar).
    theta: Damping factor (scalar).
    x_seas: Seasonal features (2D array with shape (n_steps, n_seasons)).
    x_glb_trend: Global trend feature (1D array with shape (n_steps,)).
    y: Observations (1D array with shape (n_steps,)).
    seed: Optional; random seed for reproducibility.

    Returns 
    -------
    posteriors_dict: Dictionary containing the posterior samples of the model parameters.
    """
    # # generate seed based on current time stamp
    # if seed is None:
    #     seed = generate_seed()

    # extract regressors matrix, coef loc and scale
    var_name = regression_scheme.scheme.index.to_numpy()
    coef_loc = regression_scheme.scheme['loc_prior'].to_numpy()
    coef_scale = regression_scheme.scheme['scale_prior'].to_numpy()
    covariates = covariates_df.loc[:, var_name].values if covariates_df is not None else None

    logger.info(f"var_name: {var_name}")
    logger.info(f"covariates shape: {covariates.shape}")
    logger.debug(f"coef_loc: {coef_loc}")
    logger.debug(f"coef_scale: {coef_scale}")

    init_strategy = init_to_median(num_samples=10)
    kernel = NUTS(dlt_model, init_strategy=init_strategy)
    mcmc = MCMC(kernel, **mcmc_run_args)

    mcmc.run(
        rng_key, 
        lev_sm=lev_sm, 
        slp_sm=slp_sm, 
        theta=theta,
        y=y,
        covariates=covariates
    )
    
    posteriors_dict = mcmc.get_samples(group_by_chain=True)

    # transform them into xr.Dataset
    n_chains, n_draws, n_steps = posteriors_dict['dlt_comp'].shape

    data_vars = {
        'dlt_comp': (['chain', 'draw', 'time'], posteriors_dict['dlt_comp']),
        'mu': (['chain', 'draw', 'time'], posteriors_dict['mu']),
        'reg_comp': (['chain', 'draw', 'time'], posteriors_dict['reg_comp']),
        'sigma': (['chain', 'draw'], posteriors_dict['sigma']),
        'last_lev': (['chain', 'draw'], posteriors_dict['last_lev']),
        'last_slp': (['chain', 'draw'], posteriors_dict['last_slp']),
    }
    coords={
        'draw': np.arange(n_draws),
        'chain': np.arange(n_chains),
        'time': np.arange(n_steps),
    }

    if regression_scheme is not None:
        # add seasonal regressors posteriors
        data_vars.update({
            'coef': (['chain', 'draw', 'var_name'], posteriors_dict['coef']),
        })
        coords.update({
            'var_name': var_name,
        })
        
    posteriors = xr.Dataset(data_vars=data_vars, coords=coords)

    return posteriors


# TODO: move the quantile function out
# add seed to generate noise where we can replicate
def generate_dlt_comp_samples(
    rng_key: jnp.ndarray,
    posteriors: xr.Dataset, 
    end_step: int, 
    lev_sm: float,
    slp_sm: float,
    theta: float,
) -> jnp.ndarray:
    if posteriors.get("dlt_comp") is None: 
        raise ValueError("Posteriors must contain 'dlt_comp' variable.")
    else:
        # in-sample forecast
        # (n_samples, n__train_steps)
        dlt_comp_is_samples = flatten_front_dim(posteriors["dlt_comp"].to_numpy(), n=2)

    # (n_samples, )
    sigma_samples = flatten_front_dim(posteriors["sigma"].to_numpy(), n=2)
    last_lev = flatten_front_dim(posteriors["last_lev"].to_numpy(), n=2)
    last_slp = flatten_front_dim(posteriors["last_slp"].to_numpy(), n=2)

    n_samples, n_train_steps = dlt_comp_is_samples.shape

    # log the shapes
    logger.debug(f"dlt_comp_is_samples shape: {dlt_comp_is_samples.shape}")
    logger.debug(f"sigma_samples shape: {sigma_samples.shape}")
    logger.debug(f"last_lev shape: {last_lev.shape}")
    logger.debug(f"last_slp shape: {last_slp.shape}")

    # (n_steps, n_samples)
    eps_samples = jax.random.normal(rng_key, shape=(end_step, n_samples)) * sigma_samples

    if end_step > n_train_steps:
        # scan with the partial function
        _, all_states = lax.scan(
            lambda carry, inputs: dlt_transition_step(carry, inputs, lev_sm, slp_sm, theta, oos=True), 
            (last_lev, last_slp), eps_samples[n_train_steps:end_step, :],
        )
        # out-of-sample forecast
        # (n_forecast_steps, n_samples)
        _, _, dlt_comp_oos_samples = all_states

    dlt_comp_oos_samples = jnp.transpose(dlt_comp_oos_samples, axes=(-1, -2))
    
    dlt_comp_samples = jnp.concatenate(
        (dlt_comp_is_samples, dlt_comp_oos_samples), axis=-1
    )

    logger.debug(f"dlt_comp_samples shape: {dlt_comp_samples.shape}")

    return dlt_comp_samples




