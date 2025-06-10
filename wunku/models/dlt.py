import logging
import jax.numpy as jnp
import numpy as np
from jax import lax
import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median
import xarray as xr
from typing import Optional, Dict

from ..utils import generate_seed, flatten_front_dim

logger = logging.getLogger("wunku")

def dlt_transition_step(carry, inputs, lev_sm, slp_sm, theta):
    """Damped Local Trend (DLT) transition function
    Args
    ----
    carry: tuple containing the previous level and bias
    inputs: tuple containing the current observation, growth, trend, and seasonality
    lev_sm: level smoothing factor
    slp_sm: slope smoothing factor
    theta: damping factor

    
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jax import lax
    >>> from wunku.models import dlt_transition_step

    _, res = lax.scan(lambda carry, inputs: dlt_transition_step(carry, inputs, lev_sm, slp_sm, theta), (y[0], 0), y)
    levs, slps, dlt_comp = res
    """
    lev_prev, slp_prev = carry
    y_t = inputs

    # forecast
    dlt_comp_t = lev_prev + theta * slp_prev

    def update_state(_):
        new_lev = lev_sm * y_t + (1 - lev_sm) * (lev_prev + theta * slp_prev)
        new_slp = slp_sm * (new_lev - lev_prev) + (1 - slp_sm) * slp_prev
        return new_lev, new_slp

    def keep_state(_):
        return lev_prev, slp_prev

    # update
    new_lev, new_slp = lax.cond(
        jnp.isfinite(y_t),
        update_state,
        keep_state,
        operand=None
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
    _, res = lax.scan(
        lambda carry, inputs: dlt_transition_step(carry, inputs, lev_sm, slp_sm, theta), 
        (y[0] - reg_comp[0], 0.), y - reg_comp
    )
    _, _, dlt_comp = res
    # mid point estimation
    mu = dlt_comp + reg_comp

    numpyro.deterministic("mu", mu)
    numpyro.deterministic("dlt_comp", dlt_comp)
    numpyro.deterministic("reg_comp", reg_comp)

    # likelihood
    numpyro.sample("observations", dist.Normal(loc=mu, scale=sigma), obs=y, obs_mask=jnp.isfinite(y))


def run_dlt_model(
    lev_sm, 
    slp_sm, 
    theta, 
    y, 
    mcmc_run_args: Dict[str, any],
    regression_scheme: xr.Dataset,
    seed: Optional[int] = None,
):
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
    # generate seed based on current time stamp
    if seed is None:
        seed = generate_seed()

    # extract regressors matrix, coef loc and scale
    var_name = regression_scheme['var_name'].to_numpy()
    covariates = regression_scheme['covariates'].transpose("time", "var_name").to_numpy()
    coef_loc = regression_scheme['coef_loc'].to_numpy()
    coef_scale = regression_scheme['coef_scale'].to_numpy()

    logger.info(f"var_name: {var_name}")
    logger.info(f"covariates shape: {covariates.shape}")
    logger.debug(f"coef_loc: {coef_loc}")
    logger.debug(f"coef_scale: {coef_scale}")

    init_strategy = init_to_median(num_samples=10)
    kernel = NUTS(dlt_model, init_strategy=init_strategy)
    mcmc = MCMC(kernel, **mcmc_run_args)
    rng_key = jax.random.PRNGKey(seed)
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


def generate_in_sample_forecast(posteriors: xr.Dataset, transform_callback=np.exp, q=0.05):
    mu_samples = flatten_front_dim(posteriors["mu"].to_numpy(), n=2)
    sigma_samples = flatten_front_dim(posteriors["sigma"].to_numpy(), n=2)
    eps_samples = np.transpose(
        np.random.normal(loc=0.0, scale=sigma_samples, size=(mu_samples.shape[-1], sigma_samples.shape[0])),
        axes=(1, 0)
    )
    yhat_samples = transform_callback(mu_samples + eps_samples)
    yhat_lower, yhat_mid, yhat_upper = np.quantile(yhat_samples, q=[q, 0.5, 1 - q], axis=0)
    return yhat_lower, yhat_mid, yhat_upper