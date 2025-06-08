
import jax.numpy as jnp
import numpy as np
from jax import lax
import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median
import xarray as xr
from typing import Optional

from ..utils import generate_seed

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

    # update
    new_lev = lev_sm * y_t + (1 - lev_sm) * (lev_prev + theta * slp_prev)
    new_slp = slp_sm * (new_lev - lev_prev) + (1 - slp_sm) * slp_prev

    return (new_lev, new_slp), (new_lev, new_slp, dlt_comp_t)



def dlt_model(lev_sm, slp_sm, theta, x_seas, x_glb_trend, y):
    """Damped Local Trend (DLT) model for time series forecasting.
    Args
    ----
    lev_sm: Level smoothing factor (scalar).
    slp_sm: Slope smoothing factor (scalar).
    theta: Damping factor (scalar).
    x_seas: Seasonal features (2D array with shape (n_steps, n_seasons))
    x_glb_trend: Global trend feature (1D array with shape (n_steps,)).
    y: Observations (1D array with shape (n_steps,))
    """

    sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))
    alpha_glb_trend = numpyro.sample("alpha_glb_trend", dist.Normal(0, 1.0))
    beta_glb_trend = numpyro.sample("beta_glb_trend", dist.Normal(0, 1.0))
    beta_seas = numpyro.sample("beta_seas", dist.Normal(0, 0.3).expand([x_seas.shape[1]]))

    # (n_steps, )
    seas = jnp.sum(x_seas * beta_seas, axis=-1)
    # (n_steps, )
    glb_trend = alpha_glb_trend + x_glb_trend * beta_glb_trend
    reg_comp = seas + glb_trend

    # scan with the partial function
    _, res = lax.scan(
        lambda carry, inputs: dlt_transition_step(carry, inputs, lev_sm, slp_sm, theta), 
        (y[0] - reg_comp[0], 0), y - reg_comp
    )
    _, _, dlt_comp = res
    # mid point estimation
    mu = dlt_comp + reg_comp

    numpyro.deterministic("mu", mu)
    numpyro.deterministic("dlt_comp", dlt_comp)
    numpyro.deterministic("reg_comp", reg_comp)

    # likelihood
    numpyro.sample("observations", dist.Normal(loc=mu, scale=sigma), obs=y)


def run_dlt_model(
    lev_sm, 
    slp_sm, 
    theta, 
    x_seas, 
    x_glb_trend, 
    y, 
    seed: Optional[int] = None
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

    init_strategy = init_to_median(num_samples=10)
    kernel = NUTS(dlt_model, init_strategy=init_strategy)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=4)
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(
        rng_key, 
        lev_sm=lev_sm, 
        slp_sm=slp_sm, 
        theta=theta,
        x_seas=x_seas,
        x_glb_trend=x_glb_trend,
        y=y
    )
    
    posteriors_dict = mcmc.get_samples()

    # transform them into xr.Dataset
    n_samples = posteriors_dict['alpha_glb_trend'].shape[0]
    n_steps = posteriors_dict['dlt_comp'].shape[1]
    n_seas = posteriors_dict['beta_seas'].shape[1]

    # Build Dataset
    posteriors = xr.Dataset(
        {
            'alpha_glb_trend': (['sample'], posteriors_dict['alpha_glb_trend']),
            'beta_glb_trend': (['sample'], posteriors_dict['beta_glb_trend']),
            'beta_seas': (['sample', 'season'], posteriors_dict['beta_seas']),
            'dlt_comp': (['sample', 'time'], posteriors_dict['dlt_comp']),
            'mu': (['sample', 'time'], posteriors_dict['mu']),
            'reg_comp': (['sample', 'time'], posteriors_dict['reg_comp']),
            'sigma': (['sample'], posteriors_dict['sigma']),
        },
        coords={
            'sample': np.arange(n_samples),
            'time': np.arange(n_steps),
            'season': np.arange(n_seas),
        }
    )

    return posteriors

def generate_in_sample_forecast(posteriors_dict, transform_callback=np.exp, q=0.05):
    mu_samples = np.array(posteriors_dict["mu"])
    sigma_samples = np.array(posteriors_dict["sigma"])
    eps_samples = np.transpose(
        np.random.normal(loc=0.0, scale=sigma_samples, size=(mu_samples.shape[-1], sigma_samples.shape[0])),
        axes=(1, 0)
    )
    yhat_samples = transform_callback(mu_samples + eps_samples)
    yhat_lower, yhat_mid, yhat_upper = np.quantile(yhat_samples, q=[q, 0.5, 1 - q], axis=0)
    return yhat_lower, yhat_mid, yhat_upper