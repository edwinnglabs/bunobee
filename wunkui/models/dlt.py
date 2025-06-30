import logging
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax import lax
import jax
import pandas as pd
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median
import xarray as xr
from typing import Optional, Dict, Callable

from ..utils import flatten_front_dim
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


def dlt_model(
    lev_sm: float, 
    slp_sm: float, 
    theta: float, 
    y: jnp.ndarray, 
    reg_input: Optional[Dict[str, any]] = None,  
):
    """Damped Local Trend (DLT) model for time series forecasting.
    
    Args
    ----
    lev_sm: Level smoothing factor (scalar).
    slp_sm: Slope smoothing factor (scalar).
    theta: Damping factor (scalar).
    y: Observations (1D array with shape (n_steps,)).
    reg_input: Optional; a dictionary containing regression inputs.
    
    y: Observations (1D array with shape (n_steps,))
    """
    y = jnp.assay(y, dtype=y.dtype)
    sigma = numpyro.sample(
        "sigma",
        dist.HalfNormal(5.0)
    )

    if reg_input is not None:
        coefs = []
        neu_reg_input = reg_input.get("=", None)
        if neu_reg_input is not None:
            neu_coef_loc_prior = neu_reg_input["coef_loc"]
            neu_coef_scale_prior = neu_reg_input["coef_scale"]
            neu_covariates = neu_reg_input["covariates"]
            neu_coef = numpyro.sample("neu_coef", 
                dist.Normal(
                loc=neu_coef_loc_prior, 
                scale=neu_coef_scale_prior
            ))
            coefs.append(neu_coef)
            neu_reg_comp = jnp.sum(neu_covariates * neu_coef, axis=-1)
        else:
            neu_reg_comp = 0.
        neg_reg_input = reg_input.get("-", None)
        if neg_reg_input is not None:
            neg_coef_loc_prior = neg_reg_input["coef_loc"]
            neg_coef_scale_prior = neg_reg_input["coef_scale"]
            neg_covariates = neg_reg_input["covariates"]
            neg_coef = numpyro.sample("neg_coef", 
                dist.TruncatedNormal(
                loc=neg_coef_loc_prior, 
                scale=neg_coef_scale_prior,
                high=0.0,
            ))
            coefs.append(neg_coef)
            neg_reg_comp = jnp.sum(neg_covariates * neg_coef, axis=-1)
        else:
            neg_reg_comp = 0.
        pos_reg_input = reg_input.get("+", None)
        if pos_reg_input is not None:
            pos_coef_loc_prior = pos_reg_input["coef_loc"]
            pos_coef_scale_prior = pos_reg_input["coef_scale"]
            pos_covariates = pos_reg_input["covariates"]
            pos_coef = numpyro.sample("pos_coef",
                dist.TruncatedNormal(
                loc=pos_coef_loc_prior, 
                scale=pos_coef_scale_prior,
                low=0.0,
            ))
            coefs.append(pos_coef)
            pos_reg_comp = jnp.sum(pos_covariates * pos_coef, axis=-1)
        else:
            pos_reg_comp = 0.
        # (n_steps, )
        reg_comp = neu_reg_comp + neg_reg_comp + pos_reg_comp
        coef = jnp.concatenate(coefs, axis=-1)
        numpyro.deterministic("coef", coef)
    else:
        reg_comp = 0.

    # scan with the partial function
    final_states, all_states = lax.scan(
        lambda carry, inputs: dlt_transition_step(carry, inputs, lev_sm, slp_sm, theta), 
        (y[0] - reg_comp[0], 0.), y - reg_comp
    )
    _, _, dlt_comp = all_states
    last_lev, last_slp = final_states
    # mid point estimation
    yhat = dlt_comp + reg_comp

    numpyro.deterministic("yhat", yhat)
    # use for in-sample
    numpyro.deterministic("dlt_comp", dlt_comp)
    numpyro.deterministic("reg_comp", reg_comp)
    # use for out-of-sample
    numpyro.deterministic("last_lev", last_lev)
    numpyro.deterministic("last_slp", last_slp)

    # likelihood
    numpyro.sample("observations", dist.Normal(loc=yhat, scale=sigma), obs=y, obs_mask=jnp.isfinite(y))


def run_dlt_model(
    rng_key: jnp.ndarray,
    lev_sm: float, 
    slp_sm: float, 
    theta: float, 
    y: np.ndarray, 
    mcmc_run_args: Dict[str, any],
    regression_scheme: Optional[RegressionScheme] = None,
    covariates_df: Optional[pd.DataFrame] = None,
) -> xr.Dataset:
    """Run the DLT model with the provided parameters and data.

    Args
    ----
    rng_key: JAX random key for reproducibility.
    lev_sm: Level smoothing factor.
    slp_sm: Slope smoothing factor.
    theta: Damping factor.
    y: Observations (1D array with shape (n_steps,)).
    mcmc_run_args: Dictionary containing the arguments for MCMC run.
    regression_scheme: Optional; a RegressionScheme object containing the regression scheme.
    covariates_df: Optional; a pandas DataFrame containing the covariates with shape (n_steps, n_var).


    Returns 
    -------
    posteriors_dict: Dictionary containing the posterior samples of the model parameters.
    """
    if regression_scheme is not None and covariates_df is not None:
        # extract regressors matrix, coef loc and scale
        # separate three groups of regressors with the sign: "=", "+", "-"
        reg_input = {}
        var_name = []
        for sign in ["=", "+", "-"]:
            regressors = regression_scheme.scheme.loc[regression_scheme.scheme['sign'] == sign, :].index.to_list()
            if len(regressors) > 0:
                coef_loc = regression_scheme.scheme.loc[regressors, 'loc_prior'].values
                coef_loc = jnp.asarray(coef_loc, dtype=jnp.float32)
                coef_scale = regression_scheme.scheme.loc[regressors, 'scale_prior'].values
                coef_scale = jnp.asarray(coef_scale, dtype=jnp.float32)
                covariates = covariates_df.loc[:, regressors].values if len(regressors) > 0 else None
                reg_input[sign] = {
                    "coef_loc": coef_loc,
                    "coef_scale": coef_scale,
                    "covariates": covariates,
                }
                var_name += regressors

                logger.debug(f"sign: {sign}")
                logger.debug(f"regressors: {regressors}")
                logger.debug(f"covariates shape: {covariates.shape}")
                logger.debug(f"coef_loc: {coef_loc}")
                logger.debug(f"coef_scale: {coef_scale}")
    else:
        covariates = None

    init_strategy = init_to_median(num_samples=10)
    kernel = NUTS(dlt_model, init_strategy=init_strategy)
    mcmc = MCMC(kernel, **mcmc_run_args)

    mcmc.run(
        rng_key, 
        lev_sm=lev_sm, 
        slp_sm=slp_sm, 
        theta=theta,
        y=y,
        reg_input=reg_input
    )

    posteriors_dict = mcmc.get_samples(group_by_chain=True)

    # transform them into xr.Dataset
    n_chains, n_draws, n_steps = posteriors_dict['dlt_comp'].shape

    dlt_comp_p50 = jnp.median(posteriors_dict['dlt_comp'], axis=(0, 1))

    data_vars = {
        'dlt_comp': (['chain', 'draw', 'time'], posteriors_dict['dlt_comp']),
        'dlt_comp_p50': (['time'], dlt_comp_p50),  # median of in-sample forecast
        # in-sample prediction before reverse transform with original covariates
        'yhat': (['chain', 'draw', 'time'], posteriors_dict['yhat']),
        "resid": (['chain', 'draw', 'time'], y - posteriors_dict['yhat']),
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
        coef_p50 = jnp.median(posteriors_dict['coef'], axis=(0, 1))
        # (n_steps, )
        reg_comp_p50 = np.sum(coef_p50 * covariates_df[var_name].values, axis=-1)
        # (n_steps, )
        yhat_p50 = dlt_comp_p50 + reg_comp_p50
        data_vars.update({
            'coef': (['chain', 'draw', 'var_name'], posteriors_dict['coef']),
            "coef_p50": (['var_name'], coef_p50),
            'reg_comp_p50': (['time'], reg_comp_p50),
            'yhat_p50': (['time'], yhat_p50),
        })
        coords.update({
            'var_name': var_name,
        })
    else:
        # (n_steps, )
        yhat_p50 = dlt_comp_p50
        data_vars.update({
            'yhat_p50': (['time'], yhat_p50),
        })

    data_vars.update({
        "resid_p50": (['time'], y - yhat_p50),
    })

    posteriors = xr.Dataset(data_vars=data_vars, coords=coords)

    return posteriors


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
        n_samples, n_train_steps = dlt_comp_is_samples.shape
        logger.debug(f"dlt_comp_is_samples shape: {dlt_comp_is_samples.shape}")
        logger.info(f"Collecting in-sample forecasts from step 0 to {n_train_steps}.")

    # (n_samples, )
    sigma_samples = flatten_front_dim(posteriors["sigma"].to_numpy(), n=2)
    last_lev = flatten_front_dim(posteriors["last_lev"].to_numpy(), n=2)
    last_slp = flatten_front_dim(posteriors["last_slp"].to_numpy(), n=2)


    # float64 to make sure it works with both 32 or 64 bit
    last_lev = jnp.asarray(last_lev, dtype=jnp.float64)
    last_slp = jnp.asarray(last_slp, dtype=jnp.float64)

    # log the shapes
    logger.debug(f"sigma_samples shape: {sigma_samples.shape}")
    logger.debug(f"last_lev shape: {last_lev.shape}")
    logger.debug(f"last_slp shape: {last_slp.shape}")

    # (n_steps, n_samples)
    eps_samples = jax.random.normal(rng_key, shape=(end_step, n_samples)) * sigma_samples

    if end_step > n_train_steps:
        logger.info(f"Generating out-of-sample forecasts from step {n_train_steps} to {end_step}.")
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
    else:
        # logger.debug("No out-of-sample forecast needed.")
        dlt_comp_samples = dlt_comp_is_samples

    logger.debug(f"dlt_comp_samples shape: {dlt_comp_samples.shape}")

    return dlt_comp_samples


def generate_forecast_samples(
    rng_key: jnp.ndarray,
    posteriors: xr.Dataset, 
    lev_sm: float,
    slp_sm: float,
    theta: float,
    end_step: Optional[int] = None, 
    covariates_df: Optional[pd.DataFrame] = None,
    transform_callback: Optional[Callable] = None,
) -> jnp.ndarray:
    """Generate forecast samples from the DLT model posteriors.

    Args
    ----
    rng_key: JAX random key for reproducibility.
    posteriors: xr.Dataset containing the posterior samples of the DLT model.
    end_step: The step at which to stop generating forecasts.
    lev_sm: Level smoothing factor.
    slp_sm: Slope smoothing factor.
    theta: Damping factor.
    covariates_df: Optional; a pandas DataFrame containing the covariates with shape (n_steps, n_var).
    transform_callback: Optional; a callable function to transform the forecast samples.

    Returns
    -------
    forecast_samples: (n_samples, n_steps) array of forecast samples.
    """
    if "coef" in posteriors and covariates_df is not None:
        # (n_samples, n_var)
        coef = flatten_front_dim(posteriors["coef"].to_numpy(), n=2) 
        var_names = posteriors["coef"].var_name.values
        covariates = covariates_df[var_names].values
        logger.debug(f"var_names: {var_names}")
        logger.debug(f"coef shape: {coef.shape}")
        logger.debug(f"covariates shape: {covariates.shape}")
        # (n_samples, n_var) * (n_steps, n_var) -> (n_samples, n_steps)
        reg_comp_samples = np.einsum("ik,jk->ij", coef, covariates)
        logger.debug(f"reg_comp_samples shape: {reg_comp_samples.shape}")
        end_step = reg_comp_samples.shape[-1]
        logger.info(f"Overriding end_step to {end_step} based on regression components.")
    else:
        if end_step is None:
            raise ValueError("end_step must be provided if covariates are not provided.")
        logger.debug(f"end_step: {end_step}")
        reg_comp_samples = 0

    dlt_comp_samples = generate_dlt_comp_samples(
        rng_key, 
        posteriors, 
        end_step, 
        lev_sm, 
        slp_sm, 
        theta
    )

    logger.debug(f"dlt_comp_samples shape: {dlt_comp_samples.shape}")

    forecast_samples = dlt_comp_samples + reg_comp_samples
    if transform_callback is not None:
        forecast_samples = transform_callback(forecast_samples)

    logger.debug(f"forecast_samples shape: {forecast_samples.shape}")
    return forecast_samples

