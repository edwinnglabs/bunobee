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
    """Single transition step for the Damped Local Trend (DLT) model, compatible with `lax.scan`.

    Computes the one-step-ahead forecast (`dlt_comp_t = lev_prev + theta * slp_prev`),
    then updates the level and slope via exponential smoothing. NaN observations are
    skipped — the state is carried forward unchanged.

    Args
    ----
    carry: Tuple ``(lev_prev, slp_prev)`` — the level and slope from the previous step.
    inputs: In-sample mode: the scalar observation ``y_t``.
            Out-of-sample mode (``oos=True``): a noise draw ``eps_t`` added to the forecast.
    lev_sm: Level smoothing factor in ``[0, 1]``.
    slp_sm: Slope smoothing factor in ``[0, 1]``.
    theta: Damping factor applied to the slope each step.
    oos: If ``True``, treats ``inputs`` as noise rather than observed values.

    Returns
    -------
    carry: Updated ``(new_lev, new_slp)`` tuple for the next step.
    outputs: Tuple ``(new_lev, new_slp, dlt_comp_t)`` stacked by ``lax.scan``.

    Examples
    --------
    >>> from jax import lax
    >>> from wunkui.models.dlt import dlt_transition_step
    >>> _, (levs, slps, dlt_comp) = lax.scan(
    ...     lambda carry, y_t: dlt_transition_step(carry, y_t, lev_sm, slp_sm, theta),
    ...     (y[0], 0.0), y
    ... )
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
    """NumPyro probabilistic model for Damped Local Trend (DLT) time series.

    Samples ``sigma`` from a ``HalfNormal`` prior, optionally adds a regression
    component (neutral / negative-constrained / positive-constrained coefficients),
    runs the DLT recursion via ``lax.scan``, and registers a Normal likelihood.

    Deterministic sites registered: ``yhat``, ``dlt_comp``, ``reg_comp``,
    ``last_lev``, ``last_slp``, and ``coef`` (when regression is used).

    Args
    ----
    lev_sm: Level smoothing factor (scalar).
    slp_sm: Slope smoothing factor (scalar).
    theta: Damping factor (scalar).
    y: Observations, shape ``(n_steps,)``. NaN values are masked in the likelihood.
    reg_input: Optional dict keyed by sign (``"="``, ``"+"``, ``"-"``), each mapping
        to a sub-dict with ``"covariates"`` (array), ``"coef_loc"``, and
        ``"coef_scale"`` for the coefficient prior.
    """
    y = jnp.asarray(y, dtype=y.dtype)
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
    """Run NUTS/MCMC on the DLT model and return posterior samples as an ``xr.Dataset``.

    Args
    ----
    rng_key: JAX random key for reproducibility.
    lev_sm: Level smoothing factor.
    slp_sm: Slope smoothing factor.
    theta: Damping factor.
    y: Observations, shape ``(n_steps,)``.
    mcmc_run_args: Keyword arguments forwarded to ``numpyro.infer.MCMC`` (e.g.
        ``num_warmup``, ``num_samples``, ``num_chains``).
    regression_scheme: Optional ``RegressionScheme`` defining regressor signs and
        coefficient priors.
    covariates_df: Optional DataFrame of covariates, shape ``(n_steps, n_var)``.
        Required when ``regression_scheme`` is provided.

    Returns
    -------
    xr.Dataset with dimensions ``(chain, draw, time)`` containing:
        ``dlt_comp``, ``dlt_comp_p50``, ``yhat``, ``yhat_p50``, ``resid``,
        ``resid_p50``, ``reg_comp``, ``sigma``, ``last_lev``, ``last_slp``,
        and (if regression) ``coef``, ``coef_p50``, ``reg_comp_p50``.
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


def generate_dlt_components(
    rng_key: jnp.ndarray,
    posteriors: xr.Dataset,
    end_step: int,
    lev_sm: float,
    slp_sm: float,
    theta: float,
) -> jnp.ndarray:
    """Generate DLT component samples for in-sample and (optionally) out-of-sample steps.

    For steps already covered by the fitted model (``step < n_train_steps``), the
    in-sample ``dlt_comp`` draws are returned directly.  For additional steps
    (``end_step > n_train_steps``), the DLT recursion is continued from the last
    posterior level/slope using sampled noise.

    Args
    ----
    rng_key: JAX random key used to draw noise for out-of-sample steps.
    posteriors: ``xr.Dataset`` returned by ``run_dlt_model``, must contain
        ``dlt_comp``, ``sigma``, ``last_lev``, and ``last_slp``.
    end_step: Total number of time steps to generate (in-sample + out-of-sample).
    lev_sm: Level smoothing factor.
    slp_sm: Slope smoothing factor.
    theta: Damping factor.

    Returns
    -------
    jnp.ndarray of shape ``(n_samples, end_step)`` containing the DLT component
    draws across all requested time steps.
    """
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


def make_inference(
    rng_key: jnp.ndarray,
    posteriors: xr.Dataset,
    lev_sm: float,
    slp_sm: float,
    theta: float,
    end_step: Optional[int] = None,
    covariates_df: Optional[pd.DataFrame] = None,
    transform_callback: Optional[Callable] = None,
) -> xr.Dataset:
    """Generate full forecast samples by combining DLT and regression components.

    Calls ``generate_dlt_comp_samples`` for the trend, adds the regression component
    (when covariates are provided), and optionally applies a back-transform.

    When ``covariates_df`` is supplied and ``"coef"`` is present in ``posteriors``,
    ``end_step`` is inferred from the length of the covariates and must not be set
    manually.  Without covariates, ``end_step`` is required.

    Args
    ----
    rng_key: JAX random key for reproducibility.
    posteriors: ``xr.Dataset`` returned by ``run_dlt_model``.
    lev_sm: Level smoothing factor.
    slp_sm: Slope smoothing factor.
    theta: Damping factor.
    end_step: Total steps to forecast. Ignored (overridden) when covariates are given;
        required otherwise.
    covariates_df: Optional DataFrame of covariates, shape ``(n_steps, n_var)``.
        Column names must match the ``var_name`` coordinate in ``posteriors``.
    transform_callback: Optional callable applied element-wise to ``forecast_samples``
        (e.g. to reverse a log transform).

    Returns
    -------
    xr.Dataset with dimensions ``(sample, time)`` containing:
        ``dlt_comp_samples``, ``forecast_samples``, and (if regression)
        ``reg_comp_samples`` with an additional ``var_name`` dimension.
    """
    if "coef" in posteriors and covariates_df is not None:
        # (n_samples, n_var)
        coef = flatten_front_dim(posteriors["coef"].to_numpy(), n=2)
        var_names = list(posteriors["coef"].var_name.values)
        covariates = covariates_df[var_names].values
        logger.debug(f"var_names: {var_names}")
        logger.debug(f"coef shape: {coef.shape}")
        logger.debug(f"covariates shape: {covariates.shape}")
        # Keep individual variable contributions: (n_samples, n_var) * (n_steps, n_var) -> (n_samples, n_steps, n_var)
        # coef[:, None, :] is (n_samples, 1, n_var), covariates[None, :, :] is (1, n_steps, n_var)
        reg_comp_samples_per_var = coef[:, None, :] * covariates[None, :, :]
        logger.debug(f"reg_comp_samples_per_var shape: {reg_comp_samples_per_var.shape}")
        # Sum over variables to get total reg component: (n_samples, n_steps)
        reg_comp_samples_total = np.sum(reg_comp_samples_per_var, axis=-1)
        logger.debug(f"reg_comp_samples_total shape: {reg_comp_samples_total.shape}")
        end_step = reg_comp_samples_total.shape[-1]
        logger.info(f"Overriding end_step to {end_step} based on regression components.")
        has_regression = True
    else:
        if end_step is None:
            raise ValueError("end_step must be provided if covariates are not provided.")
        logger.debug(f"end_step: {end_step}")
        reg_comp_samples_total = 0
        has_regression = False

    dlt_comp_samples = generate_dlt_components(
        rng_key,
        posteriors,
        end_step,
        lev_sm,
        slp_sm,
        theta
    )

    logger.debug(f"dlt_comp_samples shape: {dlt_comp_samples.shape}")

    forecast_samples = dlt_comp_samples + reg_comp_samples_total
    if transform_callback is not None:
        forecast_samples = transform_callback(forecast_samples)

    logger.debug(f"forecast_samples shape: {forecast_samples.shape}")

    # Build xr.Dataset
    n_samples, n_steps = dlt_comp_samples.shape

    data_vars = {
        'dlt_comp_samples': (['sample', 'time'], np.asarray(dlt_comp_samples)),
        'forecast_samples': (['sample', 'time'], np.asarray(forecast_samples)),
    }

    coords = {
        'sample': np.arange(n_samples),
        'time': np.arange(n_steps),
    }

    if has_regression:
        # Add reg_comp_samples with var_name dimension
        data_vars['reg_comp_samples'] = (['sample', 'time', 'var_name'], reg_comp_samples_per_var)
        coords['var_name'] = var_names

    result = xr.Dataset(data_vars=data_vars, coords=coords)

    return result

