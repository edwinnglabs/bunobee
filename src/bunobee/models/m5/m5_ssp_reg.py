"""M5 SSP models with a static regression, fitted with NUTS.

The model of a series, on the scale of its target transform, is::

    y_t = level_t + dow_t · alpha_t + x_t · beta + eps_t,    beta_k ~ N(0, tau^2)

The level and the weekday offsets are the states of the Kalman filter (a level that drifts, six weekday states that
drift together), as in :mod:`bunobee.models.m5.m5_ssp_optim`. The regression is not: ``x_t`` is a row of a design
matrix ``X`` shared by every series, and ``beta`` is a parameter of the series with no time dimension. It is sampled
beside the sigmas, and the filter sees the series less ``X beta``. The filter then has the same few states however many
columns ``X`` has, where a covariate with a fixed coefficient costs the filter a state each.

The series are independent: each has its own sigmas and its own ``beta``, and a numpyro plate over the series only lets
NUTS fit a batch of them in one run. The forecast averages the back-transformed forecast over posterior draws, so it is
a posterior mean of the sales, where the MAP fitters give the forecast of one point.

The target transform is fitted once per series on its fit days and held fixed, as in the MAP fitters. The filter
needs no ``optax``, but the sigma priors come from :mod:`bunobee.models.m5.m5_ssp_optim`, so this module needs the
optional ``m5`` extra too: ``pip install 'bunobee[m5]'``.
"""

from __future__ import annotations

import logging
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import distributions as dist
from numpyro.diagnostics import effective_sample_size, gelman_rubin
from numpyro.infer import MCMC, NUTS

from .m5_ssp_optim import SIGMA_H_PRIOR, SIGMA_Q_PRIOR, SigmaPrior, kalman_filter_1d_batch
from .transforms import LogTransform, Transform

logger = logging.getLogger(__name__)

#: Sites of the model that hold one value per series, or one row of values per series.
SIGMA_SITES = ("sigma_h", "sigma_q_level", "sigma_q_dow")


def ssp_reg_model(
    y: jnp.ndarray,
    Z: jnp.ndarray,
    X: jnp.ndarray,
    tau: float,
    sigma_h_prior: SigmaPrior,
    sigma_q_prior: SigmaPrior,
) -> None:
    """The numpyro model of a batch of independent series.

    Parameters
    ----------
    y : jnp.ndarray
        Transformed sales, ``(n_series, n_steps)``.
    Z : jnp.ndarray
        Design of the filter's states, ``(n_steps, n_states)``: the level column, then the weekday columns. The first
        state drifts with ``sigma_q_level``, the others share ``sigma_q_dow``.
    X : jnp.ndarray
        Regression design, ``(n_steps, n_covariates)``, shared by every series. May have no column.
    tau : float
        Prior scale of each coefficient of ``beta``, which is ``N(0, tau^2)``.
    sigma_h_prior, sigma_q_prior : SigmaPrior
        Priors of the observation noise and of each of the two drifts.
    """
    n_series, n_states = y.shape[0], Z.shape[1]
    with numpyro.plate("series", n_series):
        sigma_h = numpyro.sample("sigma_h", sigma_h_prior.distribution)
        sigma_q_level = numpyro.sample("sigma_q_level", sigma_q_prior.distribution)
        sigma_q_dow = numpyro.sample("sigma_q_dow", sigma_q_prior.distribution)
        if X.shape[1]:
            beta = numpyro.sample("beta", dist.Normal(0.0, tau).expand([X.shape[1]]).to_event(1))  # (n_series, K)
            y = y - beta @ X.T
    sigma_q = jnp.column_stack([sigma_q_level, jnp.broadcast_to(sigma_q_dow[:, None], (n_series, n_states - 1))])
    log_p, *_ = kalman_filter_1d_batch(
        a0=jnp.zeros(n_states),
        P0=jnp.ones(n_states),
        Z=Z,
        sigma_h=sigma_h,
        sigma_q=sigma_q,
        y=y,
        logp=True,
        chunk_size=None,
    )
    numpyro.factor("log_lik", log_p.sum())


def _per_series_diagnostics(grouped: dict[str, np.ndarray], n_series: int) -> tuple[np.ndarray, np.ndarray]:
    """Worst R-hat and smallest effective sample size of each series over its parameters.

    Parameters
    ----------
    grouped : dict of str to np.ndarray
        Posterior samples by site, each ``(n_chains, n_draws, n_series, ...)``.
    n_series : int
        Number of series.

    Returns
    -------
    r_hat_max, n_eff_min : np.ndarray
        Shape ``(n_series,)`` each. R-hat is NaN with a single chain, which cannot estimate it.
    """
    r_hat = np.full(n_series, -np.inf)
    n_eff = np.full(n_series, np.inf)
    for samples in grouped.values():
        if samples.shape[0] > 1:
            r_hat = np.maximum(r_hat, np.asarray(gelman_rubin(samples)).reshape(n_series, -1).max(axis=1))
        n_eff = np.minimum(n_eff, np.asarray(effective_sample_size(samples)).reshape(n_series, -1).min(axis=1))
    return np.where(np.isfinite(r_hat), r_hat, np.nan), n_eff


def fit_batch_series_reg(
    sales_matrix: np.ndarray,
    Z: np.ndarray,
    X: np.ndarray | None = None,
    tau: float = 1.0,
    transform: Transform = LogTransform(),
    sigma_h_prior: SigmaPrior = SIGMA_H_PRIOR,
    sigma_q_prior: SigmaPrior = SIGMA_Q_PRIOR,
    num_warmup: int = 200,
    num_samples: int = 200,
    num_chains: int = 4,
    chain_method: Literal["parallel", "sequential", "vectorized"] = "vectorized",
    seed: int = 0,
) -> dict:
    """Fit B independent series with a static regression by NUTS, in one run.

    Each series is fitted to ``transform.forward`` of its sales, with the transform's parameters fitted on its sales
    first and then held fixed. All series share one NUTS run, so they share its step size: a batch of series that
    differ a lot in scale adapts to the hardest of them, and a smaller batch fits each series more closely.

    Parameters
    ----------
    sales_matrix : np.ndarray
        Daily unit sales, ``(B, n_steps)``.
    Z : np.ndarray
        Design of the filter's states, ``(n_steps, n_states)`` with at least two states: the level column leads, the
        other columns share a drift.
    X : np.ndarray | None
        Regression design, ``(n_steps, n_covariates)``, shared by every series. ``None``, the default, has none.
    tau : float
        Prior scale of each coefficient, ``N(0, tau^2)``, by default 1. The target is standardised by the power
        transforms, so 1 lets one coefficient move as much as the whole series does.
    transform : Transform
        Target transform, by default :class:`LogTransform`.
    sigma_h_prior, sigma_q_prior : SigmaPrior
        Priors of the observation noise and of each of the two drifts, by default the uniform ones of the MAP
        fitters.
    num_warmup, num_samples, num_chains : int
        NUTS warmup and posterior draws per chain, and the number of chains, by default 200, 200 and 4.
    chain_method : {"parallel", "sequential", "vectorized"}
        How numpyro runs the chains, by default ``"vectorized"``, which needs no extra CPU devices. ``"parallel"`` needs
        ``numpyro.set_host_device_count(num_chains)`` before JAX starts.
    seed : int
        PRNG seed, by default 0.

    Returns
    -------
    dict
        Keys: ``samples`` (dict of site to array ``(num_chains * num_samples, B, ...)``, chains stacked: the sites
        are ``sigma_h``, ``sigma_q_level``, ``sigma_q_dow`` and, with ``X``, ``beta`` ``(draws, B, K)``), ``y``
        (the transformed sales ``(B, n_steps)``), ``transform_params`` (the transform's parameters, ``(B,)`` each),
        ``transform``, ``Z``, ``X`` (the regression design as fitted, ``(n_steps, K)`` with ``K`` possibly 0),
        ``r_hat_max`` and ``n_eff_min`` (``(B,)`` each, over the series' parameters) and ``n_divergences`` (int, of
        the whole run).

    Raises
    ------
    ValueError
        If ``Z`` has fewer than two columns or ``X`` and ``Z`` have different numbers of rows.
    """
    Z = jnp.asarray(Z, dtype=jnp.float32)
    X = jnp.zeros((Z.shape[0], 0), dtype=jnp.float32) if X is None else jnp.asarray(X, dtype=jnp.float32)
    if Z.shape[1] < 2:
        raise ValueError(f"Z needs a level column and at least one more state, got {Z.shape[1]} column(s)")
    if X.shape[0] != Z.shape[0] or sales_matrix.shape[1] != Z.shape[0]:
        raise ValueError(f"sales ({sales_matrix.shape[1]}), Z ({Z.shape[0]}) and X ({X.shape[0]}) differ in steps")

    transform_params = transform.fit(sales_matrix)
    y = jnp.asarray(transform.forward(sales_matrix, transform_params))

    mcmc = MCMC(
        NUTS(ssp_reg_model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), y, Z, X, tau, sigma_h_prior, sigma_q_prior, extra_fields=("diverging",))
    n_divergences = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    if n_divergences:
        logger.warning("NUTS had %d divergent transitions over %d series", n_divergences, y.shape[0])

    grouped = {site: np.asarray(v) for site, v in mcmc.get_samples(group_by_chain=True).items()}
    r_hat_max, n_eff_min = _per_series_diagnostics(grouped, y.shape[0])
    return {
        "samples": {site: v.reshape(-1, *v.shape[2:]) for site, v in grouped.items()},
        "y": np.asarray(y),
        "transform_params": transform_params,
        "transform": transform,
        "Z": Z,
        "X": np.asarray(X),
        "r_hat_max": r_hat_max,
        "n_eff_min": n_eff_min,
        "n_divergences": n_divergences,
    }


def predict_batch_series_reg(
    fit_result: dict,
    Z_future: np.ndarray,
    X_future: np.ndarray | None = None,
    n_draws: int = 100,
    block: int = 2048,
) -> np.ndarray:
    """Posterior-mean point forecasts for B series from :func:`fit_batch_series_reg`.

    For each of ``n_draws`` posterior draws, evenly spread over the run, the filter runs over the series less
    ``X beta`` of that draw. Its last filtered state and ``beta`` give the forecast mean on the transformed scale,
    ``Z_future a_T + X_future beta``, and the last filtered variance and ``sigma_h`` its variance, ``Z_future^2 P_T +
    sigma_h^2``. ``transform.inverse`` of the two is the draw's forecast in units, as in
    :func:`~bunobee.models.m5.m5_ssp_optim.predict_batch_series_opt`. The result is the mean over the draws, so the
    uncertainty of ``beta`` and the sigmas enters through the spread of the draws.

    Parameters
    ----------
    fit_result : dict
        Output of :func:`fit_batch_series_reg`.
    Z_future : np.ndarray
        Filter design over the forecast days, ``(horizon, n_states)``.
    X_future : np.ndarray | None
        Regression design over the forecast days, ``(horizon, n_covariates)``. Required if the fit had ``X``.
    n_draws : int
        Posterior draws to average, by default 100. At most the number of draws of the run.
    block : int
        Most (draw, series) pairs filtered in one call, by default 2048, to bound the memory of the filtered states.

    Returns
    -------
    np.ndarray
        Point forecasts, ``(B, horizon)``.

    Raises
    ------
    ValueError
        If the fit had a regression and ``X_future`` does not have its number of columns.
    """
    samples, y, Z = fit_result["samples"], fit_result["y"], fit_result["Z"]
    n_series, n_steps = y.shape
    n_states = Z.shape[1]
    n_covariates = samples["beta"].shape[-1] if "beta" in samples else 0
    Z_future = np.asarray(Z_future, dtype=np.float32)
    if n_covariates and (X_future is None or np.shape(X_future)[1] != n_covariates):
        raise ValueError(f"the fit has {n_covariates} covariates, X_future must have that many columns")
    X_future = None if not n_covariates else np.asarray(X_future, dtype=np.float32)

    n_total = samples["sigma_h"].shape[0]
    draws = np.unique(np.linspace(0, n_total - 1, min(n_draws, n_total)).astype(int))
    per_block = max(1, block // n_series)
    total = np.zeros((n_series, Z_future.shape[0]))
    for start in range(0, draws.size, per_block):
        chosen = draws[start : start + per_block]
        n = chosen.size
        flat = lambda site: np.asarray(samples[site])[chosen].reshape(n * n_series)  # noqa: E731 (draw-major)
        sigma_h = flat("sigma_h")
        sigma_q = np.column_stack(
            [flat("sigma_q_level"), np.repeat(flat("sigma_q_dow")[:, None], n_states - 1, axis=1)]
        )
        resid = np.broadcast_to(y, (n, n_series, n_steps))
        if n_covariates:
            beta = np.asarray(samples["beta"])[chosen]  # (n, B, K)
            resid = resid - np.einsum("nbk,tk->nbt", beta, np.asarray(fit_result["X"]))
        _, at, Pt, *_ = kalman_filter_1d_batch(
            a0=jnp.zeros(n_states),
            P0=jnp.ones(n_states),
            Z=Z,
            sigma_h=jnp.asarray(sigma_h),
            sigma_q=jnp.asarray(sigma_q),
            y=jnp.asarray(resid.reshape(n * n_series, n_steps)),
            logp=False,
            chunk_size=None,
        )
        mu = np.asarray(at[:, -1]) @ Z_future.T
        if n_covariates:
            mu = mu + np.einsum("nbk,hk->nbh", beta, X_future).reshape(n * n_series, -1)
        var = np.asarray(Pt[:, -1]) @ (Z_future**2).T + (sigma_h**2)[:, None]
        params = {key: np.tile(value, n) for key, value in fit_result["transform_params"].items()}
        forecast = fit_result["transform"].inverse(mu, var, params)
        total += np.asarray(forecast).reshape(n, n_series, -1).sum(axis=0)
    return total / draws.size
