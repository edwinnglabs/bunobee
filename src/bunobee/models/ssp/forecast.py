from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import xarray as xr

from bunobee.models.ssp.posterior import a_to_lam
from bunobee.utils import flatten_front_dim

logger = logging.getLogger("bunobee")


def forecast_ssp(
    idata: xr.Dataset,
    Z_future: np.ndarray,
    *,
    exponent: float = 0.5,
    positivity: np.ndarray | None = None,
    noise_embed: bool = False,
    transform_callback: Callable | None = None,
    seed: int = 0,
) -> xr.Dataset:
    """Draw out-of-sample predictive samples from a fitted multi-series SSP posterior.

    This is a posterior-replay forecast: it needs no change to the filter. The
    filtered state path already lives in ``idata`` as the ``at`` site, so the
    forecast launches from the last filtered state ``a_T = at[:, -1, :]`` and
    propagates a driftless random walk forward ``horizon`` steps, injecting
    per-step process noise ``N(0, diag(sigma_q**2))`` independently per posterior
    sample::

        a_{T+h} = a_T + sum_{k=1..h} eta_k,        eta_k ~ N(0, diag(sigma_q**2))
        a_nat   = where(positivity, exp(exponent * a), a)
        mu      = einsum("hij,shj->shi", Z_future, a_nat)
        y       = mu + eps                          eps ~ N(0, sigma_h[series]**2)

    The ``eps`` term is added only when ``noise_embed=True``. The API mirrors
    ``bunobee.models.dlt.make_inference``.

    Parameters
    ----------
    idata : xr.Dataset
        Posterior carrying ``at`` with dims ``(chain, draw, n_steps, n_states)``,
        ``sigma_q`` with dims ``(chain, draw, n_states)``, and ``sigma_h`` with
        dims ``(chain, draw, n_series)``. When present, ``idata.attrs["exponent"]``
        overrides the ``exponent`` argument and a ``positivity`` data variable
        overrides the ``positivity`` argument.
    Z_future : np.ndarray, shape (horizon, n_series, n_states)
        Future per-series design matrix. ``Z_future[h, j]`` is the loading vector
        for series ``j`` at forecast step ``h``. See ``build_forecast_design`` for
        a helper that continues periodic dummies and future covariates.
    exponent : float, optional
        Exponent in the nonlinear state map ``exp(exponent * a)``. Default 0.5.
        Ignored when ``idata.attrs`` carries an ``"exponent"`` entry.
    positivity : np.ndarray or None, optional, shape (n_states,)
        Boolean mask; ``True`` selects states that use the nonlinear ``exp`` map.
        ``None`` (default) applies the nonlinear map to every state, matching the
        filter default. Pass ``np.zeros(n_states, dtype=bool)`` for a fully linear
        forecast. Ignored when ``idata`` carries a ``positivity`` data variable.
    noise_embed : bool, optional
        When ``True``, add a per-step observation-noise draw
        ``N(0, sigma_h[series]**2)`` to ``mu`` to form ``forecast_samples`` and
        return the draw as ``eps_samples``. Default ``False`` (return ``mu``).
    transform_callback : callable or None, optional
        Applied element-wise to ``forecast_samples`` only (e.g. to reverse a log
        transform). ``mu_samples`` and ``eps_samples`` are left untransformed.
    seed : int, optional
        Seed for ``numpy.random.default_rng``. A fixed seed makes the output
        bitwise-stable across runs. Default 0.

    Returns
    -------
    xr.Dataset
        Dims ``(sample, time, series)`` with integer-range coords. Contains
        ``forecast_samples`` and ``mu_samples`` always, plus ``eps_samples`` when
        ``noise_embed=True``. ``time`` has length ``horizon``. ``sample`` flattens
        the posterior ``(chain, draw)`` axes.
    """
    at = flatten_front_dim(np.asarray(idata["at"].to_numpy(), dtype=float), n=2)
    sigma_q = flatten_front_dim(np.asarray(idata["sigma_q"].to_numpy(), dtype=float), n=2)
    sigma_h = flatten_front_dim(np.asarray(idata["sigma_h"].to_numpy(), dtype=float), n=2)

    n_sample, _n_steps, n_states = at.shape
    n_series = sigma_h.shape[1]

    Z_future = np.asarray(Z_future, dtype=float)
    if Z_future.ndim != 3:
        raise ValueError(f"Z_future must be 3-D (horizon, n_series, n_states); got shape {Z_future.shape}")
    horizon, z_series, z_states = Z_future.shape
    if z_states != n_states:
        raise ValueError(f"Z_future last dim {z_states} does not match n_states {n_states} from idata['at']")
    if z_series != n_series:
        raise ValueError(f"Z_future series dim {z_series} does not match n_series {n_series} from idata['sigma_h']")

    resolved_exponent = float(idata.attrs.get("exponent", exponent))
    if "positivity" in idata:
        resolved_positivity = np.asarray(idata["positivity"].to_numpy(), dtype=bool)
    elif positivity is None:
        resolved_positivity = np.ones(n_states, dtype=bool)
    else:
        resolved_positivity = np.asarray(positivity, dtype=bool)
    if resolved_positivity.shape != (n_states,):
        raise ValueError(f"positivity must have shape ({n_states},); got {resolved_positivity.shape}")

    logger.debug(
        "forecast_ssp — n_sample: %d, horizon: %d, n_series: %d, n_states: %d, exponent: %s, noise_embed: %s",
        n_sample,
        horizon,
        n_series,
        n_states,
        resolved_exponent,
        noise_embed,
    )

    rng = np.random.default_rng(seed)

    # Driftless random walk forward from the last filtered state, one increment per step.
    a_T = at[:, -1, :]
    eta = rng.standard_normal((n_sample, horizon, n_states)) * sigma_q[:, None, :]
    a_path = a_T[:, None, :] + np.cumsum(eta, axis=1)

    # Natural scale: exp(exponent * a) on positivity states, identity elsewhere.
    a_nat = a_to_lam(a_path, resolved_exponent, resolved_positivity)

    # (sample, horizon, n_states) contracted with (horizon, n_series, n_states) -> (sample, horizon, n_series)
    mu_samples = np.einsum("hij,shj->shi", Z_future, a_nat)

    data_vars: dict[str, tuple[list[str], np.ndarray]] = {
        "mu_samples": (["sample", "time", "series"], mu_samples),
    }

    if noise_embed:
        eps_samples = rng.standard_normal((n_sample, horizon, n_series)) * sigma_h[:, None, :]
        forecast_samples = mu_samples + eps_samples
        data_vars["eps_samples"] = (["sample", "time", "series"], eps_samples)
    else:
        forecast_samples = mu_samples.copy()

    if transform_callback is not None:
        forecast_samples = transform_callback(forecast_samples)

    data_vars["forecast_samples"] = (["sample", "time", "series"], np.asarray(forecast_samples))

    coords = {
        "sample": np.arange(n_sample),
        "time": np.arange(horizon),
        "series": np.arange(n_series),
    }

    return xr.Dataset(data_vars=data_vars, coords=coords)
