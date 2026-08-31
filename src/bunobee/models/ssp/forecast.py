from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from bunobee.models.ssp.kalman_1d_st import kalman_filter_1d_st
from bunobee.models.ssp.kalman_1d_st_ekf import kalman_filter_1d_ekf_st
from bunobee.models.ssp.posterior import a_to_lam
from bunobee.regression import make_peridoic_dummies
from bunobee.utils import flatten_front_dim

logger = logging.getLogger("bunobee")

FORECAST_METHODS = ("replay", "filter")
FORECAST_LINKS = ("exp", "identity")

#: Hard floor the linear filters apply to positivity states every step (``kalman_1d_st.py``).
POSITIVITY_FLOOR = 1e-6
#: Bound the EKF puts on ``exponent * a`` before exponentiating (``kalman_1d_st_ekf.py``).
EXP_CLIP = 10.0


def _floored_walk(a_T: np.ndarray, eta: np.ndarray, positivity: np.ndarray) -> np.ndarray:
    """Walk the state forward one step at a time, flooring positivity states as we go.

    Reproduces the ``identity`` link's positivity mechanism: the linear filters
    (:func:`kalman_filter_1d_st`) clip a violating state back up **inside** the
    transition, so every clip resets the origin of the remaining walk. Applying
    ``np.maximum`` to a finished ``cumsum`` path is a different — and wrong —
    operation, because it lets the path accumulate arbitrarily far below zero and
    then lifts only the visible part back up.

    The resulting path is deliberately biased upward: a clipped random walk is not
    a martingale. That is the point — it is what the fitted model does.

    Parameters
    ----------
    a_T : np.ndarray, shape (n_sample, n_states)
        Last filtered state per posterior sample.
    eta : np.ndarray, shape (n_sample, horizon, n_states)
        Per-step state increments.
    positivity : np.ndarray, shape (n_states,)
        Boolean mask selecting the states held at or above :data:`POSITIVITY_FLOOR`.

    Returns
    -------
    np.ndarray, shape (n_sample, horizon, n_states)
        State path with the floor enforced at every step.
    """
    horizon = eta.shape[1]
    out = np.empty_like(eta)
    a = np.array(a_T, dtype=float)
    for h in range(horizon):
        a = a + eta[:, h, :]
        a[:, positivity] = np.maximum(a[:, positivity], POSITIVITY_FLOOR)
        out[:, h, :] = a
    return out


def _filter_increments(
    z: np.ndarray,
    *,
    a_T: np.ndarray,
    sigma_q: np.ndarray,
    sigma_h: np.ndarray,
    Z_future: np.ndarray,
    exponent: float,
    positivity: np.ndarray,
    link: str = "exp",
) -> np.ndarray:
    """Turn standard normals into state increments propagated by the filter itself.

    Runs the filter family named by ``link`` once per posterior sample over
    ``Z_future`` with an all-missing observation mask, so every forecast step is a
    pure predict step and the returned ``Pt`` is the exact predicted state
    covariance. The per-step increment covariance is ``dP[h] = Pt[h] - Pt[h - 1]``
    (with ``Pt[-1]`` the zero matrix the propagation starts from, since each
    posterior draw already carries the filtered uncertainty at ``T``), and the
    increments are ``chol(dP) @ z``.

    Parameters
    ----------
    z : np.ndarray, shape (n_sample, horizon, n_states)
        Standard normal draws, one per state per forecast step.
    a_T : np.ndarray, shape (n_sample, n_states)
        Last filtered state per posterior sample; the propagation starts here.
    sigma_q : np.ndarray, shape (n_sample, n_states)
        Process noise standard deviation per posterior sample.
    sigma_h : np.ndarray, shape (n_sample, n_series)
        Observation noise standard deviation per posterior sample. Unused by the
        predict-only step itself, but required by the filter signature.
    Z_future : np.ndarray, shape (horizon, n_series, n_states)
        Future design matrix.
    exponent : float
        Exponent in the nonlinear state map ``exp(exponent * a)``. Used by the
        ``"exp"`` link only.
    positivity : np.ndarray, shape (n_states,)
        Boolean mask selecting the positivity states.
    link : {"exp", "identity"}, optional
        Which filter family propagates the covariance: ``"exp"`` (default) uses
        :func:`kalman_filter_1d_ekf_st`, ``"identity"`` uses
        :func:`kalman_filter_1d_st`.

    Returns
    -------
    np.ndarray, shape (n_sample, horizon, n_states)
        State increments; the state path is ``a_T + cumsum(increments, axis=1)``
        for the ``"exp"`` link, or that walk with the per-step positivity floor
        applied for the ``"identity"`` link.

    Notes
    -----
    The filter runs in JAX's configured precision, float32 unless ``jax_enable_x64``
    is set, so the increments agree with the closed-form random walk to about
    single precision rather than bitwise.
    """
    n_sample, horizon, n_states = z.shape
    n_series = Z_future.shape[1]

    Z_jnp = jnp.asarray(Z_future)
    positivity_jnp = jnp.asarray(positivity, dtype=bool)
    P0 = jnp.zeros((n_states, n_states))
    # Fully missing observations: NaN values are never read, the mask makes each step predict-only.
    y_missing = jnp.full((horizon, n_series), jnp.nan)
    mask = jnp.zeros((horizon, n_series), dtype=bool)

    def _propagate(a0: jnp.ndarray, sig_h: jnp.ndarray, sig_q: jnp.ndarray) -> jnp.ndarray:
        kwargs = dict(
            a0=a0,
            P0=P0,
            Z=Z_jnp,
            sigma_h=sig_h,
            sigma_q=sig_q,
            y=y_missing,
            logp=False,
            mask=mask,
        )
        if link == "exp":
            # In the EKF `positivity` only selects the exp map used to linearise, so it
            # belongs in the propagation.
            _, _, Pt, _, _, _ = kalman_filter_1d_ekf_st(exponent=exponent, positivity=positivity_jnp, **kwargs)
        else:
            # In the linear filter `positivity` is the pseudo-observation correction, a
            # mean-space operation replayed per draw by `_floored_walk`. Passing it here
            # would inject a Pt inversion into a propagation that starts from P0 = 0, so
            # the covariance is propagated unconstrained and constrained afterwards.
            _, _, Pt, _, _, _ = kalman_filter_1d_st(**kwargs)
        return Pt

    Pt = jax.jit(jax.vmap(_propagate))(
        jnp.asarray(a_T),
        jnp.asarray(sigma_h),
        jnp.asarray(sigma_q),
    )
    # (n_sample, horizon, n_states, n_states)
    Pt = np.asarray(Pt, dtype=float)

    P_prev = np.concatenate([np.zeros((n_sample, 1, n_states, n_states)), Pt[:, :-1]], axis=1)
    dP = Pt - P_prev
    dP = 0.5 * (dP + np.swapaxes(dP, -1, -2))

    # Jitter keeps the factorization defined for a singular dP (sigma_q = 0 gives dP = 0);
    # the absolute floor is far below any state scale and the relative term absorbs the
    # cancellation error of the differencing above.
    scale = np.max(np.abs(dP), axis=(-2, -1), keepdims=True)
    jitter = 1e-24 + 1e-12 * scale
    L = np.linalg.cholesky(dP + jitter * np.eye(n_states))

    return np.einsum("shij,shj->shi", L, z)


def forecast_ssp(
    idata: xr.Dataset,
    Z_future: np.ndarray,
    *,
    method: str = "replay",
    link: str = "exp",
    exponent: float = 0.5,
    positivity: np.ndarray | None = None,
    noise_embed: bool = False,
    transform_callback: Callable | None = None,
    seed: int = 0,
) -> xr.Dataset:
    """Draw out-of-sample predictive samples from a fitted multi-series SSP posterior.

    The default is a posterior-replay forecast: it needs no change to the filter.
    The filtered state path already lives in ``idata`` as the ``at`` site, so the
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

    Positivity: which mechanism, and which filter
    --------------------------------------------
    The ``positivity`` mask means two different things depending on the filter
    family that produced the posterior, so ``link`` names the mechanism explicitly
    instead of leaving it to be inferred from the mask:

    ``link="exp"`` (default) — the EKF families ``kalman_filter_1d_ekf`` and
    ``kalman_filter_1d_ekf_st``. A masked state ``a`` is a **log-intensity**; its
    natural value is ``exp(exponent * a)``, positive by construction, and the
    filter enforces nothing. The forecast applies the same map, clipping
    ``exponent * a`` to ``[-EXP_CLIP, EXP_CLIP]`` exactly as the EKF does, so a long
    horizon with a large ``sigma_q`` cannot overflow to ``inf``.

    ``link="identity"`` — the linear families ``kalman_filter_1d`` and
    ``kalman_filter_1d_st``. A masked state **is already** the natural value, and
    positivity is enforced inside the transition at every step: a violating state is
    pulled back with a pseudo-observation and then floored at
    :data:`POSITIVITY_FLOOR`. The forecast never exponentiates such a state; it
    carries the enforcement forward by walking the state one step at a time and
    flooring after each increment (:func:`_floored_walk`). The clip is sequential on
    purpose — each clip resets the origin of the remaining walk, which is not the
    same as ``np.maximum`` over a finished ``cumsum`` path — and the resulting path
    is biased upward, because a clipped random walk is not a martingale and that is
    what the fitted model does.

    Passing an EKF posterior with ``link="identity"`` (or the reverse) is the failure
    this parameter exists to prevent, so a mask whose states go negative under
    ``link="identity"`` raises rather than forecasting a constraint the fit never
    allowed.

    ``method="filter"`` derives the same state path from the filter itself rather
    than hard-coding the random walk: it runs the filter for the resolved ``link``
    over ``Z_future`` with an all-missing observation mask, which makes every
    forecast step a pure predict step, then samples increments from the covariance
    the filter actually propagated::

        a_pred[h], P_pred[h] = filter(a0=a_T, P0=0, Z=Z_future, y=NaN, mask=False)
        dP[h]                = P_pred[h] - P_pred[h - 1]
        a_{T+h}              = a_pred[h] + sum_{k<=h} chol(dP[k]) @ z_k

    For the driftless random walk both paths coincide (``dP[h] = diag(sigma_q**2)``
    at every step), so ``"filter"`` reproduces the replay mean and variance up to
    floating point. It exists because the propagation is read off the filter, so a
    model whose predict step is not a plain random walk stays correct without
    touching this function.

    Parameters
    ----------
    idata : xr.Dataset
        Posterior carrying ``at`` with dims ``(chain, draw, n_steps, n_states)``,
        ``sigma_q`` with dims ``(chain, draw, n_states)``, and ``sigma_h`` with
        dims ``(chain, draw, n_series)``. When present, ``idata.attrs["exponent"]``
        overrides the ``exponent`` argument, ``idata.attrs["link"]`` overrides the
        ``link`` argument, and a ``positivity`` data variable overrides the
        ``positivity`` argument.
    Z_future : np.ndarray, shape (horizon, n_series, n_states)
        Future per-series design matrix. ``Z_future[h, j]`` is the loading vector
        for series ``j`` at forecast step ``h``. See ``build_forecast_design`` for
        a helper that continues periodic dummies and future covariates.
    method : {"replay", "filter"}, optional
        How the future state path is propagated. ``"replay"`` (default) walks the
        saved posterior state forward analytically; ``"filter"`` runs the filter
        for the resolved ``link`` over ``Z_future`` with an all-missing mask and
        samples from the covariance it propagates. Both give the same answer for
        the random-walk state equation; ``"filter"`` costs one extra filter pass
        per posterior sample.
    link : {"exp", "identity"}, optional
        Which positivity mechanism the fitted model used, and therefore how the
        ``positivity`` mask is honoured. ``"exp"`` (default) is the EKF
        reparameterization ``exp(exponent * a)``; ``"identity"`` is the linear
        filters' natural-scale state with a per-step floor at
        :data:`POSITIVITY_FLOOR`. Ignored when ``idata.attrs`` carries a ``"link"``
        entry. See the section above for which filter goes with which value.
    exponent : float, optional
        Exponent in the nonlinear state map ``exp(exponent * a)``. Default 0.5.
        Ignored when ``idata.attrs`` carries an ``"exponent"`` entry, and unused
        entirely when ``link="identity"``.
    positivity : np.ndarray or None, optional, shape (n_states,)
        Boolean mask; ``True`` selects the positivity states — those that use the
        nonlinear ``exp`` map under ``link="exp"``, or that are floored every step
        under ``link="identity"``. ``None`` (default) marks every state, matching
        the filter default. Pass ``np.zeros(n_states, dtype=bool)`` for a fully
        unconstrained linear forecast. Ignored when ``idata`` carries a
        ``positivity`` data variable.
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

    Raises
    ------
    ValueError
        If ``method`` or ``link`` is not one of the supported values, if
        ``Z_future`` does not conform to ``idata``, or if ``link="identity"`` is
        paired with a posterior whose masked states go negative — the signature of
        a posterior that no linear positivity enforcement ever produced, most often
        an EKF (log-scale) posterior forecast under the wrong mechanism.
    """
    if method not in FORECAST_METHODS:
        raise ValueError(f"method must be one of {FORECAST_METHODS}; got {method!r}")
    resolved_link = str(idata.attrs.get("link", link))
    if resolved_link not in FORECAST_LINKS:
        raise ValueError(f"link must be one of {FORECAST_LINKS}; got {resolved_link!r}")

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

    if resolved_link == "identity" and resolved_positivity.any():
        # A linear filter floors every masked state at POSITIVITY_FLOOR on every training
        # step, so a negative masked state cannot come out of one. If we see one the
        # mechanism is misdeclared -- typically an EKF (log-scale) posterior asked to
        # forecast on the natural scale -- and honouring it silently would forecast a
        # constraint the fit never allowed.
        offenders = np.flatnonzero(resolved_positivity & (at < 0.0).any(axis=(0, 1)))
        if offenders.size:
            raise ValueError(
                f"link='identity' marks states {offenders.tolist()} as positivity states, but idata['at'] "
                "holds negative values for them; a linear filter floors positivity states every step, so "
                "this posterior was not fitted with that mechanism. Use link='exp' for an EKF posterior "
                "(states are log-intensities), or drop those states from the positivity mask."
            )

    logger.debug(
        "forecast_ssp — n_sample: %d, horizon: %d, n_series: %d, n_states: %d, exponent: %s, "
        "link: %s, method: %s, noise_embed: %s",
        n_sample,
        horizon,
        n_series,
        n_states,
        resolved_exponent,
        resolved_link,
        method,
        noise_embed,
    )

    rng = np.random.default_rng(seed)

    # Standard normals are drawn identically by both methods so a fixed seed keeps the
    # two paths comparable draw for draw.
    a_T = at[:, -1, :]
    z = rng.standard_normal((n_sample, horizon, n_states))
    if method == "replay":
        # Driftless random walk forward from the last filtered state, one increment per step.
        eta = z * sigma_q[:, None, :]
    else:
        eta = _filter_increments(
            z,
            a_T=a_T,
            sigma_q=sigma_q,
            sigma_h=sigma_h,
            Z_future=Z_future,
            exponent=resolved_exponent,
            positivity=resolved_positivity,
            link=resolved_link,
        )

    if resolved_link == "exp":
        # EKF posterior: the state is a log-intensity, positivity holds by construction.
        a_path = a_T[:, None, :] + np.cumsum(eta, axis=1)
        # Natural scale: exp(exponent * a) on positivity states, identity elsewhere. The
        # clip mirrors kalman_filter_1d_ekf_st so a long horizon cannot overflow to inf.
        a_nat = a_to_lam(a_path, resolved_exponent, resolved_positivity, clip=EXP_CLIP)
    else:
        # Linear posterior: the state is already natural scale, so the walk itself carries
        # the filter's per-step floor and nothing is exponentiated.
        a_nat = _floored_walk(a_T, eta, resolved_positivity)

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


def _normalize_columns(columns, n_states: int) -> np.ndarray:
    """Resolve a slice / index sequence into a 1-D array of column positions."""
    if isinstance(columns, slice):
        return np.arange(n_states)[columns]
    idx = np.atleast_1d(np.asarray(columns))
    if idx.dtype == bool:
        if idx.shape != (n_states,):
            raise ValueError(f"boolean column mask must have shape ({n_states},); got {idx.shape}")
        return np.flatnonzero(idx)
    return idx.astype(int)


def _periodic_specs(periodic) -> list[Mapping]:
    """Normalize the ``periodic`` argument into a list of block specs."""
    if periodic is None:
        return []
    if isinstance(periodic, Mapping):
        return [periodic]
    if isinstance(periodic, Sequence):
        return list(periodic)
    raise TypeError(f"periodic must be a mapping or a sequence of mappings; got {type(periodic).__name__}")


def _covariate_items(covariates_future, n_states: int) -> dict[int, np.ndarray]:
    """Normalize ``covariates_future`` into a ``{column: future values}`` mapping."""
    if covariates_future is None:
        return {}
    if isinstance(covariates_future, Mapping):
        return {int(col): np.asarray(values, dtype=float) for col, values in covariates_future.items()}

    arr = np.asarray(covariates_future, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim not in (2, 3):
        raise ValueError(
            "covariates_future array must be (horizon, k) or (horizon, n_series, k); " f"got shape {arr.shape}"
        )
    n_cov = arr.shape[-1]
    if n_cov > n_states:
        raise ValueError(f"covariates_future supplies {n_cov} columns but Z_train has only {n_states}")
    cols = range(n_states - n_cov, n_states)
    return {col: arr[..., i] for i, col in enumerate(cols)}


def build_forecast_design(
    Z_train: np.ndarray,
    horizon: int,
    *,
    periodic: Mapping | Sequence[Mapping] | None = None,
    covariates_future: Mapping[int, np.ndarray] | np.ndarray | None = None,
) -> np.ndarray:
    """Continue a training design matrix ``horizon`` steps past the end of the sample.

    Builds the ``Z_future`` argument that :func:`forecast_ssp` consumes. Every state
    column of ``Z_train`` is resolved by exactly one of three rules:

    1. **Periodic** — columns named by a ``periodic`` block are regenerated with
       ``make_peridoic_dummies(n_steps + horizon, period)[n_steps:]``, so the cycle
       phase continues unbroken from the training window.
    2. **Covariate** — columns supplied through ``covariates_future`` take the given
       future values.
    3. **Constant** — any remaining column must be constant over training time (an
       intercept, a per-series indicator, a static control); its value is carried
       forward. A remaining column that varies over time raises ``ValueError``,
       since its future path cannot be known.

    ``forecast_ssp`` also accepts a hand-built ``Z_future``; this helper is a
    convenience for the common intercept + seasonal-dummy + covariate layout.

    Parameters
    ----------
    Z_train : np.ndarray, shape (n_steps, n_series, n_states) or (n_steps, n_states)
        In-sample design matrix. A 2-D array is treated as a single series and the
        returned array still carries an explicit series axis of length 1.
    horizon : int
        Number of future steps to build. Must be positive.
    periodic : mapping or sequence of mappings or None, optional
        One spec per periodic dummy block. Keys:

        ``"columns"``
            Slice, integer sequence, or boolean mask selecting the block's columns.
        ``"period"``
            Cycle length (e.g. 7 for weekly dummies).
        ``"drop_first"``
            Whether the block dropped its first dummy column. Default ``True``.

        The block width must equal ``period - int(drop_first)`` and the training rows
        of the block must match the dummies implied by the spec, otherwise
        ``ValueError`` is raised.
    covariates_future : mapping or np.ndarray or None, optional
        Future values for time-varying non-periodic columns. As a mapping, keys are
        column positions and values broadcast to ``(horizon, n_series)``. As a plain
        array of shape ``(horizon, k)`` or ``(horizon, n_series, k)``, the ``k``
        columns fill the last ``k`` columns of the design matrix. Any leading axis of
        the wrong length raises ``ValueError``.

    Returns
    -------
    np.ndarray, shape (horizon, n_series, n_states)
        Future design matrix, shape-compatible with ``forecast_ssp``.

    Examples
    --------
    >>> import numpy as np
    >>> from bunobee.regression import make_peridoic_dummies
    >>> n_steps = 28
    >>> Z_train = np.concatenate([np.ones((n_steps, 1)), make_peridoic_dummies(n_steps, 7)], axis=1)
    >>> Z_future = build_forecast_design(Z_train, 14, periodic={"columns": slice(1, None), "period": 7})
    >>> Z_future.shape
    (14, 1, 7)
    """
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError(f"horizon must be a positive integer; got {horizon}")

    Z_train = np.asarray(Z_train, dtype=float)
    if Z_train.ndim == 2:
        Z_train = Z_train[:, None, :]
    elif Z_train.ndim != 3:
        raise ValueError(
            f"Z_train must be 2-D (n_steps, n_states) or 3-D (n_steps, n_series, n_states); got {Z_train.shape}"
        )
    n_steps, n_series, n_states = Z_train.shape
    if n_steps == 0:
        raise ValueError("Z_train must have at least one training step")

    Z_future = np.zeros((horizon, n_series, n_states), dtype=float)
    resolved = np.zeros(n_states, dtype=bool)

    for spec in _periodic_specs(periodic):
        period = int(spec["period"])
        drop_first = bool(spec.get("drop_first", True))
        cols = _normalize_columns(spec["columns"], n_states)
        width = period - int(drop_first)
        if cols.size != width:
            raise ValueError(
                f"periodic block with period {period} and drop_first={drop_first} needs {width} columns; "
                f"got {cols.size}"
            )
        dummies = np.asarray(
            make_peridoic_dummies(n_steps + horizon, period=period, drop_first=drop_first), dtype=float
        )
        if not np.allclose(Z_train[:, :, cols], dummies[:n_steps][:, None, :]):
            raise ValueError(
                f"columns {cols.tolist()} of Z_train do not match periodic dummies with period {period} and "
                f"drop_first={drop_first}; check the column selection and the cycle phase"
            )
        Z_future[:, :, cols] = dummies[n_steps:][:, None, :]
        resolved[cols] = True

    for col, values in _covariate_items(covariates_future, n_states).items():
        values = np.asarray(values, dtype=float)
        if values.ndim == 0 or values.shape[0] != horizon:
            raise ValueError(
                f"covariates_future for column {col} must have leading dimension {horizon}; got {values.shape}"
            )
        try:
            Z_future[:, :, col] = np.broadcast_to(values.reshape(horizon, -1), (horizon, n_series))
        except ValueError as exc:
            raise ValueError(
                f"covariates_future for column {col} must broadcast to ({horizon}, {n_series}); got {values.shape}"
            ) from exc
        resolved[col] = True

    for col in np.flatnonzero(~resolved):
        block = Z_train[:, :, col]
        if not np.allclose(block, block[0]):
            raise ValueError(
                f"column {col} of Z_train varies over time but is neither periodic nor supplied through "
                "covariates_future; its future path is unknown"
            )
        Z_future[:, :, col] = block[0]

    logger.debug(
        "build_forecast_design — n_steps: %d, horizon: %d, n_series: %d, n_states: %d",
        n_steps,
        horizon,
        n_series,
        n_states,
    )

    return Z_future
