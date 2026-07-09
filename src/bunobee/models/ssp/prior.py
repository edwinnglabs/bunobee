from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import xarray as xr

from bunobee.models.ssp.kalman_1d import kalman_filter_1d, kalman_rts_smoother_1d
from bunobee.models.ssp.transforms import validate_prior

REQUIRED_COORDS: tuple[str, ...] = ("time", "state")


@dataclass(frozen=True)
class SspPrior:
    """Validated wrapper around a complete SSP time-point prior.

    Enforces the full SSP prior contract (see :func:`validate_prior`) plus
    two additions specific to the complete, filter-ready stage: ``a_obs`` /
    ``P_obs`` are unconditionally required (not optional), and ``time`` /
    ``state`` must be actual coordinates, not just bare dimensions. Runs at
    construction time, regardless of how ``dataset`` was built -- via
    :func:`~bunobee.simulation.ssp.construct_states_prior`, a
    hand-built ``xr.Dataset``, or one loaded from disk. Any additional
    data vars, coords, or attrs (``sdy``,
    ``sigma_q_loc_prior``, ...) are unrestricted and pass through unchanged.

    Parameters
    ----------
    dataset : xr.Dataset
        Prior dataset satisfying the SSP prior schema.

    Raises
    ------
    ValueError
        If ``dataset`` violates the SSP prior contract, is missing
        ``a_obs`` / ``P_obs``, or is missing the ``time`` / ``state``
        coordinates.
    """

    dataset: xr.Dataset

    def __post_init__(self) -> None:
        _validate(self.dataset)

    @property
    def a0(self) -> xr.DataArray:
        """Initial state mean, dims ``(state,)``."""
        return self.dataset["a0"]

    @property
    def P0(self) -> xr.DataArray:
        """Initial state covariance: diagonal ``(state,)`` or full ``(state, state_dual)``."""
        return self.dataset["P0"]

    @property
    def a_obs(self) -> xr.DataArray:
        """Disclosed-state means, dims ``(time, state)``."""
        return self.dataset["a_obs"]

    @property
    def P_obs(self) -> xr.DataArray:
        """Disclosed-state variances, dims ``(time, state)``; ``inf`` where undisclosed."""
        return self.dataset["P_obs"]

    @property
    def positivity(self) -> xr.DataArray:
        """Boolean positivity mask, dims ``(state,)``."""
        return self.dataset["positivity"]

    def __getitem__(self, item: str):
        return self.dataset[item]

    def __getattr__(self, name: str):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self) -> int:
        return len(self.dataset)


def _validate(ds: xr.Dataset) -> None:
    """Check ``ds`` against the complete SSP prior schema.

    Delegates the shared contract to :func:`validate_prior`
    (``require_init=True``), then adds the two constraints specific to
    :class:`SspPrior`: ``a_obs`` / ``P_obs`` are required (not optional),
    and ``time`` / ``state`` must be actual coordinates.

    Parameters
    ----------
    ds : xr.Dataset
        Candidate prior dataset.

    Raises
    ------
    ValueError
        If ``ds`` fails :func:`validate_prior`, is missing ``a_obs`` /
        ``P_obs``, or is missing the ``time`` / ``state`` coordinates.
    """
    validate_prior(ds, require_init=True)

    for name in ("a_obs", "P_obs"):
        if name not in ds.data_vars:
            raise ValueError(f"SspPrior requires a `{name}` variable")
    for name in REQUIRED_COORDS:
        if name not in ds.coords:
            raise ValueError(f"SspPrior requires a `{name}` coordinate")


def disclosed_idx(ssp_prior: xr.Dataset | SspPrior) -> np.ndarray:
    """Return the time indices where the prior discloses state information.

    Replaces the previously stored ``obs_idx`` variable, which was redundant
    with ``P_obs``: a disclosure exists at any timestep with at least one
    finite-variance state, since undisclosed timesteps carry ``inf`` variance
    (zero precision, a pure filter carry-through).

    Parameters
    ----------
    ssp_priors : xr.Dataset or SspPrior
        Prior dataset containing a ``P_obs`` variable with dims
        ``(time, state)``.

    Returns
    -------
    np.ndarray
        Sorted integer indices into the ``time`` axis with at least one
        disclosed (finite-variance) state.

    Raises
    ------
    KeyError
        If ``ssp_priors`` has no ``P_obs`` variable.
    """
    if "P_obs" not in ssp_prior:
        raise KeyError("ssp_priors has no `P_obs`; cannot derive disclosure indices")
    p_obs = np.asarray(ssp_prior["P_obs"].values)
    return np.where(np.isfinite(p_obs).any(axis=1))[0]


def extend_states_prior_nearest(
    ssp_priors: xr.Dataset,
    Q: float | np.ndarray,
) -> xr.Dataset:
    r"""Extend disclosed anchors along the nearest-anchor random-walk marginal.

    :func:`~bunobee.simulation.ssp.construct_states_prior` discloses anchors at a handful of timesteps
    and leaves every other step at ``P_obs = inf`` (undisclosed / uninformed).
    For a single isolated channel touched by no other data those null steps are
    not truly uninformed: the driftless random-walk transition already implies
    an exact marginal that propagates each anchor both forward and backward —
    the mean is carried constant and the variance grows linearly with the time
    lag.  This function fills each formerly-``inf`` entry with that two-sided
    marginal,

    .. math::

        a_{\mathrm{obs}}[t] = a^*, \qquad
        P_{\mathrm{obs}}[t] = P^* + |t - t^*|\,Q,

    where ``(a*, P*)`` is the disclosed anchor at the nearest disclosure time
    ``t*`` in that state (minimum variogram distance ``|t - t*|``) and ``Q`` is
    the per-state process variance ``σ_q²``.  States with no disclosed anchor
    are left fully undisclosed (``P_obs`` stays ``inf``).

    This is the exact marginal only for an isolated channel; treat the result as
    a prior fed into the augmented-measurement step, not a final posterior.

    **When to use.** This is the cheap heuristic: a pure-``numpy`` nearest-anchor
    fill, exact for a single-anchor channel but discontinuous and variance-inflating
    once a state carries two or more anchors (it keeps only the nearest anchor and
    never blends means). For an exact multi-anchor marginal that fuses every anchor,
    use :func:`extend_states_prior_smoothed` instead.

    Parameters
    ----------
    ssp_priors : xr.Dataset
        Disclosed prior as produced by
        :func:`~bunobee.simulation.ssp.construct_states_prior`, carrying
        ``a_obs`` and ``P_obs`` over dims ``(time, state)``.  All other
        variables (``positivity``, any ``a0`` / ``P0`` or ``sigma_q`` block, and
        attrs) are passed through unchanged.
    Q : float or np.ndarray
        Per-state process variance ``σ_q²``.  A scalar is broadcast to every
        state; an array must have length ``n_states``.  Must be finite-or-``inf``
        and non-negative.  ``Q → inf`` recovers the un-extended prior (only the
        original anchors stay informed), while ``Q → 0`` holds each anchor's
        variance flat across its region.

    Returns
    -------
    xr.Dataset
        Copy of ``ssp_priors`` whose ``a_obs`` / ``P_obs`` have every anchored
        state's undisclosed steps filled by the nearest-anchor random-walk
        marginal.  Passes :func:`validate_prior` with ``require_init=False``.

    Raises
    ------
    ValueError
        If ``ssp_priors`` lacks the ``a_obs`` / ``P_obs`` disclosure block, or
        ``Q`` is negative, ``NaN``, or has the wrong length.
    """
    if "a_obs" not in ssp_priors or "P_obs" not in ssp_priors:
        raise ValueError("ssp_priors must contain `a_obs` and `P_obs` to extend the prior")

    a_obs = np.array(ssp_priors["a_obs"].values, dtype=float)
    p_obs = np.array(ssp_priors["P_obs"].values, dtype=float)
    n_steps, n_states = p_obs.shape

    q_vec = np.broadcast_to(np.asarray(Q, dtype=float), (n_states,)).astype(float, copy=True)
    if np.any(np.isnan(q_vec)) or np.any(q_vec < 0):
        raise ValueError("Q must be non-negative (finite or inf) for every state")

    tgrid = np.arange(n_steps)
    for s in range(n_states):
        anchor_t = np.where(np.isfinite(p_obs[:, s]))[0]
        if anchor_t.size == 0:
            continue  # no anchor -> state stays fully undisclosed (inf)

        anchor_a = a_obs[anchor_t, s]
        anchor_p = p_obs[anchor_t, s]
        dist = np.abs(tgrid[:, None] - anchor_t[None, :])  # (n_steps, n_anchors)

        # Multiply only at nonzero lags so 0 * inf never arises: anchors stay
        # exact (lag 0 -> variance P*) even when Q is inf.
        lag_var = np.zeros_like(dist, dtype=float)
        np.multiply(dist, q_vec[s], out=lag_var, where=dist != 0)
        cand_p = anchor_p[None, :] + lag_var  # (n_steps, n_anchors)

        # Nearest anchor by variogram distance; ties broken toward smaller variance.
        min_dist = dist.min(axis=1, keepdims=True)
        chosen = np.argmin(np.where(dist == min_dist, cand_p, np.inf), axis=1)

        a_obs[:, s] = anchor_a[chosen]
        p_obs[:, s] = cand_p[tgrid, chosen]

    out = ssp_priors.copy()
    out["a_obs"] = (("time", "state"), a_obs)
    out["P_obs"] = (("time", "state"), p_obs)
    validate_prior(out, require_init=False)
    return out


def extend_states_prior_smoothed(
    ssp_priors: xr.Dataset,
    Q: float | np.ndarray,
    P0_diffuse: float = 1e8,
) -> xr.Dataset:
    r"""Extend disclosed anchors via an exact KF-forward + RTS-backward smoother.

    Like :func:`extend_states_prior_nearest`, this fills every undisclosed step of an
    anchored state along the driftless random walk :math:`x_t = x_{t-1} + w_t`,
    :math:`\mathrm{Var}(w_t) = Q`.  But rather than snapping each step to its nearest
    anchor, it treats each state column as an independent univariate time series and
    runs bunobee's own diagonal filter/smoother over it:

    1. :func:`~bunobee.models.ssp.kalman_1d.kalman_filter_1d` in *extension mode* --
       a zero design matrix ``Z`` so there is no scalar ``y`` observation coupling the
       states.  Each disclosed anchor :math:`(a^*, P^*)` enters through the filter's
       precision-weighted state-fusion path (as a *soft* observation with noise
       variance :math:`P^*`), while ``P_obs = inf`` steps are pure predict-through.  A
       large finite ``P0_diffuse`` stands in for an uninformative initial prior.
    2. :func:`~bunobee.models.ssp.kalman_1d.kalman_rts_smoother_1d` -- a backward pass
       that returns the smoothed mean **and** marginal variance at every step.

    For this linear-Gaussian random walk the RTS smoother is exact: it returns the true
    per-step posterior marginal, fusing *every* anchor in the channel by inverse-variance
    weighting.  Compared with the nearest-anchor heuristic it blends means between anchors
    instead of hard-switching, tightens the variance (never wider), and revises even the
    disclosed anchors once more than one is present.  States with no disclosed anchor come
    back diffuse-but-finite from the filter and are reset here to ``inf`` (fully
    undisclosed), matching :func:`extend_states_prior_nearest`.

    **When to use.** Prefer this for genuinely multi-anchor channels, where the
    nearest-anchor heuristic is a conservative, discontinuous approximation.  For a
    single-anchor channel the two agree to numerical noise (set by the diffuse-prior
    proxy ``P0_diffuse``), so the cheaper :func:`extend_states_prior_nearest` suffices.

    Parameters
    ----------
    ssp_priors : xr.Dataset
        Disclosed prior as produced by
        :func:`~bunobee.simulation.ssp.construct_states_prior`, carrying ``a_obs`` and
        ``P_obs`` over dims ``(time, state)``.  All other variables (``positivity``, any
        ``a0`` / ``P0`` or ``sigma_q`` block, and attrs) are passed through unchanged.
    Q : float or np.ndarray
        Per-state process variance ``σ_q²``.  A scalar is broadcast to every state; an
        array must have length ``n_states``.  Must be finite-or-``inf`` and non-negative.
        The filter itself takes a standard deviation, so internally this passes
        ``sqrt(Q)``.
    P0_diffuse : float, optional
        Large finite initial variance standing in for an uninformative prior, by default
        ``1e8`` (comfortable in float64).

    Returns
    -------
    xr.Dataset
        Copy of ``ssp_priors`` whose ``a_obs`` / ``P_obs`` have every anchored state's
        undisclosed steps filled by the exact smoother marginal.  Passes
        :func:`validate_prior` with ``require_init=False``.

    Raises
    ------
    ValueError
        If ``ssp_priors`` lacks the ``a_obs`` / ``P_obs`` disclosure block, or ``Q`` is
        negative, ``NaN``, or has the wrong length.
    """
    if "a_obs" not in ssp_priors or "P_obs" not in ssp_priors:
        raise ValueError("ssp_priors must contain `a_obs` and `P_obs` to extend the prior")

    a_obs = np.array(ssp_priors["a_obs"].values, dtype=float)
    p_obs = np.array(ssp_priors["P_obs"].values, dtype=float)
    n_steps, n_states = p_obs.shape

    q_vec = np.broadcast_to(np.asarray(Q, dtype=float), (n_states,)).astype(float, copy=True)
    if np.any(np.isnan(q_vec)) or np.any(q_vec < 0):
        raise ValueError("Q must be non-negative (finite or inf) for every state")

    sigma_q = jnp.sqrt(jnp.asarray(q_vec))

    # Extension mode: Z = 0 -> zero Kalman gain -> no y-update, states stay independent
    # univariate random walks fed only through the anchor state-fusion path.
    a0 = jnp.zeros(n_states)
    P0 = jnp.full(n_states, P0_diffuse)
    Z = jnp.zeros((n_steps, n_states))
    y = jnp.zeros(n_steps)

    _, at, Pt, *_ = kalman_filter_1d(
        a0=a0,
        P0=P0,
        Z=Z,
        sigma_h=jnp.array(1.0),
        sigma_q=sigma_q,
        y=y,
        a_obs=jnp.asarray(a_obs),
        P_obs=jnp.asarray(p_obs),
    )
    at_smooth, Pt_smooth = kalman_rts_smoother_1d(at=at, Pt=Pt, sigma_q=sigma_q)

    a_out = np.array(at_smooth, dtype=float)
    p_out = np.array(Pt_smooth, dtype=float)

    # A state with no anchor comes back diffuse-but-finite from the diffuse P0; reset it
    # to fully-undisclosed (inf), matching extend_states_prior_nearest.
    unanchored = ~np.isfinite(p_obs).any(axis=0)
    a_out[:, unanchored] = a_obs[:, unanchored]
    p_out[:, unanchored] = p_obs[:, unanchored]

    out = ssp_priors.copy()
    out["a_obs"] = (("time", "state"), a_out)
    out["P_obs"] = (("time", "state"), p_out)
    validate_prior(out, require_init=False)
    return out
