from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import xarray as xr

from bunobee.models.ssp.kalman_1d import kalman_filter_1d, kalman_rts_smoother_1d
from bunobee.models.ssp.transforms import _unwrap, validate_prior

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

    Read access is a thin facade over the wrapped dataset: ``prior["a0"]``,
    ``prior.sizes``, ``"time" in prior``, ``len(prior)``, and iteration all
    behave exactly as they do on the dataset itself. Every public
    prior-consuming function accepts the wrapper and a bare ``xr.Dataset``
    interchangeably, so promoting a prior is never forced on a caller.

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

    Notes
    -----
    Two limits are known and deliberately not papered over.

    ``xr.merge`` rejects the wrapper outright, raising ``TypeError: objects
    must be an iterable containing only ... Dataset(s), DataArray(s), and
    dictionaries``. Merging is the usual construction idiom -- ``prior_ds =
    xr.merge([base_prior, states_prior])`` -- so a *wrapped* prior must be
    unwrapped with :meth:`to_dataset` before it can take part in one.

    The wrapper does not survive xarray operations. ``.copy()``, ``.sel()``,
    ``.isel()``, ``.assign()``, and netCDF / zarr round-trips all return a bare
    ``xr.Dataset``, silently and without error. The invariant therefore holds
    **at function boundaries only**: :class:`SspPrior` is a checkpoint that a
    dataset was valid when it was promoted, not a durable guarantee that
    everything derived from it still is. Re-promote with
    :meth:`from_dataset` after any such operation to re-check the contract.
    """

    dataset: xr.Dataset

    def __post_init__(self) -> None:
        _validate(self.dataset)

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> SspPrior:
        """Promote an ``xr.Dataset`` to a validated :class:`SspPrior`.

        Named alternative to the constructor, for readability at the call site
        and for re-checking a dataset that fell out of the wrapper through an
        xarray operation (see the class Notes).

        Parameters
        ----------
        dataset : xr.Dataset
            Prior dataset satisfying the SSP prior schema.

        Returns
        -------
        SspPrior
            Wrapper around ``dataset``, validated at construction.

        Raises
        ------
        ValueError
            If ``dataset`` violates the complete SSP prior contract.
        """
        return cls(dataset=dataset)

    def to_dataset(self) -> xr.Dataset:
        """Return the wrapped ``xr.Dataset``.

        The inverse of :meth:`from_dataset`, and the escape hatch for the
        xarray APIs that reject the wrapper -- ``xr.merge`` above all (see the
        class Notes). ``SspPrior.from_dataset(prior.to_dataset()) == prior``.

        Returns
        -------
        xr.Dataset
            The underlying dataset, not a copy.
        """
        return self.dataset

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

    def __contains__(self, key: object) -> bool:
        """Delegate membership to the dataset, so coords count as present.

        ``xr.Dataset.__contains__`` tests every variable, coordinates
        included, while ``__iter__`` yields data-var names only. Without this
        method ``in`` would fall back to iteration and ``"time" in prior``
        would be ``False`` where ``"time" in prior.dataset`` is ``True``.
        """
        return key in self.dataset

    def __iter__(self):
        """Iterate the dataset's data-var names, as ``xr.Dataset`` does."""
        return iter(self.dataset)

    def __len__(self) -> int:
        """Return ``len(dataset)`` -- the number of data vars, as xarray counts them."""
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
    ssp_prior = _unwrap(ssp_prior)
    if "P_obs" not in ssp_prior:
        raise KeyError("ssp_priors has no `P_obs`; cannot derive disclosure indices")
    p_obs = np.asarray(ssp_prior["P_obs"].values)
    return np.where(np.isfinite(p_obs).any(axis=1))[0]


def _write_init_moments(
    out: xr.Dataset,
    a_init: np.ndarray,
    P_init: np.ndarray,
    overwrite_init: bool,
) -> None:
    """Write the derived ``a0`` / ``P0`` onto ``out`` without outranking a real prior.

    The moments an extension derives — the driftless random-walk marginal evaluated one step
    before the series — are only a *placeholder*: the authoritative initial-state prior is the
    one the caller supplies downstream. So each of ``a0`` / ``P0`` is written only when it is
    absent from ``out``, independently of the other, unless ``overwrite_init`` says otherwise.

    Parameters
    ----------
    out : xr.Dataset
        Extended prior, modified in place.
    a_init : np.ndarray
        Derived initial-state mean over dims ``(state,)``.
    P_init : np.ndarray
        Derived initial-state variance over dims ``(state,)``.
    overwrite_init : bool
        Whether the derived moments replace an ``a0`` / ``P0`` already present on ``out``.
    """
    if overwrite_init or "a0" not in out:
        out["a0"] = (("state",), a_init)
    if overwrite_init or "P0" not in out:
        out["P0"] = (("state",), P_init)


def extend_states_prior_nearest(
    ssp_priors: xr.Dataset | SspPrior,
    Q: float | np.ndarray,
    *,
    overwrite_init: bool = False,
) -> SspPrior:
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

    The same marginal is evaluated one step *before* the series at ``t = -1`` and
    returned as the initial-state moments ``a0`` / ``P0`` over dims ``(state,)``.
    The nearest anchor to ``t = -1`` is always the *first* one, so in closed form
    ``a0 = a*`` and ``P0 = P* + (t* + 1)·Q``; an unanchored state gets
    ``a0 = 0``, ``P0 = inf``, the same "no information" encoding ``P_obs``
    already uses.  The ``(time, state)`` rectangle keeps its shape — ``time``
    stays length ``T`` — so nothing that indexes by step is disturbed.

    **Precedence.** Those derived moments are a *placeholder*, not an authority: a real
    initial-state prior the caller supplies (the ``base_ds`` block merged downstream)
    always wins.  Each of ``a0`` / ``P0`` is therefore written only when it is absent from
    ``ssp_priors``, independently of the other; pass ``overwrite_init=True`` to let the
    derived placeholder replace one that is already there.

    This is the exact marginal only for an isolated channel; treat the result as
    a prior fed into the augmented-measurement step, not a final posterior.

    **When to use.** This is the cheap heuristic: a pure-``numpy`` nearest-anchor
    fill, exact for a single-anchor channel but discontinuous and variance-inflating
    once a state carries two or more anchors (it keeps only the nearest anchor and
    never blends means). For an exact multi-anchor marginal that fuses every anchor,
    use :func:`extend_states_prior_smoothed` instead.

    Parameters
    ----------
    ssp_priors : xr.Dataset or SspPrior
        Disclosed prior as produced by
        :func:`~bunobee.simulation.ssp.construct_states_prior`, carrying
        ``a_obs`` and ``P_obs`` over dims ``(time, state)``.  All other
        variables (``positivity``, any ``sigma_q`` block, and attrs) are passed
        through unchanged; an ``a0`` / ``P0`` already present is **preserved**
        unless ``overwrite_init`` is set.
    Q : float or np.ndarray
        Per-state process variance ``σ_q²``.  A scalar is broadcast to every
        state; an array must have length ``n_states``.  Must be finite-or-``inf``
        and non-negative.  ``Q → inf`` recovers the un-extended prior (only the
        original anchors stay informed, and ``P0`` stays ``inf``), while
        ``Q → 0`` holds each anchor's variance flat across its region.
    overwrite_init : bool, optional
        Whether the derived ``a0`` / ``P0`` replace ones already present on
        ``ssp_priors``, by default ``False`` (a real initial-state prior wins).

    Returns
    -------
    SspPrior
        Copy of ``ssp_priors`` whose ``a_obs`` / ``P_obs`` have every anchored
        state's undisclosed steps filled by the nearest-anchor random-walk
        marginal, plus ``a0`` / ``P0`` over ``(state,)`` from the same marginal
        at ``t = -1`` wherever the input did not already carry them, promoted
        to an :class:`SspPrior` — the output satisfies the complete-prior
        contract by construction.  A ``P0 = inf`` column (an unanchored state)
        still needs handling before the a-space transforms can consume it.
        Reach the plain dataset with :meth:`SspPrior.to_dataset` when an
        xarray API needs it.

    Raises
    ------
    ValueError
        If ``ssp_priors`` lacks the ``a_obs`` / ``P_obs`` disclosure block, is
        missing the ``time`` / ``state`` coordinates the :class:`SspPrior`
        contract requires, or ``Q`` is negative, ``NaN``, or has the wrong
        length.
    """
    ssp_priors = _unwrap(ssp_priors)
    if "a_obs" not in ssp_priors or "P_obs" not in ssp_priors:
        raise ValueError("ssp_priors must contain `a_obs` and `P_obs` to extend the prior")

    a_obs = np.array(ssp_priors["a_obs"].values, dtype=float)
    p_obs = np.array(ssp_priors["P_obs"].values, dtype=float)
    n_steps, n_states = p_obs.shape

    q_vec = np.broadcast_to(np.asarray(Q, dtype=float), (n_states,)).astype(float, copy=True)
    if np.any(np.isnan(q_vec)) or np.any(q_vec < 0):
        raise ValueError("Q must be non-negative (finite or inf) for every state")

    # Row 0 of the grid is t = -1, the initial-state step: the same marginal one
    # step before the series, which becomes (a0, P0).  Rows 1: are the rectangle.
    tgrid = np.arange(-1, n_steps)
    rows = np.arange(tgrid.size)
    a_init = np.zeros(n_states)
    P_init = np.full(n_states, np.inf)

    for s in range(n_states):
        anchor_t = np.where(np.isfinite(p_obs[:, s]))[0]
        if anchor_t.size == 0:
            continue  # no anchor -> state stays undisclosed (inf) at every step, a0 = 0 / P0 = inf

        anchor_a = a_obs[anchor_t, s]
        anchor_p = p_obs[anchor_t, s]
        dist = np.abs(tgrid[:, None] - anchor_t[None, :])  # (n_steps + 1, n_anchors)

        # Multiply only at nonzero lags so 0 * inf never arises: anchors stay
        # exact (lag 0 -> variance P*) even when Q is inf.
        lag_var = np.zeros_like(dist, dtype=float)
        np.multiply(dist, q_vec[s], out=lag_var, where=dist != 0)
        cand_p = anchor_p[None, :] + lag_var  # (n_steps + 1, n_anchors)

        # Nearest anchor by variogram distance; ties broken toward smaller variance.
        min_dist = dist.min(axis=1, keepdims=True)
        chosen = np.argmin(np.where(dist == min_dist, cand_p, np.inf), axis=1)

        filled_a = anchor_a[chosen]
        filled_p = cand_p[rows, chosen]
        a_init[s], P_init[s] = filled_a[0], filled_p[0]
        a_obs[:, s], p_obs[:, s] = filled_a[1:], filled_p[1:]

    out = ssp_priors.copy()
    out["a_obs"] = (("time", "state"), a_obs)
    out["P_obs"] = (("time", "state"), p_obs)
    _write_init_moments(out, a_init, P_init, overwrite_init)
    return SspPrior(out)


def extend_states_prior_smoothed(
    ssp_priors: xr.Dataset | SspPrior,
    Q: float | np.ndarray,
    P0_diffuse: float = 1e8,
    *,
    overwrite_init: bool = False,
) -> SspPrior:
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

    The pass runs over ``T + 1`` steps rather than ``T``: one all-``inf``
    (predict-only) disclosure step is prepended, so the anchors are carried one
    step *before* the series to ``t = -1``, and row 0 of the smoother output is
    returned as the initial-state moments ``a0`` / ``P0`` over dims ``(state,)``
    while rows ``1:`` are the rectangle.  Unanchored states get ``a0 = 0``,
    ``P0 = inf``, the same "no information" encoding ``P_obs`` already uses.  The
    ``(time, state)`` rectangle keeps its shape — ``time`` stays length ``T`` —
    so nothing that indexes by step is disturbed.

    **Precedence.** Those derived moments are a *placeholder*, not an authority: a real
    initial-state prior the caller supplies (the ``base_ds`` block merged downstream) always
    wins.  Each of ``a0`` / ``P0`` is therefore written only when it is absent from
    ``ssp_priors``, independently of the other; pass ``overwrite_init=True`` to let the derived
    placeholder replace one that is already there.

    **When to use.** Prefer this for genuinely multi-anchor channels, where the
    nearest-anchor heuristic is a conservative, discontinuous approximation.  For a
    single-anchor channel the two agree to numerical noise (set by the diffuse-prior
    proxy ``P0_diffuse``), so the cheaper :func:`extend_states_prior_nearest` suffices.

    Parameters
    ----------
    ssp_priors : xr.Dataset or SspPrior
        Disclosed prior as produced by
        :func:`~bunobee.simulation.ssp.construct_states_prior`, carrying ``a_obs`` and
        ``P_obs`` over dims ``(time, state)``.  All other variables (``positivity``, any
        ``sigma_q`` block, and attrs) are passed through unchanged; an ``a0`` / ``P0``
        already present is **preserved** unless ``overwrite_init`` is set.
    Q : float or np.ndarray
        Per-state process variance ``σ_q²``.  A scalar is broadcast to every state; an
        array must have length ``n_states``.  Must be finite-or-``inf`` and non-negative.
        The filter itself takes a standard deviation, so internally this passes
        ``sqrt(Q)``.
    P0_diffuse : float, optional
        Large finite initial variance standing in for an uninformative prior, by default
        ``1e8`` (comfortable in float64).
    overwrite_init : bool, optional
        Whether the derived ``a0`` / ``P0`` replace ones already present on ``ssp_priors``,
        by default ``False`` (a real initial-state prior wins).

    Returns
    -------
    SspPrior
        Copy of ``ssp_priors`` whose ``a_obs`` / ``P_obs`` have every anchored state's
        undisclosed steps filled by the exact smoother marginal, plus ``a0`` / ``P0``
        over ``(state,)`` from the same marginal at ``t = -1`` wherever the input did
        not already carry them, promoted to an :class:`SspPrior` — the output satisfies
        the complete-prior contract by construction.  A ``P0 = inf`` column (an
        unanchored state) still needs handling before the a-space transforms can consume
        it.  Reach the plain dataset with :meth:`SspPrior.to_dataset` when an xarray API
        needs it.

    Raises
    ------
    ValueError
        If ``ssp_priors`` lacks the ``a_obs`` / ``P_obs`` disclosure block, is missing the
        ``time`` / ``state`` coordinates the :class:`SspPrior` contract requires, or ``Q``
        is negative, ``NaN``, or has the wrong length.
    """
    ssp_priors = _unwrap(ssp_priors)
    if "a_obs" not in ssp_priors or "P_obs" not in ssp_priors:
        raise ValueError("ssp_priors must contain `a_obs` and `P_obs` to extend the prior")

    a_obs = np.array(ssp_priors["a_obs"].values, dtype=float)
    p_obs = np.array(ssp_priors["P_obs"].values, dtype=float)
    n_steps, n_states = p_obs.shape

    q_vec = np.broadcast_to(np.asarray(Q, dtype=float), (n_states,)).astype(float, copy=True)
    if np.any(np.isnan(q_vec)) or np.any(q_vec < 0):
        raise ValueError("Q must be non-negative (finite or inf) for every state")

    sigma_q = jnp.sqrt(jnp.asarray(q_vec))

    # One prepended all-inf (predict-only) disclosure step extends the pass back to
    # t = -1, whose smoothed marginal is the initial-state prior (a0, P0).
    a_obs_ext = np.concatenate([np.zeros((1, n_states)), a_obs], axis=0)
    p_obs_ext = np.concatenate([np.full((1, n_states), np.inf), p_obs], axis=0)

    # Extension mode: Z = 0 -> zero Kalman gain -> no y-update, states stay independent
    # univariate random walks fed only through the anchor state-fusion path.
    a_diffuse = jnp.zeros(n_states)
    P_diffuse = jnp.full(n_states, P0_diffuse)
    Z = jnp.zeros((n_steps + 1, n_states))
    y = jnp.zeros(n_steps + 1)

    _, at, Pt, *_ = kalman_filter_1d(
        a0=a_diffuse,
        P0=P_diffuse,
        Z=Z,
        sigma_h=jnp.array(1.0),
        sigma_q=sigma_q,
        y=y,
        a_obs=jnp.asarray(a_obs_ext),
        P_obs=jnp.asarray(p_obs_ext),
    )
    at_smooth, Pt_smooth = kalman_rts_smoother_1d(at=at, Pt=Pt, sigma_q=sigma_q)

    # Row 0 is t = -1 (the initial-state moments); rows 1: are the (time, state) rectangle.
    a_ext = np.array(at_smooth, dtype=float)
    p_ext = np.array(Pt_smooth, dtype=float)
    a_init, P_init = a_ext[0], p_ext[0]
    a_out, p_out = a_ext[1:], p_ext[1:]

    # A state with no anchor comes back diffuse-but-finite from the diffuse P0; reset it
    # to fully-undisclosed (inf), matching extend_states_prior_nearest.
    unanchored = ~np.isfinite(p_obs).any(axis=0)
    a_out[:, unanchored] = a_obs[:, unanchored]
    p_out[:, unanchored] = p_obs[:, unanchored]
    a_init[unanchored] = 0.0
    P_init[unanchored] = np.inf

    out = ssp_priors.copy()
    out["a_obs"] = (("time", "state"), a_out)
    out["P_obs"] = (("time", "state"), p_out)
    _write_init_moments(out, a_init, P_init, overwrite_init)
    return SspPrior(out)
