from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from jax import numpy as jnp

from bunobee.models.ssp.transforms import validate_prior


def construct_states_prior(
    n_steps: int,
    n_states: int,
    true_states: jnp.ndarray,
    regressors: list,
    n_periods: int = 3,
    n_points: int = 7,
    seed: int = 42,
    obs_scale: float = 0.1,
    positivity: np.ndarray | None = None,
) -> xr.Dataset:
    """Construct a_obs and P_obs by disclosing the ground-truth latent
    state over n_periods randomly drawn windows of n_points consecutive steps.

    Parameters
    ----------
    n_steps : int
        Total number of time steps.
    n_states : int
        Number of latent states (intercept + regressors).
    true_states : jnp.ndarray, shape (n_states,)
        Ground-truth state vector; intercept entry is ignored (var stays inf).
    regressors : list[str]
        Regressor names; determines which state indices get a finite variance.
    n_periods : int
        Number of disclosure windows to draw.
    n_points : int
        Number of consecutive steps per window.
    seed : int
        RNG seed for reproducibility.
    obs_scale : float
        Standard deviation expressing confidence in the disclosed state.
        Smaller → tighter prior; larger → more diffuse.
    positivity : np.ndarray or None, optional
        Boolean mask of length ``n_states`` indicating which states are
        positivity-constrained.  Defaults to ``[False, True, …]`` — intercept
        is linear, all regressors are positivity states.

    Returns
    -------
    xr.Dataset
        Variables ``a_obs`` and ``P_obs`` with dims ``(time, state)`` and
        ``positivity`` with dim ``state``.  Coords: ``time`` (0…n_steps-1) and
        ``state`` (["intercept", *regressors]).  Disclosure time indices are
        not stored; derive them with :func:`disclosed_idx`.
    """
    rng = np.random.default_rng(seed)
    # sample n_periods window starts; ensure each window fits within n_steps
    starts = rng.choice(n_steps - n_points + 1, size=n_periods, replace=False)
    obs_idx = np.unique(np.concatenate([np.arange(s, s + n_points) for s in starts]))

    a_obs = jnp.zeros((n_steps, n_states))
    # default inf = zero precision = no information; pure filter carries through
    P_obs = jnp.full((n_steps, n_states), jnp.inf)

    a_obs = a_obs.at[obs_idx].set(true_states)
    # intercept has no priors so its variance stays inf
    var_row = jnp.array([jnp.inf] + [obs_scale**2] * len(regressors))
    P_obs = P_obs.at[obs_idx].set(var_row)

    if positivity is None:
        positivity = np.array([False] + [True] * len(regressors))
    else:
        positivity = np.asarray(positivity, dtype=bool)

    state_labels = ["intercept", *regressors]
    ds = xr.Dataset(
        {
            "a_obs": (("time", "state"), np.asarray(a_obs)),
            "P_obs": (("time", "state"), np.asarray(P_obs)),
            "positivity": (("state",), positivity),
        },
        coords={
            "time": np.arange(n_steps),
            "state": state_labels,
        },
    )
    # Fail fast on a malformed disclosure block; a0/P0 are supplied downstream.
    validate_prior(ds, require_init=False)
    return ds


def disclosed_idx(ssp_priors: xr.Dataset) -> np.ndarray:
    """Return the time indices where the prior discloses state information.

    Replaces the previously stored ``obs_idx`` variable, which was redundant
    with ``P_obs``: a disclosure exists at any timestep with at least one
    finite-variance state, since undisclosed timesteps carry ``inf`` variance
    (zero precision, a pure filter carry-through).

    Parameters
    ----------
    ssp_priors : xr.Dataset
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
    if "P_obs" not in ssp_priors:
        raise KeyError("ssp_priors has no `P_obs`; cannot derive disclosure indices")
    p_obs = np.asarray(ssp_priors["P_obs"].values)
    return np.where(np.isfinite(p_obs).any(axis=1))[0]


def extend_states_prior(
    ssp_priors: xr.Dataset,
    Q: float | np.ndarray,
) -> xr.Dataset:
    r"""Extend disclosed anchors along the driftless random-walk marginal.

    :func:`construct_states_prior` discloses anchors at a handful of timesteps
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

    Parameters
    ----------
    ssp_priors : xr.Dataset
        Disclosed prior as produced by :func:`construct_states_prior`, carrying
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


def a_to_lam(
    arr: np.ndarray,
    exponent: float,
    positivity: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a-space values to λ-space for positivity states.

    Parameters
    ----------
    arr : np.ndarray
        Array in a-space, shape ``(..., n_states)`` — e.g. ``a_obs``.
    exponent : float
        EKF nonlinearity exponent: ``λ = exp(exponent · a)``.
    positivity : np.ndarray or None, optional
        Boolean mask of length ``n_states``.  ``True`` = positivity state.
        ``None`` treats every state as positivity.

    Returns
    -------
    np.ndarray
        Same shape.  Positivity columns transformed via ``exp(exponent · a)``;
        linear columns passed through unchanged.
    """
    out = np.array(arr, dtype=float)
    n_states = out.shape[-1]
    mask = np.ones(n_states, dtype=bool) if positivity is None else np.asarray(positivity, dtype=bool)
    out[..., mask] = np.exp(exponent * out[..., mask])
    return out


def lam_to_a(
    arr: np.ndarray,
    exponent: float,
    positivity: np.ndarray | None = None,
) -> np.ndarray:
    """Convert λ-space values to a-space for positivity states.

    Parameters
    ----------
    arr : np.ndarray
        Array in λ-space, shape ``(..., n_states)``.  Positivity columns must
        be strictly positive; ``log`` is applied element-wise.
    exponent : float
        EKF nonlinearity exponent: ``a = log(λ) / exponent``.
    positivity : np.ndarray or None, optional
        Boolean mask of length ``n_states``.  ``None`` treats all as positivity.

    Returns
    -------
    np.ndarray
        Same shape.  Positivity columns transformed via ``log(λ) / exponent``;
        linear columns passed through unchanged.
    """
    out = np.array(arr, dtype=float)
    n_states = out.shape[-1]
    mask = np.ones(n_states, dtype=bool) if positivity is None else np.asarray(positivity, dtype=bool)
    out[..., mask] = np.log(out[..., mask]) / exponent
    return out


def posterior_to_xarray(
    posterior: Mapping[str, np.ndarray],
    *,
    dims: Mapping[str, Sequence[str]] | None = None,
    coords: Mapping[str, Sequence] | None = None,
    drop: Sequence[str] | None = None,
    keep: Sequence[str] | None = None,
) -> xr.Dataset:
    """Convert a chain-grouped numpyro posterior dict to an ``xarray.Dataset``.

    Each value of ``posterior`` must have leading ``(chain, draw, ...)`` axes,
    matching the output of ``mcmc.get_samples(group_by_chain=True)``. The
    resulting dataset is suitable for wrapping in
    ``arviz.InferenceData(posterior=ds)`` for downstream diagnostics
    (``az.summary``, ``az.plot_trace``, ``az.plot_rank``, ...).

    Parameters
    ----------
    posterior : mapping[str, np.ndarray]
        Mapping from site name to draws of shape ``(n_chains, n_draws, *event)``.
    dims : mapping[str, sequence of str], optional
        Names for each variable's event axes (everything past ``chain`` and
        ``draw``). Variables omitted here get auto-named axes
        ``"<name>_dim_<i>"``.
    coords : mapping[str, sequence], optional
        Coordinate values keyed by dimension name. Dimensions without an entry
        fall back to a plain integer range.
    drop : sequence of str, optional
        Site names to omit from the dataset. Mutually exclusive with ``keep``.
    keep : sequence of str, optional
        Site names to retain; everything else is dropped. Mutually exclusive
        with ``drop``.

    Returns
    -------
    xarray.Dataset
        One ``DataArray`` per kept variable with dims
        ``(chain, draw, *event_dims)``.
    """
    if drop is not None and keep is not None:
        raise ValueError("pass at most one of `drop` or `keep`, not both")

    drop_set = set(drop or ())
    keep_set = set(keep) if keep is not None else None
    items = {
        k: np.asarray(v) for k, v in posterior.items() if k not in drop_set and (keep_set is None or k in keep_set)
    }
    if not items:
        raise ValueError("no variables left to convert after applying drop/keep")

    dims = dict(dims or {})
    user_coords = dict(coords or {})

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    out_coords: dict[str, np.ndarray] = {}

    for name, arr in items.items():
        if arr.ndim < 2:
            raise ValueError(
                f"variable {name!r} has shape {arr.shape}; expected leading "
                "(chain, draw) axes from group_by_chain=True samples"
            )
        event_shape = arr.shape[2:]
        event_dims = list(dims.get(name) or [f"{name}_dim_{i}" for i in range(len(event_shape))])
        if len(event_dims) != len(event_shape):
            raise ValueError(f"dims for {name!r} has length {len(event_dims)} but event " f"shape is {event_shape}")

        for d, size in zip(event_dims, event_shape):
            if d in out_coords:
                if len(out_coords[d]) != size:
                    raise ValueError(
                        f"dim {d!r} reused with inconsistent size: " f"{len(out_coords[d])} vs {size} (from {name!r})"
                    )
                continue
            if d in user_coords:
                values = np.asarray(user_coords[d])
                if values.shape != (size,):
                    raise ValueError(f"coord {d!r} has shape {values.shape} but {name!r} " f"expects size {size}")
                out_coords[d] = values
            else:
                out_coords[d] = np.arange(size)

        data_vars[name] = (("chain", "draw", *event_dims), arr)

    sample_arr = next(iter(items.values()))
    out_coords.setdefault("chain", np.arange(sample_arr.shape[0]))
    out_coords.setdefault("draw", np.arange(sample_arr.shape[1]))

    return xr.Dataset(data_vars=data_vars, coords=out_coords)


def plot_states(
    posterior: Mapping[str, np.ndarray] | xr.Dataset,
    dates: np.ndarray,
    state_labels: list[str],
    *,
    states_key: str | Sequence[str] = "at_smooth",
    coefs_df: pd.DataFrame | None = None,
    obs_idx: np.ndarray | None = None,
    a_obs: np.ndarray | None = None,
    P_obs: np.ndarray | None = None,
    title: str | None = None,
    n_cols: int = 4,
    ci: tuple[float, float, float] = (0.05, 0.5, 0.95),
    colors: dict[str, str] | Sequence[str] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot posterior quantile ribbons for latent states across MCMC samples.

    Works for filtered states (``"at"``), smoothed states (``"at_smooth"``), or
    EKF multiplicative intensities (``"lam"``).  Pass a single key for one
    ribbon, or a list of keys to overlay multiple posteriors on the same axes
    (e.g. ``["at", "at_smooth"]`` to compare filtered vs. smoothed).

    Parameters
    ----------
    posterior : mapping[str, np.ndarray] or xarray.Dataset
        Either a flat sample dict from ``mcmc.get_samples()`` (each entry shape
        ``(n_samples, T, n_states)``) or an ``xarray.Dataset`` produced by
        :func:`posterior_to_xarray` whose variables have dims
        ``(chain, draw, ...)``.  In the dataset case the chain and draw axes
        are flattened internally before computing quantiles.  Must contain
        every entry in ``states_key``.
    dates : np.ndarray
        Length-T array of date values used as the x-axis.
    state_labels : list[str]
        Human-readable name for each state dimension (length ``n_states``).
    states_key : str or sequence of str, optional
        Key(s) in ``posterior`` to visualise, by default ``"at_smooth"``.  When
        a sequence is given, each key is overlaid with its own colour.
    coefs_df : pd.DataFrame or None, optional
        DataFrame with columns ``["regressor", "coef"]`` providing ground-truth
        reference lines.  Skipped when ``None``.
    obs_idx : np.ndarray or None, optional
        Integer indices into ``dates`` where disclosures occurred.  Disclosure
        scatter markers are omitted when ``None``.
    a_obs : np.ndarray or None, optional
        Disclosed state means, shape ``(T, n_states)``.  Required together with
        ``obs_idx`` and ``P_obs`` to draw scatter markers.
    P_obs : np.ndarray or None, optional
        Disclosed state variances, shape ``(T, n_states)``.  ``isfinite`` is
        used as the active-disclosure mask per state.
    title : str or None, optional
        Figure suptitle.  Auto-generated from ``states_key`` when ``None``.
    n_cols : int, optional
        Number of subplot columns, by default 4.
    ci : tuple[float, float, float], optional
        Quantile triple ``(lo, mid, hi)``, by default ``(0.05, 0.5, 0.95)``.
    colors : dict[str, str] or sequence of str or None, optional
        Per-overlay colours.  Pass a dict mapping each key in ``states_key``
        to a colour (e.g. ``{"at": "steelblue", "at_smooth": "darkgreen"}``),
        or a plain sequence ordered the same as ``states_key``.  Keys missing
        from the dict fall back to the default palette.  Defaults to
        ``matplotlib``'s ``tab10`` cycle, with ``"darkgreen"`` as the first
        colour to preserve the original single-overlay appearance.

    Returns
    -------
    fig : plt.Figure
    axes : np.ndarray
        Flattened array of all ``Axes`` objects (including hidden ones).
    """
    keys = [states_key] if isinstance(states_key, str) else list(states_key)
    if not keys:
        raise ValueError("states_key must contain at least one key")

    if isinstance(posterior, xr.Dataset):
        posterior = {k: posterior[k].stack(sample=("chain", "draw")).transpose("sample", ...).values for k in keys}

    default_colors = ["darkgreen", *plt.get_cmap("tab10").colors]
    if colors is None:
        palette = default_colors[: len(keys)]
    elif isinstance(colors, dict):
        palette = [colors.get(k, default_colors[i]) for i, k in enumerate(keys)]
    else:
        palette = list(colors)
        if len(palette) < len(keys):
            raise ValueError(f"need at least {len(keys)} colours for {len(keys)} overlays, got {len(palette)}")

    quantiles = [np.quantile(np.asarray(posterior[k]), ci, axis=0) for k in keys]
    ci_pct = int(round((ci[2] - ci[0]) * 100))

    coefs_lookup = coefs_df.set_index("regressor")["coef"] if coefs_df is not None else None

    has_disclosure = obs_idx is not None and len(obs_idx) > 0 and a_obs is not None and P_obs is not None
    obs_dates = dates[obs_idx] if has_disclosure else []

    n_states = len(state_labels)
    n_rows = math.ceil(n_states / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.2 * n_rows), sharex=False)
    axes = np.atleast_1d(axes).flatten()

    single = len(keys) == 1
    for i, (ax, label) in enumerate(zip(axes, state_labels)):
        for key, (lo, mid, hi), color in zip(keys, quantiles, palette):
            median_label = "median" if single else f"{key} median"
            ribbon_label = f"{ci_pct}% CI" if single else f"{key} {ci_pct}% CI"
            ax.plot(dates, mid[:, i], color=color, linewidth=0.9, label=median_label)
            ax.fill_between(dates, lo[:, i], hi[:, i], alpha=0.25, color=color, label=ribbon_label)

        if i > 0 and coefs_lookup is not None and label in coefs_lookup.index:
            ax.axhline(coefs_lookup[label], color="grey", linestyle=":", linewidth=1.0, label="true coef")

        if has_disclosure:
            anchor_mask = np.isfinite(np.asarray(P_obs)[obs_idx, i])
            if anchor_mask.any():
                ax.scatter(
                    obs_dates[anchor_mask],
                    np.asarray(a_obs)[obs_idx][anchor_mask, i],
                    s=14,
                    color="crimson",
                    marker="x",
                    label="prior anchor",
                    zorder=3,
                )

        ax.set_title(label, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", labelsize=7, rotation=30)

    for ax in axes[n_states:]:
        ax.set_visible(False)

    axes[0].legend(fontsize=7)

    if title is None:
        title = keys[0] if single else " vs ".join(keys)
    fig.suptitle(title, y=1.01)
    plt.tight_layout()

    return fig, axes


def plot_prior_heatmap(
    ssp_priors: xr.Dataset,
    *,
    quantity: str = "mean",
    dates: np.ndarray | None = None,
    cmap: str = "viridis",
    undisclosed_color: str = "0.9",
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Visualise a time-point prior as a state-by-time heatmap.

    Draws the disclosed prior with states on the y-axis and time on the x-axis.
    Cells with no disclosure (``P_obs`` is ``inf``) are masked and rendered in
    ``undisclosed_color`` so the sparse disclosure windows stand out.

    Parameters
    ----------
    ssp_priors : xr.Dataset
        Prior dataset as produced by :func:`construct_states_prior`, i.e. with
        variables ``a_obs`` and ``P_obs`` over dims ``(time, state)`` and a
        ``state`` coordinate giving the state labels.
    quantity : {"mean", "var", "both"}, optional
        Which field(s) to draw, by default ``"mean"``.  ``"mean"`` shows the
        disclosed state means ``a_obs``; ``"var"`` shows the disclosed variances
        ``P_obs``; ``"both"`` stacks the two panels vertically.
    dates : np.ndarray or None, optional
        Length-T array of x-axis tick values.  Falls back to the dataset's
        ``time`` coordinate (or an integer range) when ``None``.
    cmap : str, optional
        Matplotlib colormap name for the disclosed values, by default
        ``"viridis"``.
    undisclosed_color : str, optional
        Matplotlib color for masked (undisclosed) cells, by default ``"0.9"``.
    title : str or None, optional
        Figure suptitle.  Auto-generated from ``quantity`` when ``None``.

    Returns
    -------
    fig : plt.Figure
    axes : np.ndarray
        Flattened array of the panel ``Axes`` (one per drawn quantity).

    Raises
    ------
    ValueError
        If ``quantity`` is not one of ``"mean"``, ``"var"``, or ``"both"``, or
        if ``ssp_priors`` lacks the ``a_obs`` / ``P_obs`` disclosure block.
    """
    valid = {"mean", "var", "both"}
    if quantity not in valid:
        raise ValueError(f"quantity must be one of {sorted(valid)}, got {quantity!r}")
    if "a_obs" not in ssp_priors or "P_obs" not in ssp_priors:
        raise ValueError("ssp_priors must contain `a_obs` and `P_obs` to plot the prior")

    a_obs = np.asarray(ssp_priors["a_obs"].values, dtype=float)
    p_obs = np.asarray(ssp_priors["P_obs"].values, dtype=float)
    state_labels = [str(s) for s in np.asarray(ssp_priors["state"].values)]

    if dates is None:
        dates = np.asarray(ssp_priors["time"].values) if "time" in ssp_priors.coords else np.arange(a_obs.shape[0])
    dates = np.asarray(dates)

    # disclosed wherever the variance is finite; everything else is masked
    disclosed = np.isfinite(p_obs)
    fields = {"mean": (a_obs, "a_obs (mean)"), "var": (p_obs, "P_obs (variance)")}
    panels = ["mean", "var"] if quantity == "both" else [quantity]

    n_states = len(state_labels)
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 1.6 + 0.5 * n_states * len(panels)), squeeze=False)
    axes = axes.flatten()

    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad(undisclosed_color)
    for ax, key in zip(axes, panels):
        values, label = fields[key]
        # transpose to (state, time); mask undisclosed and any residual non-finite cells
        grid = np.ma.masked_where(~disclosed.T | ~np.isfinite(values.T), values.T)
        mesh = ax.pcolormesh(dates, np.arange(n_states), grid, cmap=colormap, shading="nearest")
        fig.colorbar(mesh, ax=ax, label=label, pad=0.01)
        ax.set_yticks(np.arange(n_states))
        ax.set_yticklabels(state_labels, fontsize=8)
        ax.set_ylabel("state")
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=7, rotation=30)
    axes[-1].set_xlabel("time")

    if title is None:
        title = "time-point prior (mean & variance)" if quantity == "both" else "time-point prior"
    fig.suptitle(title, y=1.01)
    plt.tight_layout()

    return fig, axes
