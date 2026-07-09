from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from bunobee.models.ssp.prior import SspPrior, disclosed_idx


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
        :func:`~bunobee.models.ssp.posterior.posterior_to_xarray` whose
        variables have dims ``(chain, draw, ...)``.  In the dataset case the
        chain and draw axes are flattened internally before computing
        quantiles.  Must contain every entry in ``states_key``.
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
    ssp_prior: xr.Dataset | SspPrior,
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
    ssp_prior : xr.Dataset
        Prior dataset as produced by
        :func:`~bunobee.simulation.ssp.construct_states_prior`, i.e. with
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
        if ``ssp_prior`` lacks the ``a_obs`` / ``P_obs`` disclosure block.
    """
    valid = {"mean", "var", "both"}
    if quantity not in valid:
        raise ValueError(f"quantity must be one of {sorted(valid)}, got {quantity!r}")
    if isinstance(ssp_prior, SspPrior):
        ssp_prior = ssp_prior.dataset
    if "a_obs" not in ssp_prior or "P_obs" not in ssp_prior:
        raise ValueError("ssp_prior must contain `a_obs` and `P_obs` to plot the prior")

    a_obs = np.asarray(ssp_prior["a_obs"].values, dtype=float)
    p_obs = np.asarray(ssp_prior["P_obs"].values, dtype=float)
    state_labels = [str(s) for s in np.asarray(ssp_prior["state"].values)]

    if dates is None:
        dates = np.asarray(ssp_prior["time"].values) if "time" in ssp_prior.coords else np.arange(a_obs.shape[0])
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


def _gaussian_pdf(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Evaluate the 1-D Gaussian density ``N(y; mean, var)`` elementwise.

    Parameters
    ----------
    y : np.ndarray
        State value(s) at which to evaluate the density; broadcast against
        ``mean`` / ``var``.
    mean : np.ndarray
        Marginal mean(s).
    var : np.ndarray
        Marginal variance(s); must be finite and positive to yield a proper
        density (``inf`` variance yields ``0``, ``0`` variance yields ``inf``).

    Returns
    -------
    np.ndarray
        Density values, broadcast shape of the inputs.
    """
    return np.exp(-0.5 * (y - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)


def _read_states_prior(prior: xr.Dataset | SspPrior) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract ``a_obs`` / ``P_obs`` and state labels from a states prior.

    Parameters
    ----------
    prior : xr.Dataset or SspPrior
        Prior carrying ``a_obs`` / ``P_obs`` over dims ``(time, state)``.

    Returns
    -------
    a_obs : np.ndarray, shape (n_steps, n_states)
    p_obs : np.ndarray, shape (n_steps, n_states)
    state_labels : list[str]

    Raises
    ------
    ValueError
        If ``prior`` lacks the ``a_obs`` / ``P_obs`` disclosure block.
    """
    if isinstance(prior, SspPrior):
        prior = prior.dataset
    if "a_obs" not in prior or "P_obs" not in prior:
        raise ValueError("prior must contain `a_obs` and `P_obs` to plot the prior density")
    a_obs = np.asarray(prior["a_obs"].values, dtype=float)
    p_obs = np.asarray(prior["P_obs"].values, dtype=float)
    if "state" in prior.coords:
        state_labels = [str(s) for s in np.asarray(prior["state"].values)]
    else:
        state_labels = [f"s{i}" for i in range(a_obs.shape[1])]
    return a_obs, p_obs, state_labels


def plot_prior_density(
    prior: xr.Dataset | SspPrior,
    *,
    anchors: xr.Dataset | SspPrior | None = None,
    n_grid: int = 400,
    pad: float = 3.5,
    cmap: str = "magma",
    mean_color: str = "white",
    anchor_color: str = "cyan",
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    r"""Draw the per-step Gaussian marginal of a states prior as density panels.

    Companion to :func:`plot_prior_heatmap`.  For each state, the prior's
    per-step marginal ``N(a_obs[t], P_obs[t])`` is evaluated on a shared value
    grid and rendered as a single ``time x value`` heatmap panel (one subplot
    per state, brighter = higher density), with the smoothed mean threaded
    across the top.  This promotes the ad-hoc ``density_field`` view from the
    ``ssp_extend_prior_multi_anchor`` prototype notebook into the package.

    Undisclosed steps (``P_obs`` is ``inf``) are masked, so a raw, un-extended
    prior -- which carries ``inf`` variance at most steps -- still renders
    gracefully: only the finite-variance steps contribute to the value grid and
    the density block, and undisclosed columns are left blank.

    Parameters
    ----------
    prior : xr.Dataset or SspPrior
        States prior carrying ``a_obs`` and ``P_obs`` over dims
        ``(time, state)``.  Typically the extended / smoothed prior (every step
        finite) whose full marginal is the quantity being visualised, but a raw
        disclosed prior renders too.
    anchors : xr.Dataset or SspPrior or None, optional
        Separate prior whose disclosures supply the raw anchor markers.  Needed
        when ``prior`` is the already-extended prior (every step finite), so its
        own disclosures no longer isolate the raw anchors.  When ``None`` the
        anchors are read from ``prior`` itself via
        :func:`~bunobee.models.ssp.prior.disclosed_idx`.
    n_grid : int, optional
        Number of points on the shared value grid, by default 400.
    pad : float, optional
        Grid half-width in standard deviations beyond the mean envelope, by
        default 3.5 (captures ~99.95% of every bell's mass).
    cmap : str, optional
        Matplotlib colormap for the density field, by default ``"magma"``.
    mean_color : str, optional
        Colour of the overlaid mean line, by default ``"white"``.
    anchor_color : str, optional
        Colour of the raw-anchor markers, by default ``"cyan"``.
    title : str or None, optional
        Figure suptitle.  Auto-generated when ``None``.

    Returns
    -------
    fig : plt.Figure
    axes : np.ndarray
        Flattened array of the panel ``Axes`` (one per state).

    Raises
    ------
    ValueError
        If ``prior`` lacks the ``a_obs`` / ``P_obs`` disclosure block.
    """
    a_obs, p_obs, state_labels = _read_states_prior(prior)
    n_steps, n_states = a_obs.shape
    x = np.arange(n_steps)

    # Anchor source: an explicit `anchors` prior, else the prior's own disclosures.
    anchor_a, anchor_p, _ = _read_states_prior(anchors) if anchors is not None else (a_obs, p_obs, state_labels)
    anchor_t = disclosed_idx(anchors if anchors is not None else prior)

    fig, axes = plt.subplots(1, n_states, figsize=(7.0 * n_states, 4.0), squeeze=False)
    axes = axes.flatten()

    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad(alpha=0.0)

    for i, (ax, label) in enumerate(zip(axes, state_labels)):
        a_i = a_obs[:, i]
        p_i = p_obs[:, i]
        finite = np.isfinite(p_i) & np.isfinite(a_i) & (p_i > 0)

        # Build the shared value grid from disclosed steps only (both this prior
        # and, if given, the anchor source) so `inf`-variance steps never blow up.
        grid_a = [a_i[finite]]
        grid_sd = [np.sqrt(p_i[finite])]
        if anchors is not None:
            a_src, p_src = anchor_a[:, i], anchor_p[:, i]
            src_finite = np.isfinite(p_src) & np.isfinite(a_src) & (p_src > 0)
            grid_a.append(a_src[src_finite])
            grid_sd.append(np.sqrt(p_src[src_finite]))
        centres = np.concatenate(grid_a)
        spreads = np.concatenate(grid_sd)
        if centres.size == 0:
            # Nothing disclosed for this state: draw an empty panel gracefully.
            y = np.linspace(-1.0, 1.0, n_grid)
        else:
            y = np.linspace((centres - pad * spreads).min(), (centres + pad * spreads).max(), n_grid)

        # Density block (n_grid, n_steps); mask undisclosed columns so they stay blank.
        safe_var = np.where(finite, p_i, 1.0)  # avoid 1/inf and 1/0 in the evaluation
        dens = _gaussian_pdf(y[:, None], a_i[None, :], safe_var[None, :])
        dens = np.ma.masked_where(np.broadcast_to(~finite[None, :], dens.shape), dens)

        im = ax.imshow(
            dens,
            origin="lower",
            aspect="auto",
            extent=[0, max(n_steps - 1, 1), y[0], y[-1]],
            cmap=colormap,
        )
        # Mean line, broken across undisclosed steps.
        ax.plot(x, np.where(finite, a_i, np.nan), color=mean_color, lw=1.0, label="mean")

        m = np.isin(x, anchor_t) & np.isfinite(anchor_p[:, i]) & np.isfinite(anchor_a[:, i])
        if m.any():
            ax.scatter(
                x[m],
                anchor_a[m, i],
                s=30,
                marker="x",
                color=anchor_color,
                zorder=3,
                label="raw anchor",
            )

        ax.set_title(f"{label}: prior density", fontsize=9)
        ax.set_xlabel("time step")
        ax.legend(fontsize=7, loc="upper right")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="density")

    axes[0].set_ylabel("state value")

    if title is None:
        title = "per-state prior density"
    fig.suptitle(title, y=1.02)
    plt.tight_layout()

    return fig, axes
