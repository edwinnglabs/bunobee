from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from bunobee.models.ssp.prior import SspPrior


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
