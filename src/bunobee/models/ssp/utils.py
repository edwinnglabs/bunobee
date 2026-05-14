from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import numpy as jnp


def construct_states_prior(
    n_steps: int,
    n_states: int,
    true_states: jnp.ndarray,
    regressors: list,
    n_periods: int = 3,
    n_points: int = 7,
    seed: int = 42,
    obs_scale: float = 0.1,
) -> dict[str, np.ndarray]:
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

    Returns
    -------
    a_obs : jnp.ndarray, shape (n_steps, n_states)
        Disclosed state means; zero where not disclosed.
    P_obs : jnp.ndarray, shape (n_steps, n_states)
        Disclosed state variances; inf where no information is provided
        (intercept column always inf, undisclosed timesteps always inf).
    obs_idx : np.ndarray
        Sorted array of all disclosed time indices (for plotting).
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
    var_row = jnp.array([jnp.inf] + [obs_scale ** 2] * len(regressors))
    P_obs = P_obs.at[obs_idx].set(var_row)

    return {"a_obs": a_obs, "P_obs": P_obs, "obs_idx": obs_idx}


def a_to_lam(
    arr: np.ndarray,
    exponent: float,
    positivity_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a-space values to λ-space for positivity states.

    Parameters
    ----------
    arr : np.ndarray
        Array in a-space, shape ``(..., n_states)`` — e.g. ``a_obs``.
    exponent : float
        EKF nonlinearity exponent: ``λ = exp(exponent · a)``.
    positivity_idx : np.ndarray or None, optional
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
    mask = (
        np.ones(n_states, dtype=bool)
        if positivity_idx is None
        else np.asarray(positivity_idx, dtype=bool)
    )
    out[..., mask] = np.exp(exponent * out[..., mask])
    return out


def lam_to_a(
    arr: np.ndarray,
    exponent: float,
    positivity_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Convert λ-space values to a-space for positivity states.

    Parameters
    ----------
    arr : np.ndarray
        Array in λ-space, shape ``(..., n_states)``.  Positivity columns must
        be strictly positive; ``log`` is applied element-wise.
    exponent : float
        EKF nonlinearity exponent: ``a = log(λ) / exponent``.
    positivity_idx : np.ndarray or None, optional
        Boolean mask of length ``n_states``.  ``None`` treats all as positivity.

    Returns
    -------
    np.ndarray
        Same shape.  Positivity columns transformed via ``log(λ) / exponent``;
        linear columns passed through unchanged.
    """
    out = np.array(arr, dtype=float)
    n_states = out.shape[-1]
    mask = (
        np.ones(n_states, dtype=bool)
        if positivity_idx is None
        else np.asarray(positivity_idx, dtype=bool)
    )
    out[..., mask] = np.log(out[..., mask]) / exponent
    return out


def plot_states(
    posterior: dict[str, np.ndarray],
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
    colors: Sequence[str] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot posterior quantile ribbons for latent states across MCMC samples.

    Works for filtered states (``"at"``), smoothed states (``"at_smooth"``), or
    EKF multiplicative intensities (``"lam"``).  Pass a single key for one
    ribbon, or a list of keys to overlay multiple posteriors on the same axes
    (e.g. ``["at", "at_smooth"]`` to compare filtered vs. smoothed).

    Parameters
    ----------
    posterior : dict[str, np.ndarray]
        Sample dict from ``mcmc.get_samples()``.  Must contain every entry in
        ``states_key`` with shape ``(n_samples, T, n_states)``.
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
    colors : sequence of str or None, optional
        Per-overlay colours, one per entry in ``states_key``.  Defaults to
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

    default_colors = ["darkgreen", *plt.get_cmap("tab10").colors]
    palette = list(colors) if colors is not None else default_colors
    if len(palette) < len(keys):
        raise ValueError(
            f"need at least {len(keys)} colours for {len(keys)} overlays, got {len(palette)}"
        )

    quantiles = [np.quantile(np.asarray(posterior[k]), ci, axis=0) for k in keys]
    ci_pct = int(round((ci[2] - ci[0]) * 100))

    coefs_lookup = coefs_df.set_index("regressor")["coef"] if coefs_df is not None else None

    has_disclosure = (
        obs_idx is not None
        and len(obs_idx) > 0
        and a_obs is not None
        and P_obs is not None
    )
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
            ax.fill_between(
                dates, lo[:, i], hi[:, i], alpha=0.25, color=color, label=ribbon_label
            )

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
