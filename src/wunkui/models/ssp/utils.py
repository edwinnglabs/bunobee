from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def a_to_lam(
    arr: np.ndarray,
    exponent: float,
    positivity_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a-space values to λ-space for positivity states.

    Parameters
    ----------
    arr : np.ndarray
        Array in a-space, shape ``(..., n_states)`` — e.g. ``a_obs_loc``.
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
    states_key: str = "at_smooth",
    coefs_df: pd.DataFrame | None = None,
    obs_idx: np.ndarray | None = None,
    a_obs_loc: np.ndarray | None = None,
    a_obs_var: np.ndarray | None = None,
    title: str | None = None,
    n_cols: int = 4,
    ci: tuple[float, float, float] = (0.05, 0.5, 0.95),
) -> tuple[plt.Figure, np.ndarray]:
    """Plot posterior quantile ribbons for latent states across MCMC samples.

    Works for filtered states (``"at"``), smoothed states (``"at_smooth"``), or
    EKF multiplicative intensities (``"lam"``); just pass the appropriate
    ``states_key``.

    Parameters
    ----------
    posterior : dict[str, np.ndarray]
        Sample dict from ``mcmc.get_samples()``.  Must contain ``states_key``
        with shape ``(n_samples, T, n_states)``.
    dates : np.ndarray
        Length-T array of date values used as the x-axis.
    state_labels : list[str]
        Human-readable name for each state dimension (length ``n_states``).
    states_key : str, optional
        Key in ``posterior`` to visualise, by default ``"at_smooth"``.
    coefs_df : pd.DataFrame or None, optional
        DataFrame with columns ``["regressor", "coef"]`` providing ground-truth
        reference lines.  Skipped when ``None``.
    obs_idx : np.ndarray or None, optional
        Integer indices into ``dates`` where disclosures occurred.  Disclosure
        scatter markers are omitted when ``None``.
    a_obs_loc : np.ndarray or None, optional
        Disclosed state means, shape ``(T, n_states)``.  Required together with
        ``obs_idx`` and ``a_obs_var`` to draw scatter markers.
    a_obs_var : np.ndarray or None, optional
        Disclosed state variances, shape ``(T, n_states)``.  ``isfinite`` is
        used as the active-disclosure mask per state.
    title : str or None, optional
        Figure suptitle.  Auto-generated from ``states_key`` when ``None``.
    n_cols : int, optional
        Number of subplot columns, by default 4.
    ci : tuple[float, float, float], optional
        Quantile triple ``(lo, mid, hi)``, by default ``(0.05, 0.5, 0.95)``.

    Returns
    -------
    fig : plt.Figure
    axes : np.ndarray
        Flattened array of all ``Axes`` objects (including hidden ones).
    """
    states = np.asarray(posterior[states_key])
    lo, mid, hi = np.quantile(states, ci, axis=0)

    coefs_lookup = coefs_df.set_index("regressor")["coef"] if coefs_df is not None else None

    has_disclosure = (
        obs_idx is not None
        and len(obs_idx) > 0
        and a_obs_loc is not None
        and a_obs_var is not None
    )
    obs_dates = dates[obs_idx] if has_disclosure else []

    n_states = len(state_labels)
    n_rows = math.ceil(n_states / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.2 * n_rows), sharex=False)
    axes = np.atleast_1d(axes).flatten()

    for i, (ax, label) in enumerate(zip(axes, state_labels)):
        ax.plot(dates, mid[:, i], color="darkgreen", linewidth=0.9, label="median")
        ax.fill_between(dates, lo[:, i], hi[:, i], alpha=0.25, color="darkgreen", label=f"{int(round((ci[2]-ci[0])*100))}% CI")

        if i > 0 and coefs_lookup is not None and label in coefs_lookup.index:
            ax.axhline(coefs_lookup[label], color="grey", linestyle=":", linewidth=1.0, label="true coef")

        if has_disclosure:
            anchor_mask = np.isfinite(np.asarray(a_obs_var)[obs_idx, i])
            if anchor_mask.any():
                ax.scatter(
                    obs_dates[anchor_mask],
                    np.asarray(a_obs_loc)[obs_idx][anchor_mask, i],
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
        title = states_key
    fig.suptitle(title, y=1.01)
    plt.tight_layout()

    return fig, axes
