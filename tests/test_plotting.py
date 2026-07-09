from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede any pyplot import

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from bunobee.models.ssp import plot_prior_density
from bunobee.models.ssp.prior import extend_states_prior_smoothed


def _prior(a_obs: np.ndarray, p_obs: np.ndarray) -> xr.Dataset:
    """Build a minimal disclosure-only states prior for plotting tests.

    Parameters
    ----------
    a_obs : np.ndarray, shape (n_steps, n_states)
        Disclosed state means.
    p_obs : np.ndarray, shape (n_steps, n_states)
        Disclosed state variances; ``inf`` marks undisclosed steps.

    Returns
    -------
    xr.Dataset
        Prior with ``a_obs`` / ``P_obs`` over ``(time, state)`` and a
        ``positivity`` mask.
    """
    a_obs = np.asarray(a_obs, dtype=float)
    p_obs = np.asarray(p_obs, dtype=float)
    n_steps, n_states = p_obs.shape
    return xr.Dataset(
        {
            "a_obs": (("time", "state"), a_obs),
            "P_obs": (("time", "state"), p_obs),
            "positivity": (("state",), np.zeros(n_states, dtype=bool)),
        },
        coords={"time": np.arange(n_steps), "state": [f"s{i}" for i in range(n_states)]},
    )


def _multi_state_prior(n_steps: int = 12) -> xr.Dataset:
    """A 2-state prior with a couple of disclosed anchors per state."""
    a_obs = np.zeros((n_steps, 2))
    p_obs = np.full((n_steps, 2), np.inf)
    # state 0: anchors at t=2, t=9
    a_obs[2, 0], p_obs[2, 0] = 0.30, 0.02
    a_obs[9, 0], p_obs[9, 0] = 0.50, 0.02
    # state 1: anchors at t=4, t=10
    a_obs[4, 1], p_obs[4, 1] = 0.65, 0.05
    a_obs[10, 1], p_obs[10, 1] = 0.40, 0.03
    return _prior(a_obs, p_obs)


def test_returns_figure_with_one_axis_per_state():
    prior = _multi_state_prior()
    extended = extend_states_prior_smoothed(prior, Q=0.02)

    fig, axes = plot_prior_density(extended, anchors=prior)

    assert isinstance(fig, plt.Figure)
    assert axes.shape == (2,)  # one panel per state
    assert all(isinstance(ax, plt.Axes) for ax in axes)
    plt.close(fig)


def test_raw_prior_with_inf_variance_renders_without_raising():
    # A raw, un-extended prior carries inf variance at most steps.
    prior = _multi_state_prior()
    assert np.isinf(prior["P_obs"].values).any()

    fig, axes = plot_prior_density(prior)  # anchors default to the prior's own disclosures

    assert isinstance(fig, plt.Figure)
    assert len(axes) == 2
    # No NaN/inf blowups leaked into the rendered density image extents.
    for ax in axes:
        images = ax.get_images()
        assert images, "expected a density image on each panel"
        arr = images[0].get_array()
        assert np.all(np.isfinite(np.ma.filled(arr, 0.0)))
    plt.close(fig)
