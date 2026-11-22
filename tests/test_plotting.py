from __future__ import annotations

from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede any pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from bunobee.models.ssp import plot_prior_cross_sections, plot_prior_density, plot_states
from bunobee.models.ssp.prior import disclosed_idx, extend_states_prior_smoothed


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


def _wide_prior(n_states: int, n_steps: int = 8) -> xr.Dataset:
    """An ``n_states`` prior with one disclosed anchor per state."""
    a_obs = np.zeros((n_steps, n_states))
    p_obs = np.full((n_steps, n_states), np.inf)
    a_obs[2, :], p_obs[2, :] = 0.4, 0.03
    return _prior(a_obs, p_obs)


def _grid_shape(axes: np.ndarray) -> tuple[int, int]:
    """Count distinct rows and columns of ``axes`` from their on-figure positions."""
    boxes = [ax.get_position() for ax in axes]
    return len({round(b.y0, 3) for b in boxes}), len({round(b.x0, 3) for b in boxes})


def test_default_layout_is_one_row_with_default_size():
    fig, axes = plot_prior_density(_wide_prior(3))

    assert axes.shape == (3,)
    assert _grid_shape(axes) == (1, 3)
    assert tuple(fig.get_size_inches()) == (21.0, 4.0)
    plt.close(fig)


def test_n_cols_wraps_into_rows_and_hides_unused_axes():
    fig, axes = plot_prior_density(_wide_prior(5), n_cols=2)

    assert axes.shape == (6,)  # 3x2 grid
    assert _grid_shape(axes) == (3, 2)
    assert [ax.get_visible() for ax in axes] == [True] * 5 + [False]
    assert tuple(fig.get_size_inches()) == (14.0, 12.0)
    plt.close(fig)


def test_n_rows_alone_derives_columns():
    fig, axes = plot_prior_density(_wide_prior(5), n_rows=2)

    assert axes.shape == (6,)  # 2x3 grid
    assert _grid_shape(axes) == (2, 3)
    plt.close(fig)


def test_figsize_and_dpi_are_applied():
    fig, _ = plot_prior_density(_wide_prior(2), figsize=(9.0, 3.0), dpi=50)

    assert tuple(fig.get_size_inches()) == (9.0, 3.0)
    assert fig.dpi == 50
    plt.close(fig)


def test_grid_too_small_or_non_positive_raises():
    prior = _wide_prior(5)
    with pytest.raises(ValueError, match="cannot hold"):
        plot_prior_density(prior, n_rows=2, n_cols=2)
    with pytest.raises(ValueError, match="positive"):
        plot_prior_density(prior, n_cols=0)


def _cross_section_priors() -> dict[str, xr.Dataset]:
    """A raw two-state prior with several anchors, alongside its smoothed extension."""
    prior = _multi_state_prior()
    return {"raw": prior, "smoothed": extend_states_prior_smoothed(prior, Q=0.02)}


def test_cross_sections_draw_a_normalised_bell_at_the_marginal_mean():
    prior = _wide_prior(1)
    state = str(prior["state"].values[0])

    fig, axes = plot_prior_cross_sections({"raw": prior}, state, [2])

    x, y = axes[0].get_lines()[0].get_data()
    assert x[np.argmax(y)] == pytest.approx(0.4, abs=0.01)
    assert np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)) == pytest.approx(1.0, abs=1e-3)
    plt.close(fig)


def test_cross_sections_overlay_every_prior_and_skip_undisclosed_steps():
    priors = _cross_section_priors()
    state = str(priors["raw"]["state"].values[0])
    t_anchor = int(np.flatnonzero(np.isfinite(priors["raw"]["P_obs"].values[:, 0]))[0])
    t_gap = int(np.flatnonzero(np.isinf(priors["raw"]["P_obs"].values[:, 0]))[0])

    fig, axes = plot_prior_cross_sections(priors, state, [t_anchor, t_gap])

    assert [line.get_label() for line in axes[0].get_lines()] == ["raw", "smoothed"]
    assert [line.get_label() for line in axes[1].get_lines()] == ["smoothed"]
    plt.close(fig)


def test_cross_sections_panel_with_nothing_disclosed_stays_empty():
    prior = _wide_prior(1)
    state = str(prior["state"].values[0])

    fig, axes = plot_prior_cross_sections({"raw": prior}, state, [0])

    assert axes[0].get_lines() == []
    plt.close(fig)


def test_cross_sections_colors_follow_each_prior():
    priors = _cross_section_priors()
    state = str(priors["raw"]["state"].values[0])
    t_anchor = int(np.flatnonzero(np.isfinite(priors["raw"]["P_obs"].values[:, 0]))[0])

    fig, axes = plot_prior_cross_sections(priors, state, [t_anchor], colors={"smoothed": "crimson"})

    assert axes[0].get_lines()[1].get_color() == "crimson"
    plt.close(fig)


def test_cross_sections_layout_wraps_and_hides_unused_axes():
    prior = _wide_prior(1)
    state = str(prior["state"].values[0])

    fig, axes = plot_prior_cross_sections({"raw": prior}, state, [2, 2, 2], n_cols=2)

    assert axes.shape == (4,)
    assert [ax.get_visible() for ax in axes] == [True, True, True, False]
    plt.close(fig)


@pytest.mark.parametrize(
    "priors_empty, state, times, colors, match",
    [
        (True, None, [2], None, "at least one prior"),
        (False, None, [], None, "at least one timestep"),
        (False, "nope", [2], None, "not in prior"),
        (False, None, [99], None, "outside prior"),
        (False, None, [-1], None, "outside prior"),
        (False, None, [2], [], "colours"),
    ],
)
def test_cross_sections_rejects_bad_arguments(priors_empty, state, times, colors, match):
    prior = _wide_prior(1)
    state = state or str(prior["state"].values[0])
    priors = {} if priors_empty else {"raw": prior}

    with pytest.raises(ValueError, match=match):
        plot_prior_cross_sections(priors, state, times, colors=colors)


def _states_inputs(n_steps: int = 12, n_draws: int = 50):
    """Posterior draws, dates and the two-state disclosure block for ``plot_states``."""
    prior = _multi_state_prior(n_steps)
    draws = np.random.default_rng(0).normal(0.5, 0.1, size=(n_draws, n_steps, 2))
    return {"at": draws}, np.arange(n_steps), ["s0", "s1"], prior


def _anchor_whiskers(ax: plt.Axes) -> list:
    return [c for c in ax.collections if str(c.get_label()).startswith("prior anchor ") and "CI" in c.get_label()]


def test_plot_states_draws_anchor_whiskers_at_the_ribbon_ci():
    posterior, dates, labels, prior = _states_inputs()
    a_obs, p_obs = prior["a_obs"].values, prior["P_obs"].values

    fig, axes = plot_states(
        posterior,
        dates,
        labels,
        states_key="at",
        obs_idx=disclosed_idx(prior),
        a_obs=a_obs,
        P_obs=p_obs,
        ci=(0.05, 0.5, 0.95),
    )

    z = NormalDist().inv_cdf(0.95)
    for i, ax in enumerate(axes[:2]):
        (whiskers,) = _anchor_whiskers(ax)
        assert whiskers.get_label() == "prior anchor 90% CI"
        segments = whiskers.get_segments()
        t = np.flatnonzero(np.isfinite(p_obs[:, i]))
        assert len(segments) == len(t)  # one whisker per finite disclosure of this state, none for inf
        for seg, step in zip(segments, t):
            (x0, lo), (x1, hi) = seg
            assert x0 == x1 == dates[step]
            sd = np.sqrt(p_obs[step, i])
            np.testing.assert_allclose([lo, hi], [a_obs[step, i] - z * sd, a_obs[step, i] + z * sd])
    plt.close(fig)


def test_plot_states_whiskers_follow_an_asymmetric_ci():
    posterior, dates, labels, prior = _states_inputs()
    a_obs, p_obs = prior["a_obs"].values, prior["P_obs"].values

    fig, axes = plot_states(
        posterior,
        dates,
        labels,
        obs_idx=disclosed_idx(prior),
        a_obs=a_obs,
        P_obs=p_obs,
        states_key="at",
        ci=(0.1, 0.5, 0.8),
    )

    (whiskers,) = _anchor_whiskers(axes[0])
    (_, lo), (_, hi) = whiskers.get_segments()[0]
    sd = np.sqrt(p_obs[2, 0])
    np.testing.assert_allclose(
        [lo, hi], [a_obs[2, 0] + NormalDist().inv_cdf(0.1) * sd, a_obs[2, 0] + NormalDist().inv_cdf(0.8) * sd]
    )
    plt.close(fig)


def test_plot_states_anchor_ci_off_keeps_only_the_markers():
    posterior, dates, labels, prior = _states_inputs()

    fig, axes = plot_states(
        posterior,
        dates,
        labels,
        states_key="at",
        obs_idx=disclosed_idx(prior),
        a_obs=prior["a_obs"].values,
        P_obs=prior["P_obs"].values,
        anchor_ci=False,
    )

    for ax in axes[:2]:
        assert not _anchor_whiskers(ax)
        assert [c.get_label() for c in ax.collections].count("prior anchor") == 1
    plt.close(fig)


def test_plot_states_legend_lists_anchors_drawn_on_later_panels_only():
    posterior, dates, labels, prior = _states_inputs()
    p_obs = prior["P_obs"].values.copy()
    p_obs[:, 0] = np.inf  # the first panel has no disclosure, as for an undisclosed level

    fig, axes = plot_states(
        posterior,
        dates,
        labels,
        states_key="at",
        obs_idx=disclosed_idx(prior),
        a_obs=prior["a_obs"].values,
        P_obs=p_obs,
    )

    texts = [t.get_text() for t in axes[0].get_legend().get_texts()]
    assert texts == ["median", "90% CI", "prior anchor 90% CI", "prior anchor"]
    plt.close(fig)


def test_plot_states_defaults_keep_the_original_look():
    posterior, dates, labels, prior = _states_inputs()

    fig, axes = plot_states(
        posterior,
        dates,
        labels,
        states_key="at",
        obs_idx=disclosed_idx(prior),
        a_obs=prior["a_obs"].values,
        P_obs=prior["P_obs"].values,
    )

    ribbon = next(c for c in axes[0].collections if c.get_label() == "90% CI")
    (whiskers,) = _anchor_whiskers(axes[0])
    markers = next(c for c in axes[0].collections if c.get_label() == "prior anchor")
    assert ribbon.get_alpha() == 0.25
    assert whiskers.get_alpha() == 0.45
    np.testing.assert_allclose(whiskers.get_colors()[0][:3], matplotlib.colors.to_rgb("crimson"))
    np.testing.assert_allclose(markers.get_facecolors()[0][:3], matplotlib.colors.to_rgb("crimson"))
    plt.close(fig)


def test_plot_states_anchor_color_and_alphas_are_applied():
    posterior, dates, labels, prior = _states_inputs()

    fig, axes = plot_states(
        posterior,
        dates,
        labels,
        states_key="at",
        obs_idx=disclosed_idx(prior),
        a_obs=prior["a_obs"].values,
        P_obs=prior["P_obs"].values,
        anchor_color="black",
        ribbon_alpha=0.1,
        anchor_alpha=0.8,
    )

    for ax in axes[:2]:
        ribbon = next(c for c in ax.collections if c.get_label() == "90% CI")
        (whiskers,) = _anchor_whiskers(ax)
        markers = next(c for c in ax.collections if c.get_label() == "prior anchor")
        assert ribbon.get_alpha() == 0.1
        assert whiskers.get_alpha() == 0.8
        np.testing.assert_allclose(whiskers.get_colors()[0][:3], (0.0, 0.0, 0.0))
        np.testing.assert_allclose(markers.get_facecolors()[0][:3], (0.0, 0.0, 0.0))
        assert markers.get_alpha() is None  # the markers stay opaque
    plt.close(fig)


@pytest.mark.parametrize("arg", ["ribbon_alpha", "anchor_alpha"])
@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_plot_states_rejects_alpha_outside_unit_interval(arg, value):
    posterior, dates, labels, _ = _states_inputs()
    with pytest.raises(ValueError, match=f"{arg} must be in"):
        plot_states(posterior, dates, labels, states_key="at", **{arg: value})
