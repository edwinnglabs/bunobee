from __future__ import annotations

import jax
import numpy as np
import pytest
import xarray as xr

from bunobee.models.ssp.prior import (
    disclosed_idx,
    extend_states_prior_nearest,
    extend_states_prior_smoothed,
)
from bunobee.models.ssp.transforms import validate_prior

jax.config.update("jax_enable_x64", True)  # exact single-anchor agreement needs float64


def _prior(a_obs: np.ndarray, p_obs: np.ndarray, positivity: np.ndarray | None = None) -> xr.Dataset:
    """Build a minimal disclosure-only prior dataset for testing.

    Parameters
    ----------
    a_obs : np.ndarray, shape (n_steps, n_states)
        Disclosed state means.
    p_obs : np.ndarray, shape (n_steps, n_states)
        Disclosed state variances; ``inf`` marks undisclosed steps.
    positivity : np.ndarray or None, optional
        Boolean positivity mask over ``state``; defaults to all-linear.

    Returns
    -------
    xr.Dataset
        Prior with ``a_obs`` / ``P_obs`` over ``(time, state)`` and a
        ``positivity`` mask, matching the ``construct_states_prior`` contract.
    """
    a_obs = np.asarray(a_obs, dtype=float)
    p_obs = np.asarray(p_obs, dtype=float)
    n_steps, n_states = p_obs.shape
    if positivity is None:
        positivity = np.zeros(n_states, dtype=bool)
    return xr.Dataset(
        {
            "a_obs": (("time", "state"), a_obs),
            "P_obs": (("time", "state"), p_obs),
            "positivity": (("state",), np.asarray(positivity, dtype=bool)),
        },
        coords={"time": np.arange(n_steps), "state": [f"s{i}" for i in range(n_states)]},
    )


def _single_anchor(n_steps: int, t_star: int, a_star: float, p_star: float) -> xr.Dataset:
    a_obs = np.zeros((n_steps, 1))
    p_obs = np.full((n_steps, 1), np.inf)
    a_obs[t_star, 0] = a_star
    p_obs[t_star, 0] = p_star
    return _prior(a_obs, p_obs)


def test_fills_inf_and_matches_random_walk_marginal():
    # Every undisclosed step becomes finite with the a* + |t-t*|.Q marginal.
    ds = _single_anchor(n_steps=7, t_star=3, a_star=2.0, p_star=0.5)
    q = 0.1
    out = extend_states_prior_nearest(ds, q)

    p = out["P_obs"].values[:, 0]
    a = out["a_obs"].values[:, 0]

    assert np.all(np.isfinite(p))
    expected_p = 0.5 + np.abs(np.arange(7) - 3) * q
    assert np.allclose(p, expected_p)
    # constant anchor mean everywhere
    assert np.allclose(a, 2.0)


def test_anchor_entry_preserved_exactly():
    # The disclosed step itself is untouched (lag 0 -> variance stays P*).
    ds = _single_anchor(n_steps=9, t_star=4, a_star=-1.5, p_star=0.3)
    out = extend_states_prior_nearest(ds, 0.2)
    assert out["P_obs"].values[4, 0] == pytest.approx(0.3)
    assert out["a_obs"].values[4, 0] == pytest.approx(-1.5)


def test_forward_backward_symmetry():
    # Variance is symmetric about the anchor: P[t*-k] == P[t*+k].
    t_star = 5
    ds = _single_anchor(n_steps=11, t_star=t_star, a_star=1.0, p_star=0.4)
    out = extend_states_prior_nearest(ds, 0.07)
    p = out["P_obs"].values[:, 0]
    for k in range(1, 6):
        assert p[t_star - k] == pytest.approx(p[t_star + k])


def test_linear_growth_slope_is_Q():
    # Away from the anchor the variance increments by exactly Q per step.
    q = 0.05
    ds = _single_anchor(n_steps=11, t_star=5, a_star=1.0, p_star=0.4)
    p = extend_states_prior_nearest(ds, q)["P_obs"].values[:, 0]
    forward = np.diff(p[5:])  # t = 5..10
    backward = np.diff(p[:6][::-1])  # t = 5..0
    assert np.allclose(forward, q)
    assert np.allclose(backward, q)


def test_Q_to_inf_recovers_no_extension():
    # Q -> inf keeps only the anchor informed; all other steps stay inf.
    ds = _single_anchor(n_steps=8, t_star=2, a_star=3.0, p_star=0.25)
    out = extend_states_prior_nearest(ds, np.inf)

    p = out["P_obs"].values[:, 0]
    assert p[2] == pytest.approx(0.25)
    off_anchor = np.delete(p, 2)
    assert np.all(np.isinf(off_anchor))
    # disclosure set is unchanged relative to the input
    assert np.array_equal(disclosed_idx(out), disclosed_idx(ds))


def test_Q_zero_holds_variance_flat():
    # Q = 0 carries the anchor's variance flat across its whole region.
    ds = _single_anchor(n_steps=6, t_star=1, a_star=2.0, p_star=0.5)
    out = extend_states_prior_nearest(ds, 0.0)
    assert np.allclose(out["P_obs"].values[:, 0], 0.5)
    assert np.allclose(out["a_obs"].values[:, 0], 2.0)


def test_multiple_anchors_use_nearest():
    # Each step adopts the nearest anchor's mean and grows from its variance.
    a_obs = np.zeros((11, 1))
    p_obs = np.full((11, 1), np.inf)
    a_obs[2, 0], p_obs[2, 0] = 1.0, 0.1
    a_obs[8, 0], p_obs[8, 0] = 3.0, 0.1
    ds = _prior(a_obs, p_obs)
    q = 0.05
    out = extend_states_prior_nearest(ds, q)

    a = out["a_obs"].values[:, 0]
    p = out["P_obs"].values[:, 0]

    # t=4 is nearer anchor at t=2 (dist 2) than t=8 (dist 6)
    assert a[4] == pytest.approx(1.0)
    assert p[4] == pytest.approx(0.1 + 2 * q)
    # t=6 is nearer anchor at t=8 (dist 2) than t=2 (dist 4)
    assert a[6] == pytest.approx(3.0)
    assert p[6] == pytest.approx(0.1 + 2 * q)
    # both anchors preserved exactly
    assert p[2] == pytest.approx(0.1)
    assert p[8] == pytest.approx(0.1)


def test_equidistant_tie_breaks_to_smaller_variance():
    # A step equidistant from two anchors picks the more informative (smaller P*).
    a_obs = np.zeros((7, 1))
    p_obs = np.full((7, 1), np.inf)
    a_obs[0, 0], p_obs[0, 0] = 10.0, 0.9  # far/loose anchor
    a_obs[6, 0], p_obs[6, 0] = 20.0, 0.1  # tight anchor
    ds = _prior(a_obs, p_obs)
    out = extend_states_prior_nearest(ds, 0.05)
    # t=3 is equidistant (dist 3); the tighter anchor at t=6 wins
    assert out["a_obs"].values[3, 0] == pytest.approx(20.0)
    assert out["P_obs"].values[3, 0] == pytest.approx(0.1 + 3 * 0.05)


def test_state_without_anchor_stays_inf():
    # A fully-undisclosed state column is left at inf variance.
    a_obs = np.zeros((5, 2))
    p_obs = np.full((5, 2), np.inf)
    a_obs[1, 0], p_obs[1, 0] = 4.0, 0.2  # only state 0 has an anchor
    ds = _prior(a_obs, p_obs)
    out = extend_states_prior_nearest(ds, 0.1)

    assert np.all(np.isfinite(out["P_obs"].values[:, 0]))
    assert np.all(np.isinf(out["P_obs"].values[:, 1]))


def test_per_state_vector_Q():
    # A length-n_states Q applies its own slope to each state.
    a_obs = np.zeros((5, 2))
    p_obs = np.full((5, 2), np.inf)
    a_obs[0, 0], p_obs[0, 0] = 1.0, 0.0
    a_obs[0, 1], p_obs[0, 1] = 2.0, 0.0
    ds = _prior(a_obs, p_obs)
    out = extend_states_prior_nearest(ds, np.array([0.1, 0.3]))

    lag = np.arange(5)
    assert np.allclose(out["P_obs"].values[:, 0], lag * 0.1)
    assert np.allclose(out["P_obs"].values[:, 1], lag * 0.3)


def test_output_passes_validate_prior():
    ds = _single_anchor(n_steps=6, t_star=2, a_star=1.0, p_star=0.2)
    out = extend_states_prior_nearest(ds, 0.05)
    validate_prior(out, require_init=False)  # must not raise


def test_input_dataset_not_mutated():
    ds = _single_anchor(n_steps=6, t_star=2, a_star=1.0, p_star=0.2)
    before_p = ds["P_obs"].values.copy()
    before_a = ds["a_obs"].values.copy()
    extend_states_prior_nearest(ds, 0.05)
    assert np.array_equal(ds["P_obs"].values, before_p, equal_nan=True)
    assert np.array_equal(ds["a_obs"].values, before_a)


def test_passthrough_variables_preserved():
    # positivity and any extra vars survive the extension unchanged.
    ds = _single_anchor(n_steps=6, t_star=2, a_star=1.0, p_star=0.2)
    positivity = np.array([True])
    ds["positivity"] = (("state",), positivity)
    out = extend_states_prior_nearest(ds, 0.05)
    assert np.array_equal(out["positivity"].values, positivity)


def test_negative_Q_raises():
    ds = _single_anchor(n_steps=5, t_star=2, a_star=1.0, p_star=0.2)
    with pytest.raises(ValueError, match="non-negative"):
        extend_states_prior_nearest(ds, -0.1)


def test_missing_obs_block_raises():
    ds = xr.Dataset(
        {"positivity": (("state",), np.array([False]))},
        coords={"state": ["s0"]},
    )
    with pytest.raises(ValueError, match="a_obs.*P_obs"):
        extend_states_prior_nearest(ds, 0.1)


# --------------------------------------------------------------------------- #
# extend_states_prior_smoothed (KF-forward + RTS-backward, fusing all anchors) #
# --------------------------------------------------------------------------- #


def _multi_anchor() -> xr.Dataset:
    # Three anchors: tight neighbours flanking a loose outlier in the middle.
    a_obs = np.zeros((40, 1))
    p_obs = np.full((40, 1), np.inf)
    a_obs[5, 0], p_obs[5, 0] = 0.30, 0.02
    a_obs[20, 0], p_obs[20, 0] = 0.65, 0.15
    a_obs[34, 0], p_obs[34, 0] = 0.35, 0.02
    return _prior(a_obs, p_obs)


def test_smoothed_single_anchor_matches_nearest():
    # For a single-anchor channel the smoother marginal equals the nearest-anchor
    # heuristic to numerical noise (set by the diffuse-prior proxy P0 in float64).
    ds = _single_anchor(n_steps=25, t_star=12, a_star=0.6, p_star=0.03)
    q = 0.02
    nearest = extend_states_prior_nearest(ds, q)
    smoothed = extend_states_prior_smoothed(ds, q)

    assert np.allclose(smoothed["a_obs"].values, nearest["a_obs"].values, atol=1e-7)
    assert np.allclose(smoothed["P_obs"].values, nearest["P_obs"].values, atol=1e-7)


def test_smoothed_variance_never_exceeds_nearest_multi_anchor():
    # Fusing every anchor is never less informative than picking the nearest one:
    # the smoother variance is <= the heuristic at every step.
    ds = _multi_anchor()
    q = 0.02
    nearest = extend_states_prior_nearest(ds, q)
    smoothed = extend_states_prior_smoothed(ds, q)

    p_near = nearest["P_obs"].values[:, 0]
    p_smooth = smoothed["P_obs"].values[:, 0]
    assert np.all(p_smooth <= p_near + 1e-9)
    # and strictly tighter somewhere (the midpoints between anchors)
    assert np.any(p_smooth < p_near - 1e-6)


def test_smoothed_revises_loose_anchor_toward_tight_neighbours():
    # Unlike the heuristic, which freezes each anchor, the smoother reads P* as
    # observation noise: the loose middle anchor is pulled toward its tight
    # neighbours rather than reproduced exactly.
    ds = _multi_anchor()
    smoothed = extend_states_prior_smoothed(ds, 0.02)
    a20 = smoothed["a_obs"].values[20, 0]
    assert a20 < 0.65  # revised down from the raw loose disclosure
    assert a20 > 0.35  # but still above the tight neighbours


def test_smoothed_state_without_anchor_stays_inf():
    # A fully-undisclosed state column comes back at inf despite the diffuse P0.
    a_obs = np.zeros((6, 2))
    p_obs = np.full((6, 2), np.inf)
    a_obs[2, 0], p_obs[2, 0] = 0.5, 0.05  # only state 0 has an anchor
    ds = _prior(a_obs, p_obs)
    out = extend_states_prior_smoothed(ds, 0.02)

    assert np.all(np.isfinite(out["P_obs"].values[:, 0]))
    assert np.all(np.isinf(out["P_obs"].values[:, 1]))


def _multi_state_multi_anchor() -> xr.Dataset:
    """Build a four-column prior spanning every anchoring regime.

    Returns
    -------
    xr.Dataset
        30-step prior whose columns are, in order: unanchored (a ``level``-like
        state), two tight anchors, three anchors with a loose middle outlier,
        and a single anchor.
    """
    a_obs = np.zeros((30, 4))
    p_obs = np.full((30, 4), np.inf)
    # s0: no anchor at all
    a_obs[4, 1], p_obs[4, 1] = 0.20, 0.02  # s1: two tight anchors
    a_obs[25, 1], p_obs[25, 1] = 0.45, 0.02
    a_obs[3, 2], p_obs[3, 2] = 0.30, 0.02  # s2: loose middle outlier
    a_obs[15, 2], p_obs[15, 2] = 0.65, 0.15
    a_obs[27, 2], p_obs[27, 2] = 0.35, 0.02
    a_obs[10, 3], p_obs[10, 3] = 0.60, 0.03  # s3: single anchor
    return _prior(a_obs, p_obs)


def test_smoothed_per_state_vector_Q():
    # A length-n_states Q gives each column its own random-walk slope: two columns
    # carrying an identical single anchor open up at their own Q away from it.
    a_obs = np.zeros((11, 2))
    p_obs = np.full((11, 2), np.inf)
    a_obs[5, :], p_obs[5, :] = 0.5, 0.01
    ds = _prior(a_obs, p_obs)
    q = np.array([0.02, 0.2])
    out = extend_states_prior_smoothed(ds, q)

    # Single anchor per column, so the smoother marginal is the exact
    # P* + |t - t*|.Q cone, with a different slope on each column.
    lag = np.abs(np.arange(11) - 5)
    assert np.allclose(out["P_obs"].values[:, 0], 0.01 + lag * q[0], atol=1e-7)
    assert np.allclose(out["P_obs"].values[:, 1], 0.01 + lag * q[1], atol=1e-7)
    assert np.allclose(out["a_obs"].values, 0.5, atol=1e-7)


def test_smoothed_multi_state_columns_are_decoupled():
    # The filter runs with Z = 0, so the state columns never couple: extending
    # each column alone as a one-state prior reproduces the joint run exactly.
    ds = _multi_state_multi_anchor()
    q = np.array([0.02, 0.01, 0.02, 0.05])
    joint = extend_states_prior_smoothed(ds, q)

    for i in range(ds.sizes["state"]):
        solo = extend_states_prior_smoothed(ds.isel(state=[i]), q[i])
        assert np.array_equal(solo["a_obs"].values[:, 0], joint["a_obs"].values[:, i])
        assert np.array_equal(solo["P_obs"].values[:, 0], joint["P_obs"].values[:, i])

    # The unanchored column stays fully undisclosed; every other one is filled.
    assert np.all(np.isinf(joint["P_obs"].values[:, 0]))
    assert np.all(np.isfinite(joint["P_obs"].values[:, 1:]))


def test_smoothed_multi_state_tighter_than_nearest_per_column():
    # Each anchored column is smoothed on its own terms: never wider than the
    # nearest-anchor heuristic, and strictly tighter wherever it has two anchors.
    ds = _multi_state_multi_anchor()
    q = np.array([0.02, 0.01, 0.02, 0.05])
    nearest = extend_states_prior_nearest(ds, q)
    smoothed = extend_states_prior_smoothed(ds, q)

    p_near = nearest["P_obs"].values
    p_smooth = smoothed["P_obs"].values
    for i in (1, 2, 3):
        assert np.all(p_smooth[:, i] <= p_near[:, i] + 1e-9)
    for i in (1, 2):  # multi-anchor columns fuse both directions
        assert np.any(p_smooth[:, i] < p_near[:, i] - 1e-6)
    # single-anchor column: the two agree to the diffuse-prior noise floor
    assert np.allclose(p_smooth[:, 3], p_near[:, 3], atol=1e-7)


def test_smoothed_output_passes_validate_prior():
    out = extend_states_prior_smoothed(_multi_anchor(), 0.02)
    validate_prior(out, require_init=False)  # must not raise


def test_smoothed_input_dataset_not_mutated():
    ds = _multi_anchor()
    before_p = ds["P_obs"].values.copy()
    before_a = ds["a_obs"].values.copy()
    extend_states_prior_smoothed(ds, 0.02)
    assert np.array_equal(ds["P_obs"].values, before_p, equal_nan=True)
    assert np.array_equal(ds["a_obs"].values, before_a)


def test_smoothed_negative_Q_raises():
    ds = _single_anchor(n_steps=5, t_star=2, a_star=0.5, p_star=0.05)
    with pytest.raises(ValueError, match="non-negative"):
        extend_states_prior_smoothed(ds, -0.1)


def test_smoothed_missing_obs_block_raises():
    ds = xr.Dataset(
        {"positivity": (("state",), np.array([False]))},
        coords={"state": ["s0"]},
    )
    with pytest.raises(ValueError, match="a_obs.*P_obs"):
        extend_states_prior_smoothed(ds, 0.02)
