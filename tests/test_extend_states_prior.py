from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from bunobee.models.ssp.prior import disclosed_idx, extend_states_prior
from bunobee.models.ssp.transforms import validate_prior


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
    out = extend_states_prior(ds, q)

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
    out = extend_states_prior(ds, 0.2)
    assert out["P_obs"].values[4, 0] == pytest.approx(0.3)
    assert out["a_obs"].values[4, 0] == pytest.approx(-1.5)


def test_forward_backward_symmetry():
    # Variance is symmetric about the anchor: P[t*-k] == P[t*+k].
    t_star = 5
    ds = _single_anchor(n_steps=11, t_star=t_star, a_star=1.0, p_star=0.4)
    out = extend_states_prior(ds, 0.07)
    p = out["P_obs"].values[:, 0]
    for k in range(1, 6):
        assert p[t_star - k] == pytest.approx(p[t_star + k])


def test_linear_growth_slope_is_Q():
    # Away from the anchor the variance increments by exactly Q per step.
    q = 0.05
    ds = _single_anchor(n_steps=11, t_star=5, a_star=1.0, p_star=0.4)
    p = extend_states_prior(ds, q)["P_obs"].values[:, 0]
    forward = np.diff(p[5:])  # t = 5..10
    backward = np.diff(p[:6][::-1])  # t = 5..0
    assert np.allclose(forward, q)
    assert np.allclose(backward, q)


def test_Q_to_inf_recovers_no_extension():
    # Q -> inf keeps only the anchor informed; all other steps stay inf.
    ds = _single_anchor(n_steps=8, t_star=2, a_star=3.0, p_star=0.25)
    out = extend_states_prior(ds, np.inf)

    p = out["P_obs"].values[:, 0]
    assert p[2] == pytest.approx(0.25)
    off_anchor = np.delete(p, 2)
    assert np.all(np.isinf(off_anchor))
    # disclosure set is unchanged relative to the input
    assert np.array_equal(disclosed_idx(out), disclosed_idx(ds))


def test_Q_zero_holds_variance_flat():
    # Q = 0 carries the anchor's variance flat across its whole region.
    ds = _single_anchor(n_steps=6, t_star=1, a_star=2.0, p_star=0.5)
    out = extend_states_prior(ds, 0.0)
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
    out = extend_states_prior(ds, q)

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
    out = extend_states_prior(ds, 0.05)
    # t=3 is equidistant (dist 3); the tighter anchor at t=6 wins
    assert out["a_obs"].values[3, 0] == pytest.approx(20.0)
    assert out["P_obs"].values[3, 0] == pytest.approx(0.1 + 3 * 0.05)


def test_state_without_anchor_stays_inf():
    # A fully-undisclosed state column is left at inf variance.
    a_obs = np.zeros((5, 2))
    p_obs = np.full((5, 2), np.inf)
    a_obs[1, 0], p_obs[1, 0] = 4.0, 0.2  # only state 0 has an anchor
    ds = _prior(a_obs, p_obs)
    out = extend_states_prior(ds, 0.1)

    assert np.all(np.isfinite(out["P_obs"].values[:, 0]))
    assert np.all(np.isinf(out["P_obs"].values[:, 1]))


def test_per_state_vector_Q():
    # A length-n_states Q applies its own slope to each state.
    a_obs = np.zeros((5, 2))
    p_obs = np.full((5, 2), np.inf)
    a_obs[0, 0], p_obs[0, 0] = 1.0, 0.0
    a_obs[0, 1], p_obs[0, 1] = 2.0, 0.0
    ds = _prior(a_obs, p_obs)
    out = extend_states_prior(ds, np.array([0.1, 0.3]))

    lag = np.arange(5)
    assert np.allclose(out["P_obs"].values[:, 0], lag * 0.1)
    assert np.allclose(out["P_obs"].values[:, 1], lag * 0.3)


def test_output_passes_validate_prior():
    ds = _single_anchor(n_steps=6, t_star=2, a_star=1.0, p_star=0.2)
    out = extend_states_prior(ds, 0.05)
    validate_prior(out, require_init=False)  # must not raise


def test_input_dataset_not_mutated():
    ds = _single_anchor(n_steps=6, t_star=2, a_star=1.0, p_star=0.2)
    before_p = ds["P_obs"].values.copy()
    before_a = ds["a_obs"].values.copy()
    extend_states_prior(ds, 0.05)
    assert np.array_equal(ds["P_obs"].values, before_p, equal_nan=True)
    assert np.array_equal(ds["a_obs"].values, before_a)


def test_passthrough_variables_preserved():
    # positivity and any extra vars survive the extension unchanged.
    ds = _single_anchor(n_steps=6, t_star=2, a_star=1.0, p_star=0.2)
    positivity = np.array([True])
    ds["positivity"] = (("state",), positivity)
    out = extend_states_prior(ds, 0.05)
    assert np.array_equal(out["positivity"].values, positivity)


def test_negative_Q_raises():
    ds = _single_anchor(n_steps=5, t_star=2, a_star=1.0, p_star=0.2)
    with pytest.raises(ValueError, match="non-negative"):
        extend_states_prior(ds, -0.1)


def test_missing_obs_block_raises():
    ds = xr.Dataset(
        {"positivity": (("state",), np.array([False]))},
        coords={"state": ["s0"]},
    )
    with pytest.raises(ValueError, match="a_obs.*P_obs"):
        extend_states_prior(ds, 0.1)
