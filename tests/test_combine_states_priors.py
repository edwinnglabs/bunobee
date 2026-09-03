from __future__ import annotations

import itertools
from functools import reduce

import numpy as np
import pytest
import xarray as xr

from bunobee.models.ssp import combine_states_priors
from bunobee.models.ssp.prior import SspPrior
from bunobee.models.ssp.transforms import validate_prior


def _prior(
    a_obs: np.ndarray,
    p_obs: np.ndarray,
    a0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
    positivity: np.ndarray | None = None,
    n_states_coord: list[str] | None = None,
) -> xr.Dataset:
    """Build a minimal complete prior dataset for testing.

    Parameters
    ----------
    a_obs : np.ndarray, shape (n_steps, n_states)
        Disclosed state means.
    p_obs : np.ndarray, shape (n_steps, n_states)
        Disclosed state variances; ``inf`` marks undisclosed steps.
    a0 : np.ndarray or None, optional
        Initial-state mean over ``state``; defaults to zeros.
    P0 : np.ndarray or None, optional
        Initial-state variance over ``state``; defaults to ones.
    positivity : np.ndarray or None, optional
        Boolean positivity mask over ``state``; defaults to all-linear.
    n_states_coord : list of str or None, optional
        Explicit ``state`` coordinate labels; defaults to ``s0``, ``s1``, ...

    Returns
    -------
    xr.Dataset
        Dataset satisfying the complete ``SspPrior`` contract.
    """
    a_obs = np.asarray(a_obs, dtype=float)
    p_obs = np.asarray(p_obs, dtype=float)
    n_steps, n_states = p_obs.shape
    if a0 is None:
        a0 = np.zeros(n_states)
    if P0 is None:
        P0 = np.ones(n_states)
    if positivity is None:
        positivity = np.zeros(n_states, dtype=bool)
    if n_states_coord is None:
        n_states_coord = [f"s{i}" for i in range(n_states)]
    return xr.Dataset(
        {
            "a0": (("state",), np.asarray(a0, dtype=float)),
            "P0": (("state",), np.asarray(P0, dtype=float)),
            "a_obs": (("time", "state"), a_obs),
            "P_obs": (("time", "state"), p_obs),
            "positivity": (("state",), np.asarray(positivity, dtype=bool)),
        },
        coords={"time": np.arange(n_steps), "state": n_states_coord},
    )


def _simple(a: float, p: float, n_steps: int = 3, n_states: int = 2) -> xr.Dataset:
    return _prior(
        np.full((n_steps, n_states), a),
        np.full((n_steps, n_states), p),
        a0=np.full(n_states, a),
        P0=np.full(n_states, p),
    )


def test_returns_validated_ssp_prior_from_datasets():
    out = combine_states_priors(_simple(1.0, 2.0), _simple(1.0, 2.0))
    assert isinstance(out, SspPrior)
    validate_prior(out, require_init=True)


def test_accepts_wrapped_priors_on_either_side():
    ds = _simple(1.0, 2.0)
    wrapped = SspPrior(ds)
    for left, right in ((wrapped, ds), (ds, wrapped), (wrapped, wrapped)):
        out = combine_states_priors(left, right)
        assert isinstance(out, SspPrior)
        np.testing.assert_allclose(out["P_obs"].values, 1.0)


def test_identical_priors_halve_the_variance():
    # N(a, P) fused with itself is N(a, P/2): precision adds, the mean is unmoved.
    out = combine_states_priors(_simple(3.0, 0.4), _simple(3.0, 0.4))
    np.testing.assert_allclose(out["a_obs"].values, 3.0)
    np.testing.assert_allclose(out["P_obs"].values, 0.2)
    np.testing.assert_allclose(out["a0"].values, 3.0)
    np.testing.assert_allclose(out["P0"].values, 0.2)


def test_tight_prior_dominates_loose_prior():
    tight = _simple(10.0, 1e-4)
    loose = _simple(0.0, 1e4)
    out = combine_states_priors(tight, loose)
    np.testing.assert_allclose(out["a_obs"].values, 10.0, rtol=1e-6)
    assert np.all(out["P_obs"].values < 1e-4)
    # Symmetric: the operand order does not change the fused moments.
    flipped = combine_states_priors(loose, tight)
    np.testing.assert_allclose(out["a_obs"].values, flipped["a_obs"].values)
    np.testing.assert_allclose(out["P_obs"].values, flipped["P_obs"].values)


def test_closed_form_matches_hand_computed_fusion():
    left = _prior(np.array([[2.0]]), np.array([[4.0]]), a0=np.array([1.0]), P0=np.array([2.0]))
    right = _prior(np.array([[5.0]]), np.array([[1.0]]), a0=np.array([4.0]), P0=np.array([6.0]))
    out = combine_states_priors(left, right)

    expected_p = 1.0 / (1.0 / 4.0 + 1.0 / 1.0)
    expected_a = expected_p * (2.0 / 4.0 + 5.0 / 1.0)
    np.testing.assert_allclose(out["P_obs"].values, expected_p)
    np.testing.assert_allclose(out["a_obs"].values, expected_a)

    expected_p0 = 1.0 / (1.0 / 2.0 + 1.0 / 6.0)
    expected_a0 = expected_p0 * (1.0 / 2.0 + 4.0 / 6.0)
    np.testing.assert_allclose(out["P0"].values, expected_p0)
    np.testing.assert_allclose(out["a0"].values, expected_a0)


def test_fully_undisclosed_operand_leaves_the_first_prior_unchanged():
    left = _prior(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[0.5, np.inf], [2.0, 1.0]]),
        a0=np.array([7.0, 8.0]),
        P0=np.array([3.0, np.inf]),
    )
    undisclosed = _prior(
        np.zeros((2, 2)),
        np.full((2, 2), np.inf),
        a0=np.zeros(2),
        P0=np.full(2, np.inf),
    )
    out = combine_states_priors(left, undisclosed)
    np.testing.assert_allclose(out["a_obs"].values, left["a_obs"].values)
    np.testing.assert_allclose(out["P_obs"].values, left["P_obs"].values)
    np.testing.assert_allclose(out["a0"].values, left["a0"].values)
    np.testing.assert_allclose(out["P0"].values, left["P0"].values)


def test_two_undisclosed_steps_stay_undisclosed_not_nan():
    left = _prior(np.array([[1.0]]), np.array([[np.inf]]), a0=np.array([1.0]), P0=np.array([np.inf]))
    right = _prior(np.array([[9.0]]), np.array([[np.inf]]), a0=np.array([9.0]), P0=np.array([np.inf]))
    out = combine_states_priors(left, right)

    assert np.isinf(out["P_obs"].values).all()
    assert np.isinf(out["P0"].values).all()
    assert not np.isnan(out["a_obs"].values).any()
    assert not np.isnan(out["a0"].values).any()
    # The left operand's mean is preserved on a doubly-undisclosed entry.
    np.testing.assert_allclose(out["a_obs"].values, 1.0)
    np.testing.assert_allclose(out["a0"].values, 1.0)


def test_zero_variance_disclosure_wins_without_nan():
    left = _prior(np.array([[6.0]]), np.array([[0.0]]), a0=np.array([6.0]), P0=np.array([0.0]))
    right = _prior(np.array([[1.0]]), np.array([[2.0]]), a0=np.array([1.0]), P0=np.array([2.0]))
    for out in (combine_states_priors(left, right), combine_states_priors(right, left)):
        np.testing.assert_allclose(out["a_obs"].values, 6.0)
        np.testing.assert_allclose(out["P_obs"].values, 0.0)
        np.testing.assert_allclose(out["a0"].values, 6.0)


def test_mixed_disclosure_fuses_per_entry():
    # Entry-wise: (finite, inf) passes through, (finite, finite) fuses, (inf, inf) stays inf.
    left = _prior(np.array([[2.0, 4.0, 0.0]]), np.array([[1.0, 1.0, np.inf]]))
    right = _prior(np.array([[9.0, 6.0, 0.0]]), np.array([[np.inf, 1.0, np.inf]]))
    out = combine_states_priors(left, right)
    np.testing.assert_allclose(out["a_obs"].values, np.array([[2.0, 5.0, 0.0]]))
    np.testing.assert_allclose(out["P_obs"].values, np.array([[1.0, 0.5, np.inf]]))


def test_metadata_comes_from_the_left_operand():
    left = _simple(1.0, 2.0)
    left["sdy"] = 1.5
    left["sigma_q_loc_prior"] = (("state",), np.array([0.05, 0.05]))
    left.attrs["sigma_q_family"] = "truncated_normal"
    left.attrs["source"] = "vendor"

    right = _simple(1.0, 2.0)
    right["sdy"] = 9.9
    right.attrs["source"] = "panel"

    out = combine_states_priors(left, right)
    assert float(out["sdy"]) == pytest.approx(1.5)
    assert out.attrs["source"] == "vendor"
    assert "sigma_q_loc_prior" in out


def test_mismatched_time_coord_raises():
    left = _simple(1.0, 2.0, n_steps=3)
    right = _simple(1.0, 2.0, n_steps=3).assign_coords(time=np.arange(10, 13))
    with pytest.raises(ValueError, match="time"):
        combine_states_priors(left, right)


def test_mismatched_time_length_raises():
    with pytest.raises(ValueError, match="time"):
        combine_states_priors(_simple(1.0, 2.0, n_steps=3), _simple(1.0, 2.0, n_steps=4))


def test_mismatched_state_coord_raises():
    left = _prior(np.zeros((2, 2)), np.ones((2, 2)), n_states_coord=["trend", "seas"])
    right = _prior(np.zeros((2, 2)), np.ones((2, 2)), n_states_coord=["trend", "promo"])
    with pytest.raises(ValueError, match="state"):
        combine_states_priors(left, right)


def test_mismatched_positivity_raises():
    left = _prior(np.zeros((2, 2)), np.ones((2, 2)), positivity=np.array([True, False]))
    right = _prior(np.zeros((2, 2)), np.ones((2, 2)), positivity=np.array([False, False]))
    with pytest.raises(ValueError, match="positivity"):
        combine_states_priors(left, right)


def test_full_covariance_P0_raises_naming_the_shape():
    left = _simple(1.0, 2.0)
    full = _simple(1.0, 2.0).drop_vars("P0")
    full["P0"] = (("state", "state_dual"), np.eye(2))
    with pytest.raises(ValueError, match=r"state_dual"):
        combine_states_priors(left, full)
    with pytest.raises(ValueError, match=r"diagonal `P0`"):
        combine_states_priors(full, left)


def test_incomplete_operand_raises_naming_the_side():
    incomplete = _simple(1.0, 2.0).drop_vars("a0")
    with pytest.raises(ValueError, match="prior_b is not a complete SSP prior"):
        combine_states_priors(_simple(1.0, 2.0), incomplete)


def test_negative_variance_raises():
    bad = _prior(np.zeros((2, 1)), np.array([[1.0], [-1.0]]))
    with pytest.raises(ValueError, match="P_obs"):
        combine_states_priors(_simple(1.0, 2.0, n_steps=2, n_states=1), bad)


def test_nan_variance_raises():
    bad = _simple(1.0, 2.0, n_steps=2, n_states=1).copy()
    bad["P0"] = (("state",), np.array([np.nan]))
    with pytest.raises(ValueError, match="P0"):
        combine_states_priors(_simple(1.0, 2.0, n_steps=2, n_states=1), bad)


def test_inputs_are_not_mutated():
    left = _simple(1.0, 2.0)
    right = _simple(5.0, 8.0)
    left_a, left_p = left["a_obs"].values.copy(), left["P_obs"].values.copy()
    right_a, right_p = right["a_obs"].values.copy(), right["P_obs"].values.copy()

    combine_states_priors(left, right)

    np.testing.assert_allclose(left["a_obs"].values, left_a)
    np.testing.assert_allclose(left["P_obs"].values, left_p)
    np.testing.assert_allclose(right["a_obs"].values, right_a)
    np.testing.assert_allclose(right["P_obs"].values, right_p)


def _random(seed: int, n_steps: int = 4, n_states: int = 3) -> xr.Dataset:
    """Build a prior with pseudo-random finite moments, for order-invariance checks.

    Parameters
    ----------
    seed : int
        Seed for the moment draws.
    n_steps : int, optional
        Length of the ``time`` axis.
    n_states : int, optional
        Length of the ``state`` axis.

    Returns
    -------
    xr.Dataset
        Complete prior whose variances are all finite and strictly positive.
    """
    rng = np.random.default_rng(seed)
    return _prior(
        rng.normal(size=(n_steps, n_states)),
        rng.uniform(0.5, 4.0, size=(n_steps, n_states)),
        a0=rng.normal(size=n_states),
        P0=rng.uniform(0.5, 4.0, size=n_states),
    )


def test_fusion_is_associative():
    # Precision adds, so regrouping cannot change the result: (A * B) * C == A * (B * C).
    a, b, c = _random(1), _random(2), _random(3)
    left = combine_states_priors(combine_states_priors(a, b), c)
    right = combine_states_priors(a, combine_states_priors(b, c))
    for name in ("a_obs", "P_obs", "a0", "P0"):
        np.testing.assert_allclose(left[name].values, right[name].values, rtol=1e-12)


def test_fusion_is_associative_across_undisclosed_and_exact_steps():
    # Regrouping is safe through both degenerate precisions too, as long as the leftmost
    # operand is unchanged: `inf` contributes nothing and an exact disclosure wins outright.
    a, b, c = _random(4), _random(5), _random(6)
    a["P_obs"].values[0, 0] = np.inf
    b["P_obs"].values[0, 0] = np.inf
    b["P_obs"].values[1, 1] = 0.0
    c["P_obs"].values[0, 0] = np.inf
    c["P_obs"].values[2, 2] = np.inf
    left = combine_states_priors(combine_states_priors(a, b), c)
    right = combine_states_priors(a, combine_states_priors(b, c))
    for name in ("a_obs", "P_obs", "a0", "P0"):
        np.testing.assert_allclose(left[name].values, right[name].values, rtol=1e-12)
    # The step every operand left undisclosed is still undisclosed, not NaN.
    assert np.isinf(left["P_obs"].values[0, 0])


def test_operand_order_does_not_change_finite_moments():
    # With every variance finite the fusion is fully commutative, so all six orderings of a
    # three-way fold agree; only the non-fused metadata tracks the leftmost operand.
    priors = [_random(7), _random(8), _random(9)]
    reference = reduce(combine_states_priors, priors)
    for ordering in itertools.permutations(priors):
        out = reduce(combine_states_priors, ordering)
        for name in ("a_obs", "P_obs", "a0", "P0"):
            np.testing.assert_allclose(out[name].values, reference[name].values, rtol=1e-11)


def test_folding_many_fragments_matches_the_single_pass_closed_form():
    # N-ary fusion needs no new entry point: `reduce` over the binary op equals the one-pass
    # closed form summing all N precisions at once.
    priors = [_random(seed) for seed in range(10, 15)]
    folded = reduce(combine_states_priors, priors)

    precisions = np.stack([1.0 / p["P_obs"].values for p in priors])
    means = np.stack([p["a_obs"].values for p in priors])
    expected_p = 1.0 / precisions.sum(axis=0)
    expected_a = expected_p * (precisions * means).sum(axis=0)

    np.testing.assert_allclose(folded["P_obs"].values, expected_p, rtol=1e-12)
    np.testing.assert_allclose(folded["a_obs"].values, expected_a, rtol=1e-12)
    assert isinstance(folded, SspPrior)


def test_fusing_n_identical_priors_divides_the_variance_by_n():
    # The N-ary analogue of the two-operand halving: N copies of N(a, P) fuse to N(a, P/N).
    folded = reduce(combine_states_priors, [_simple(3.0, 0.6) for _ in range(4)])
    np.testing.assert_allclose(folded["a_obs"].values, 3.0)
    np.testing.assert_allclose(folded["P_obs"].values, 0.15)
