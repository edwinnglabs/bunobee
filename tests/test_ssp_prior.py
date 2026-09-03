from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import xarray as xr

from bunobee.models.ssp.prior import SspPrior, disclosed_idx
from bunobee.models.ssp.transforms import transform_to_ekf, transform_to_ekf_st, validate_prior


def _valid_dataset(n_steps: int = 4, n_states: int = 2) -> xr.Dataset:
    """Build a minimal, schema-conformant complete prior dataset.

    Parameters
    ----------
    n_steps : int, optional
        Number of timesteps, by default 4.
    n_states : int, optional
        Number of latent states, by default 2.

    Returns
    -------
    xr.Dataset
        Dataset with ``a0``/``P0``/``a_obs``/``P_obs``/``positivity`` and
        ``time``/``state`` coordinates, satisfying the ``SspPrior`` schema.
    """
    return xr.Dataset(
        {
            "a0": (("state",), np.zeros(n_states)),
            "P0": (("state",), np.ones(n_states)),
            "a_obs": (("time", "state"), np.zeros((n_steps, n_states))),
            "P_obs": (("time", "state"), np.full((n_steps, n_states), np.inf)),
            "positivity": (("state",), np.array([False] + [True] * (n_states - 1))),
        },
        coords={"time": np.arange(n_steps), "state": [f"s{i}" for i in range(n_states)]},
    )


def test_valid_dataset_constructs_and_exposes_properties():
    prior = SspPrior(dataset=_valid_dataset())
    assert prior.a0.dims == ("state",)
    assert prior.P0.dims == ("state",)
    assert prior.a_obs.dims == ("time", "state")
    assert prior.P_obs.dims == ("time", "state")
    assert prior.positivity.dims == ("state",)


def test_extra_metadata_passes_through_unrestricted():
    ds = _valid_dataset()
    ds["sdy"] = 1.23
    ds["sigma_q_loc_prior"] = (("state",), np.array([0.05, 0.05]))
    prior = SspPrior(dataset=ds)
    assert float(prior.dataset["sdy"]) == pytest.approx(1.23)


def test_missing_a0_raises():
    ds = _valid_dataset().drop_vars("a0")
    with pytest.raises(ValueError, match="a0"):
        SspPrior(dataset=ds)


def test_missing_a_obs_raises_even_with_init_present():
    # validate_prior alone treats a_obs/P_obs as optional-paired; SspPrior
    # tightens this to unconditionally required.
    ds = _valid_dataset().drop_vars(["a_obs", "P_obs"])
    with pytest.raises(ValueError, match="a_obs"):
        SspPrior(dataset=ds)


def test_missing_state_coordinate_raises():
    ds = _valid_dataset().reset_index("state", drop=True)
    with pytest.raises(ValueError, match="state"):
        SspPrior(dataset=ds)


def test_missing_time_coordinate_raises():
    ds = _valid_dataset().reset_index("time", drop=True)
    with pytest.raises(ValueError, match="time"):
        SspPrior(dataset=ds)


def test_dataset_field_is_frozen():
    prior = SspPrior(dataset=_valid_dataset())
    with pytest.raises(FrozenInstanceError):
        prior.dataset = _valid_dataset()


# --------------------------------------------------------------------------- #
# The facade agrees with the wrapped dataset                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", ["time", "state", "a0", "P0", "a_obs", "P_obs", "positivity", "absent"])
def test_contains_matches_the_dataset_for_coords_and_data_vars(key):
    # `in` must not fall back to __iter__, which yields data-var names only:
    # xr.Dataset.__contains__ counts coordinates too.
    ds = _valid_dataset()
    assert (key in SspPrior(ds)) == (key in ds)


def test_len_and_iter_match_the_dataset():
    ds = _valid_dataset()
    prior = SspPrior(ds)
    assert len(prior) == len(ds)
    assert list(prior) == list(ds)


def test_getitem_and_attribute_access_delegate():
    ds = _valid_dataset()
    prior = SspPrior(ds)
    assert prior["a_obs"].dims == ds["a_obs"].dims
    assert prior.sizes == ds.sizes
    assert prior.attrs == ds.attrs


def test_from_dataset_and_to_dataset_round_trip():
    ds = _valid_dataset()
    prior = SspPrior.from_dataset(ds)
    assert prior.to_dataset() is ds
    assert SspPrior(prior.to_dataset()) == prior


def test_from_dataset_validates_like_the_constructor():
    ds = _valid_dataset().drop_vars("a0")
    with pytest.raises(ValueError, match="a0"):
        SspPrior.from_dataset(ds)


# --------------------------------------------------------------------------- #
# The two documented limits of the wrapper (see the class Notes)               #
# --------------------------------------------------------------------------- #


def test_xr_merge_rejects_the_wrapper_and_accepts_the_unwrapped_dataset():
    prior = SspPrior(_valid_dataset())
    other = xr.Dataset({"sdy": ((), 2.5)})
    with pytest.raises(TypeError):
        xr.merge([prior, other])
    assert "sdy" in xr.merge([prior.to_dataset(), other])


def test_xarray_operations_do_not_preserve_the_wrapper():
    # The invariant holds at function boundaries only: any xarray op drops back
    # to a bare Dataset, silently. Re-promote with from_dataset to re-check it.
    prior = SspPrior(_valid_dataset())
    for derived in (prior.copy(), prior.isel(time=[0, 1]), prior.assign(sdy=2.5)):
        assert isinstance(derived, xr.Dataset)
        assert not isinstance(derived, SspPrior)
    assert isinstance(SspPrior.from_dataset(prior.copy()), SspPrior)


# --------------------------------------------------------------------------- #
# The prior-consuming API takes an SspPrior and an xr.Dataset interchangeably  #
# --------------------------------------------------------------------------- #


def _complete_prior(n_steps: int = 4, n_states: int = 2) -> xr.Dataset:
    """Build a transform-ready complete prior: one anchor plus a ``sigma_q`` block.

    Parameters
    ----------
    n_steps : int, optional
        Number of timesteps, by default 4.
    n_states : int, optional
        Number of latent states, by default 2. State 0 is linear; the rest are
        positivity states, so ``a0`` must be strictly positive there.

    Returns
    -------
    xr.Dataset
        Dataset satisfying both the ``SspPrior`` schema and the a-space
        transform contract, with a single disclosed timestep at ``t = 1``.
    """
    ds = _valid_dataset(n_steps, n_states)
    ds["a0"] = (("state",), np.array([0.0] + [1.5] * (n_states - 1)))
    ds["a_obs"].values[1, :] = 0.5
    ds["P_obs"].values[1, :] = 0.02
    ds["sigma_q_loc_prior"] = (("state",), np.full(n_states, 0.05))
    ds["sigma_q_scale_prior"] = (("state",), np.full(n_states, 0.02))
    return ds


def _complete_prior_st(n_steps: int = 4, n_states: int = 2) -> xr.Dataset:
    """Same as :func:`_complete_prior` but with a full 2-D ``P0``."""
    ds = _complete_prior(n_steps, n_states).drop_vars("P0")
    ds["P0"] = (("state", "state_dual"), np.eye(n_states))
    return ds


def test_validate_prior_accepts_both_forms():
    ds = _complete_prior()
    validate_prior(ds)  # must not raise
    validate_prior(SspPrior(ds))  # must not raise either


def test_validate_prior_still_rejects_a_wrapped_but_incomplete_dataset():
    ds = _complete_prior().drop_vars(["a_obs", "P_obs"])
    with pytest.raises(ValueError, match="a0"):
        validate_prior(ds.drop_vars("a0"))


def test_disclosed_idx_identical_for_both_forms():
    ds = _complete_prior()
    assert np.array_equal(disclosed_idx(ds), np.array([1]))
    assert np.array_equal(disclosed_idx(SspPrior(ds)), disclosed_idx(ds))


@pytest.mark.parametrize(
    "transform, build",
    [(transform_to_ekf, _complete_prior), (transform_to_ekf_st, _complete_prior_st)],
    ids=["ekf", "ekf_st"],
)
def test_transforms_accept_both_forms_with_identical_results(transform, build):
    ds = build()
    from_dataset = transform(ds)
    from_wrapper = transform(SspPrior(ds))

    xr.testing.assert_identical(from_wrapper, from_dataset)
    # The a-space output is a plain Dataset in both cases: same schema as a
    # natural-scale prior, different semantics.
    assert not isinstance(from_wrapper, SspPrior)
    assert isinstance(from_wrapper, xr.Dataset)
