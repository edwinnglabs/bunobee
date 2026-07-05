from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import xarray as xr

from bunobee.models.ssp.prior import SspPrior


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
