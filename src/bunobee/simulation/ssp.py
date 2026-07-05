from __future__ import annotations

import numpy as np
import xarray as xr
from jax import numpy as jnp

from bunobee.models.ssp.transforms import validate_prior


def construct_states_prior(
    n_steps: int,
    n_states: int,
    true_states: jnp.ndarray,
    regressors: list,
    n_periods: int = 3,
    n_points: int = 7,
    seed: int = 42,
    obs_scale: float = 0.1,
    positivity: np.ndarray | None = None,
) -> xr.Dataset:
    """Construct a_obs and P_obs by disclosing the ground-truth latent
    state over n_periods randomly drawn windows of n_points consecutive steps.

    Simulation-only: requires ``true_states``, a ground truth that isn't
    available outside a synthetic/prototype setting. Not usable to build a
    real prior from production data.

    Parameters
    ----------
    n_steps : int
        Total number of time steps.
    n_states : int
        Number of latent states (level + regressors).
    true_states : jnp.ndarray, shape (n_states,)
        Ground-truth state vector; level entry is ignored (var stays inf).
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
    positivity : np.ndarray or None, optional
        Boolean mask of length ``n_states`` indicating which states are
        positivity-constrained.  Defaults to ``[False, True, …]`` — level
        is linear, all regressors are positivity states.

    Returns
    -------
    xr.Dataset
        Variables ``a_obs`` and ``P_obs`` with dims ``(time, state)`` and
        ``positivity`` with dim ``state``.  Coords: ``time`` (0…n_steps-1) and
        ``state`` (["level", *regressors]).  Disclosure time indices are
        not stored; derive them with :func:`~bunobee.models.ssp.prior.disclosed_idx`.
        This is the disclosure-only intermediate, not a complete
        :class:`~bunobee.models.ssp.prior.SspPrior` -- it carries no ``a0`` /
        ``P0``, which are supplied downstream from the real (non-simulated)
        prior belief.
    """
    rng = np.random.default_rng(seed)
    # sample n_periods window starts; ensure each window fits within n_steps
    starts = rng.choice(n_steps - n_points + 1, size=n_periods, replace=False)
    obs_idx = np.unique(np.concatenate([np.arange(s, s + n_points) for s in starts]))

    a_obs = jnp.zeros((n_steps, n_states))
    # default inf = zero precision = no information; pure filter carries through
    P_obs = jnp.full((n_steps, n_states), jnp.inf)

    a_obs = a_obs.at[obs_idx].set(true_states)
    # level has no priors so its variance stays inf
    var_row = jnp.array([jnp.inf] + [obs_scale**2] * len(regressors))
    P_obs = P_obs.at[obs_idx].set(var_row)

    if positivity is None:
        positivity = np.array([False] + [True] * len(regressors))
    else:
        positivity = np.asarray(positivity, dtype=bool)

    state_labels = ["level", *regressors]
    ds = xr.Dataset(
        {
            "a_obs": (("time", "state"), np.asarray(a_obs)),
            "P_obs": (("time", "state"), np.asarray(P_obs)),
            "positivity": (("state",), positivity),
        },
        coords={
            "time": np.arange(n_steps),
            "state": state_labels,
        },
    )
    # Fail fast on a malformed disclosure block; a0/P0 are supplied downstream.
    validate_prior(ds, require_init=False)
    return ds
