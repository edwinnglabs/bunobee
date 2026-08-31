from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr


def a_to_lam(
    arr: np.ndarray,
    exponent: float,
    positivity: np.ndarray | None = None,
    clip: float | None = None,
) -> np.ndarray:
    """Convert a-space values to λ-space for positivity states.

    Parameters
    ----------
    arr : np.ndarray
        Array in a-space, shape ``(..., n_states)`` — e.g. ``a_obs``.
    exponent : float
        EKF nonlinearity exponent: ``λ = exp(exponent · a)``.
    positivity : np.ndarray or None, optional
        Boolean mask of length ``n_states``.  ``True`` = positivity state.
        ``None`` treats every state as positivity.
    clip : float or None, optional
        Symmetric bound on the exponent argument: ``exp(clip(exponent · a, -clip, clip))``.
        ``None`` (default) applies no clip.  Pass ``10.0`` to reproduce the guard
        ``kalman_filter_1d_ekf_st`` applies internally, which keeps a long-horizon
        forecast with a large ``sigma_q`` finite instead of overflowing to ``inf``.

    Returns
    -------
    np.ndarray
        Same shape.  Positivity columns transformed via ``exp(exponent · a)``;
        linear columns passed through unchanged.
    """
    out = np.array(arr, dtype=float)
    n_states = out.shape[-1]
    mask = np.ones(n_states, dtype=bool) if positivity is None else np.asarray(positivity, dtype=bool)
    scaled = exponent * out[..., mask]
    if clip is not None:
        scaled = np.clip(scaled, -abs(clip), abs(clip))
    out[..., mask] = np.exp(scaled)
    return out


def lam_to_a(
    arr: np.ndarray,
    exponent: float,
    positivity: np.ndarray | None = None,
) -> np.ndarray:
    """Convert λ-space values to a-space for positivity states.

    Parameters
    ----------
    arr : np.ndarray
        Array in λ-space, shape ``(..., n_states)``.  Positivity columns must
        be strictly positive; ``log`` is applied element-wise.
    exponent : float
        EKF nonlinearity exponent: ``a = log(λ) / exponent``.
    positivity : np.ndarray or None, optional
        Boolean mask of length ``n_states``.  ``None`` treats all as positivity.

    Returns
    -------
    np.ndarray
        Same shape.  Positivity columns transformed via ``log(λ) / exponent``;
        linear columns passed through unchanged.
    """
    out = np.array(arr, dtype=float)
    n_states = out.shape[-1]
    mask = np.ones(n_states, dtype=bool) if positivity is None else np.asarray(positivity, dtype=bool)
    out[..., mask] = np.log(out[..., mask]) / exponent
    return out


def posterior_to_xarray(
    posterior: Mapping[str, np.ndarray],
    *,
    dims: Mapping[str, Sequence[str]] | None = None,
    coords: Mapping[str, Sequence] | None = None,
    drop: Sequence[str] | None = None,
    keep: Sequence[str] | None = None,
) -> xr.Dataset:
    """Convert a chain-grouped numpyro posterior dict to an ``xarray.Dataset``.

    Each value of ``posterior`` must have leading ``(chain, draw, ...)`` axes,
    matching the output of ``mcmc.get_samples(group_by_chain=True)``. The
    resulting dataset is suitable for wrapping in
    ``arviz.InferenceData(posterior=ds)`` for downstream diagnostics
    (``az.summary``, ``az.plot_trace``, ``az.plot_rank``, ...).

    Parameters
    ----------
    posterior : mapping[str, np.ndarray]
        Mapping from site name to draws of shape ``(n_chains, n_draws, *event)``.
    dims : mapping[str, sequence of str], optional
        Names for each variable's event axes (everything past ``chain`` and
        ``draw``). Variables omitted here get auto-named axes
        ``"<name>_dim_<i>"``.
    coords : mapping[str, sequence], optional
        Coordinate values keyed by dimension name. Dimensions without an entry
        fall back to a plain integer range.
    drop : sequence of str, optional
        Site names to omit from the dataset. Mutually exclusive with ``keep``.
    keep : sequence of str, optional
        Site names to retain; everything else is dropped. Mutually exclusive
        with ``drop``.

    Returns
    -------
    xarray.Dataset
        One ``DataArray`` per kept variable with dims
        ``(chain, draw, *event_dims)``.
    """
    if drop is not None and keep is not None:
        raise ValueError("pass at most one of `drop` or `keep`, not both")

    drop_set = set(drop or ())
    keep_set = set(keep) if keep is not None else None
    items = {
        k: np.asarray(v) for k, v in posterior.items() if k not in drop_set and (keep_set is None or k in keep_set)
    }
    if not items:
        raise ValueError("no variables left to convert after applying drop/keep")

    dims = dict(dims or {})
    user_coords = dict(coords or {})

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    out_coords: dict[str, np.ndarray] = {}

    for name, arr in items.items():
        if arr.ndim < 2:
            raise ValueError(
                f"variable {name!r} has shape {arr.shape}; expected leading "
                "(chain, draw) axes from group_by_chain=True samples"
            )
        event_shape = arr.shape[2:]
        event_dims = list(dims.get(name) or [f"{name}_dim_{i}" for i in range(len(event_shape))])
        if len(event_dims) != len(event_shape):
            raise ValueError(f"dims for {name!r} has length {len(event_dims)} but event " f"shape is {event_shape}")

        for d, size in zip(event_dims, event_shape):
            if d in out_coords:
                if len(out_coords[d]) != size:
                    raise ValueError(
                        f"dim {d!r} reused with inconsistent size: " f"{len(out_coords[d])} vs {size} (from {name!r})"
                    )
                continue
            if d in user_coords:
                values = np.asarray(user_coords[d])
                if values.shape != (size,):
                    raise ValueError(f"coord {d!r} has shape {values.shape} but {name!r} " f"expects size {size}")
                out_coords[d] = values
            else:
                out_coords[d] = np.arange(size)

        data_vars[name] = (("chain", "draw", *event_dims), arr)

    sample_arr = next(iter(items.values()))
    out_coords.setdefault("chain", np.arange(sample_arr.shape[0]))
    out_coords.setdefault("draw", np.arange(sample_arr.shape[1]))

    return xr.Dataset(data_vars=data_vars, coords=out_coords)
