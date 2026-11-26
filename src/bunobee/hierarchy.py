"""Hierarchies of series: sum bottom-level forecast draws up the levels, then take their quantiles.

A forecast distribution at an aggregate level, e.g. per store or in total, comes from the bottom-level draws: sum
each draw over the series of a group, then take quantiles over the draws. The sums keep whatever correlation the
model's draws carry between series, which summing the bottom-level quantiles would not.

A :class:`Hierarchy` holds named :class:`Level` s, each a series → group index and the group labels, and is built
from a per-series table with :meth:`Hierarchy.from_frame`. :func:`aggregate_draws` sums ``(sample, time, series)``
draws, an array or the dataset ``SSP.predict`` returns, to ``(sample, time, group)``; :func:`draw_quantiles` takes
quantiles over ``sample``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse

SAMPLE_DIM = "sample"
SERIES_DIM = "series"
GROUP_DIM = "group"
DRAWS_VAR = "forecast_samples"
TOTAL_LABEL = "total"


@dataclass(frozen=True, eq=False)
class Level:
    """One level of a :class:`Hierarchy`: which group each bottom series sums into.

    Parameters
    ----------
    name : str
        Name of the level, e.g. ``"store"``.
    group_idx : np.ndarray
        Integer group of each bottom series, shape ``(n_series,)``, in ``[0, len(labels))``.
    labels : pd.Index
        Label of each group, in group order: a ``pd.MultiIndex`` named by the keys for a level of several keys.
    """

    name: str
    group_idx: np.ndarray
    labels: pd.Index
    _indicator: sparse.csr_array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        group_idx = np.asarray(self.group_idx)
        if group_idx.ndim != 1 or not np.issubdtype(group_idx.dtype, np.integer):
            raise ValueError(f"level {self.name!r}: group_idx must be a 1-D integer array, got {group_idx.dtype}")
        labels = self.labels if isinstance(self.labels, pd.Index) else pd.Index(self.labels)
        if group_idx.size and (group_idx.min() < 0 or group_idx.max() >= len(labels)):
            raise ValueError(f"level {self.name!r}: group_idx must lie in [0, {len(labels)}), one per label")
        n_series = group_idx.size
        indicator = sparse.csr_array(
            (np.ones(n_series), (np.arange(n_series), group_idx)), shape=(n_series, len(labels))
        )
        object.__setattr__(self, "group_idx", group_idx)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "_indicator", indicator)

    @property
    def n_series(self) -> int:
        """Number of bottom series."""
        return int(self.group_idx.size)

    @property
    def n_groups(self) -> int:
        """Number of groups."""
        return len(self.labels)

    def aggregate(self, draws: np.ndarray) -> np.ndarray:
        """Sum the last axis of ``draws``, one entry per bottom series, over the groups.

        Parameters
        ----------
        draws : np.ndarray
            Bottom-level values, shape ``(..., n_series)``.

        Returns
        -------
        np.ndarray
            Sum of each group's series, shape ``(..., n_groups)``, float64.
        """
        draws = np.asarray(draws, dtype=np.float64)
        if draws.ndim == 0 or draws.shape[-1] != self.n_series:
            raise ValueError(f"level {self.name!r}: expected a last axis of {self.n_series} series, got {draws.shape}")
        flat = draws.reshape(-1, self.n_series)
        return np.asarray(flat @ self._indicator).reshape(*draws.shape[:-1], self.n_groups)


@dataclass(frozen=True, eq=False)
class Hierarchy:
    """Named levels that group the same bottom series.

    Parameters
    ----------
    levels : Sequence[Level]
        The levels, in order. They must cover the same number of series and have distinct names.
    """

    levels: tuple[Level, ...]

    def __post_init__(self) -> None:
        levels = tuple(self.levels)
        if not levels:
            raise ValueError("a Hierarchy needs at least one level")
        names = [level.name for level in levels]
        if len(set(names)) != len(names):
            raise ValueError(f"level names must be distinct, got {names}")
        sizes = {level.name: level.n_series for level in levels}
        if len(set(sizes.values())) != 1:
            raise ValueError(f"every level must group the same series, got series counts {sizes}")
        object.__setattr__(self, "levels", levels)

    @classmethod
    def from_frame(cls, meta: pd.DataFrame, levels: Mapping[str, Sequence[str]]) -> Hierarchy:
        """Build a hierarchy from a per-series table.

        Parameters
        ----------
        meta : pd.DataFrame
            One row per bottom series, in the order of the draws' ``series`` axis.
        levels : Mapping[str, Sequence[str]]
            Level name → the columns of ``meta`` whose combinations are its groups, e.g.
            ``{"total": [], "store": ["store_id"], "group": ["store_id", "dept_id"]}``. A level with no columns is
            the total, one group labelled ``"total"``. Groups are sorted by their keys.

        Returns
        -------
        Hierarchy
            The levels, in the order of ``levels``.
        """
        n_series = len(meta)
        built = []
        for name, keys in levels.items():
            keys = list(keys)
            missing = [key for key in keys if key not in meta.columns]
            if missing:
                raise KeyError(f"level {name!r}: columns {missing} are not in meta")
            if not keys:
                built.append(Level(name, np.zeros(n_series, dtype=np.int64), pd.Index([TOTAL_LABEL])))
                continue
            grouped = meta.groupby(keys, sort=True, observed=True, dropna=False)
            group_idx = grouped.ngroup().to_numpy(dtype=np.int64)
            labels = grouped.size().index
            built.append(Level(name, group_idx, labels))
        return cls(tuple(built))

    @property
    def names(self) -> list[str]:
        """Names of the levels, in order."""
        return [level.name for level in self.levels]

    @property
    def n_series(self) -> int:
        """Number of bottom series."""
        return self.levels[0].n_series

    def __getitem__(self, name: str) -> Level:
        for level in self.levels:
            if level.name == name:
                return level
        raise KeyError(f"no level {name!r}; the levels are {self.names}")

    def __iter__(self) -> Iterator[Level]:
        return iter(self.levels)

    def __len__(self) -> int:
        return len(self.levels)


def aggregate_draws(
    draws: np.ndarray | xr.DataArray | xr.Dataset,
    hierarchy: Hierarchy,
    level: str | None = None,
) -> np.ndarray | xr.DataArray | dict[str, np.ndarray | xr.DataArray]:
    """Sum bottom-level draws over the groups of a level, draw by draw.

    Parameters
    ----------
    draws : np.ndarray | xr.DataArray | xr.Dataset
        Bottom-level draws: an array of shape ``(sample, time, series)`` (any leading shape; the series are the last
        axis), a ``DataArray`` with a ``series`` dim, or a dataset holding ``forecast_samples``, as ``SSP.predict``
        returns. The series are in the order of the rows ``hierarchy`` was built from.
    hierarchy : Hierarchy
        Groups of the bottom series.
    level : str | None
        The level to aggregate to, by default all of them.

    Returns
    -------
    np.ndarray | xr.DataArray | dict
        For an array, the sums of shape ``(sample, time, group)``; for a ``DataArray`` or dataset, a ``DataArray``
        whose ``series`` dim is replaced by a trailing ``group`` dim labelled with the level's groups. With
        ``level=None``, a dict of these keyed by level name, in the hierarchy's order.
    """
    if isinstance(draws, xr.Dataset):
        if DRAWS_VAR not in draws:
            raise KeyError(f"the dataset has no {DRAWS_VAR!r} variable; its variables are {list(draws.data_vars)}")
        draws = draws[DRAWS_VAR]
    if level is not None:
        return _aggregate_level(draws, hierarchy[level])
    return {lvl.name: _aggregate_level(draws, lvl) for lvl in hierarchy}


def _aggregate_level(draws: np.ndarray | xr.DataArray, level: Level) -> np.ndarray | xr.DataArray:
    """Sum ``draws`` over the groups of ``level``, keeping a ``DataArray``'s other dims and coords."""
    if not isinstance(draws, xr.DataArray):
        return level.aggregate(draws)
    if SERIES_DIM not in draws.dims:
        raise ValueError(f"the draws have no {SERIES_DIM!r} dim; their dims are {draws.dims}")
    draws = draws.transpose(..., SERIES_DIM)
    sums = level.aggregate(draws.values)
    dims = (*draws.dims[:-1], GROUP_DIM)
    coords = {name: coord for name, coord in draws.coords.items() if SERIES_DIM not in coord.dims}
    out = xr.DataArray(sums, dims=dims, coords=coords, name=draws.name)
    if isinstance(level.labels, pd.MultiIndex):
        return out.assign_coords(xr.Coordinates.from_pandas_multiindex(level.labels, GROUP_DIM))
    return out.assign_coords({GROUP_DIM: level.labels.to_numpy()})


def draw_quantiles(
    draws: np.ndarray | xr.DataArray | xr.Dataset, quantiles: float | Sequence[float]
) -> np.ndarray | xr.DataArray:
    """Quantiles of draws over the ``sample`` axis.

    Parameters
    ----------
    draws : np.ndarray | xr.DataArray | xr.Dataset
        Draws with the samples on the first axis, e.g. ``(sample, time, group)`` from :func:`aggregate_draws`; a
        ``DataArray`` with a ``sample`` dim; or a dataset holding ``forecast_samples``.
    quantiles : float | Sequence[float]
        Quantile levels in ``[0, 1]``.

    Returns
    -------
    np.ndarray | xr.DataArray
        For an array, ``np.quantile(draws, quantiles, axis=0)``: shape ``(quantile, ...)``, without the leading axis
        for a scalar ``quantiles``. For a ``DataArray`` or dataset, ``DataArray.quantile`` over ``sample``, with a
        leading ``quantile`` dim.
    """
    if isinstance(draws, xr.Dataset):
        if DRAWS_VAR not in draws:
            raise KeyError(f"the dataset has no {DRAWS_VAR!r} variable; its variables are {list(draws.data_vars)}")
        draws = draws[DRAWS_VAR]
    if isinstance(draws, xr.DataArray):
        if SAMPLE_DIM not in draws.dims:
            raise ValueError(f"the draws have no {SAMPLE_DIM!r} dim; their dims are {draws.dims}")
        return draws.quantile(quantiles, dim=SAMPLE_DIM)
    return np.quantile(np.asarray(draws), quantiles, axis=0)
