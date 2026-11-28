"""NumPyro building blocks for pooled models of many count series.

The ideas come from Juan Orduz's hierarchical state space model of the M5 series (the notebook "M5 forecasting II",
https://juanitorduz.github.io/numpyro_forecast/docs/examples/m5_hierarchical_state_space.html). This module is
bunobee's own code for them; nothing is copied from the notebook. Two of the ideas carry the model:

- :func:`origin_level` anchors each series' level at its recent mean, and :func:`block_walk` lets that level move
  over blocks of days, ending at zero on the last training block. The level at the forecast origin is then the
  intercept, not the end of a long walk.
- The walk moves once per block, not once per day. A daily walk fitted by mean-field SVI absorbs the day-to-day noise
  as level; a walk over blocks cannot.

:func:`pooled_effect` adds a zero-sum categorical effect, e.g. the weekday, shared by the series of a group, and
:func:`group_shocks` adds day-to-day shocks shared by the series of a group, zero-sum within each period.

Conventions
-----------
- Time runs along the first axis of every array a block returns and series along the last: ``(time, series)``.
- Every block runs under ``numpyro.handlers.scope(prefix=name)``, so its sites are named ``"<name>/<site>"`` and
  two uses with different names never clash. The plates a block opens are named ``"<name>/<plate>"`` by hand: the
  scope leaves plate names alone (``hide_types=["plate"]``), because renaming the frames of the caller's series
  plate would hide it from an autoguide that subsamples it.
- Groups are given as an integer index of each series' group or as a :class:`~bunobee.hierarchy.Level`.
- A block with series-level sites takes the model's series plate as ``series``, e.g. ``numpyro.plate("series", n)``,
  subsampled or not, and samples them under it. Per-series inputs (a group index, an array of scales) cover all the
  series; the block subsamples them to the plate's batch. :func:`pooled_effect` and :func:`group_shocks` sample
  group-level sites, so call them outside the series plate. :func:`block_walk` samples no plate over series; call it
  inside the series plate with a per-series ``scale``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers

from bunobee.hierarchy import Level

if TYPE_CHECKING:
    from jax import Array

__all__ = ["Horizon", "block_index", "block_walk", "group_shocks", "origin_level", "pooled_effect"]


@dataclass(frozen=True)
class Horizon:
    """The train / forecast split of one model call.

    Parameters
    ----------
    n_obs : int
        Number of training days, which end at the forecast origin.
    future : int
        Number of forecast days after the origin, ``0`` while fitting.
    """

    n_obs: int
    future: int = 0

    def __post_init__(self) -> None:
        for field_name in ("n_obs", "future"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative int, got {value!r}")
            object.__setattr__(self, field_name, int(value))

    @property
    def duration(self) -> int:
        """Number of days in all, ``n_obs + future``."""
        return self.n_obs + self.future


def _scope(name: str) -> handlers.scope:
    """Prefix the sample and deterministic sites with ``name``, leaving the plate frames as they are."""
    return handlers.scope(prefix=name, divider="/", hide_types=["plate"])


def _group_index(groups: Array | np.ndarray | Level) -> tuple[jnp.ndarray, int]:
    """The integer group of every series and the number of groups."""
    if isinstance(groups, Level):
        return jnp.asarray(groups.group_idx, dtype=jnp.int32), groups.n_groups
    index = np.asarray(groups)
    if index.ndim != 1 or not np.issubdtype(index.dtype, np.integer):
        raise ValueError(f"groups must be a 1-D integer array or a Level, got shape {index.shape} of {index.dtype}")
    if index.size and index.min() < 0:
        raise ValueError("groups must be non-negative")
    return jnp.asarray(index, dtype=jnp.int32), int(index.max()) + 1 if index.size else 0


def _batch(values: Array | float, series: numpyro.plate | None) -> Array | float:
    """``values`` over all the series, cut to the batch of the ``series`` plate; scalars pass through."""
    if series is None or jnp.ndim(values) == 0:
        return values
    with series:
        return numpyro.subsample(values, event_dim=jnp.ndim(values) - 1)


def block_index(horizon: Horizon, block_len: int, n_blocks: int) -> Array:
    """Block of every day, numbered so that the last training block is ``n_blocks - 1``.

    Blocks are ``block_len`` days long and end at the forecast origin. Days earlier than ``n_blocks`` blocks before
    the origin share block ``0``, and every forecast day is in block ``n_blocks``.

    Parameters
    ----------
    horizon : Horizon
        The train / forecast split.
    block_len : int
        Days per block.
    n_blocks : int
        Number of training blocks.

    Returns
    -------
    Array
        Integer block of each day, shape ``(horizon.duration,)``, in ``[0, n_blocks]``.
    """
    if block_len < 1 or n_blocks < 1:
        raise ValueError(f"block_len and n_blocks must be at least 1, got {block_len} and {n_blocks}")
    t = jnp.arange(horizon.duration)
    return jnp.clip((t - horizon.n_obs) // block_len + n_blocks, 0, n_blocks)


def block_walk(name: str, scale: Array | float, horizon: Horizon, block_len: int, n_blocks: int) -> Array:
    """Random walk over blocks of days, zero on the last training block.

    The walk takes one ``Normal(0, scale)`` step between consecutive training blocks. It is pinned to zero on the
    last training block, so each earlier block is minus the sum of the steps after it, and an intercept added to the
    walk is the level at the forecast origin. With ``horizon.future > 0`` one more step moves the forecast block.

    Sites: ``"<name>/step"``, shape ``(n_blocks - 1, *batch)``, and with a forecast ``"<name>/future"``, shape
    ``(1, *batch)``, under the plates ``"<name>/blocks"`` and ``"<name>/future_blocks"`` at ``dim=-2``. ``batch`` is
    the shape of ``scale`` and of the enclosing plates, e.g. the series plate.

    Parameters
    ----------
    name : str
        Prefix of the block's sites and plates.
    scale : Array | float
        Standard deviation of a step, a scalar or one per series, shape ``(n_series,)``.
    horizon : Horizon
        The train / forecast split.
    block_len : int
        Days per block.
    n_blocks : int
        Number of training blocks; days before them share the first block.

    Returns
    -------
    Array
        The walk on every day, shape ``(horizon.duration, *batch)``: constant within a block, zero on the last
        ``block_len`` training days.
    """
    block = block_index(horizon, block_len, n_blocks)
    with _scope(name):
        if n_blocks > 1:
            with numpyro.plate(f"{name}/blocks", n_blocks - 1, dim=-2):
                step = numpyro.sample("step", dist.Normal(0.0, scale))
            # Block k is minus the sum of the steps after it; the last training block is 0.
            before = -jnp.flip(jnp.cumsum(jnp.flip(step, 0), 0), 0)
            walk = jnp.concatenate([before, jnp.zeros_like(step[:1])], axis=0)
        else:
            walk = jnp.zeros((1, *jnp.shape(scale)))
        if horizon.future > 0:
            with numpyro.plate(f"{name}/future_blocks", 1, dim=-2):
                future = numpyro.sample("future", dist.Normal(0.0, scale))
            walk = jnp.concatenate([walk, future], axis=0)
    return walk[block]


def pooled_effect(
    name: str,
    index: Array | np.ndarray,
    n_levels: int,
    group: Array | np.ndarray | Level | None = None,
    scale: float = 0.3,
    deviation_scale: Array | float | None = None,
    series: numpyro.plate | None = None,
) -> Array:
    """Zero-sum categorical effect, e.g. of the weekday or the day of the month, pooled over groups of series.

    With ``group`` given, each group has one effect, ``ZeroSumNormal(scale)`` over the ``n_levels`` levels, which its
    series share. With ``deviation_scale`` each series adds its own zero-sum deviation, ``deviation_scale`` times a
    ``ZeroSumNormal(1)``. With ``group=None`` each series has its own effect, ``ZeroSumNormal(scale)``.

    Sites: ``"<name>/group"``, shape ``(n_groups, n_levels)``, under the plate ``"<name>/groups"``;
    ``"<name>/deviation"`` (with ``deviation_scale``) or ``"<name>/series"`` (with ``group=None``), shape
    ``(n_series, n_levels)``, under ``series``. Call it outside the series plate.

    Parameters
    ----------
    name : str
        Prefix of the block's sites and plates.
    index : Array | np.ndarray
        Level of each day, integers in ``[0, n_levels)``, shape ``(n_days,)``: e.g. the weekday of every day.
    n_levels : int
        Number of levels, e.g. 7 for the weekday.
    group : Array | np.ndarray | Level | None
        Group of every series, shape ``(n_series,)``, or a hierarchy level. ``None`` gives one effect per series.
    scale : float
        Scale of the group effect, or of the series effect with ``group=None``.
    deviation_scale : Array | float | None
        Scale of each series' deviation from its group's effect, a scalar or one per series, shape ``(n_series,)``;
        ``None`` for no deviation. Ignored with ``group=None``.
    series : numpyro.plate | None
        The model's plate over series, possibly subsampled. Required with ``group=None`` or ``deviation_scale``; by
        default the block opens ``"<name>/series"`` over the series of ``group``.

    Returns
    -------
    Array
        The effect on every day for every series of the batch, shape ``(n_days, n_batch)``. It sums to zero over the
        days of any run of ``index`` that takes each level once, e.g. a week of weekdays.
    """
    index = jnp.asarray(index, dtype=jnp.int32)
    with _scope(name):
        if group is None:
            if series is None:
                raise ValueError("pooled_effect with group=None needs the series plate, series=")
            with series:
                effect = numpyro.sample("series", dist.ZeroSumNormal(scale, event_shape=(n_levels,)))
            return effect[:, index].T

        group_idx, n_groups = _group_index(group)
        if series is None:
            series = numpyro.plate(f"{name}/series", group_idx.shape[0])
        with numpyro.plate(f"{name}/groups", n_groups):
            group_effect = numpyro.sample("group", dist.ZeroSumNormal(scale, event_shape=(n_levels,)))
        effect = group_effect[_batch(group_idx, series)]
        if deviation_scale is not None:
            dev_scale = _batch(jnp.asarray(deviation_scale), series)
            with series:
                deviation = numpyro.sample("deviation", dist.ZeroSumNormal(1.0, event_shape=(n_levels,)))
            effect = effect + jnp.expand_dims(dev_scale, -1) * deviation
    return effect[:, index].T


def group_shocks(
    name: str,
    scale: Array | float,
    horizon: Horizon,
    groups: Array | np.ndarray | Level,
    period: int = 7,
    series: numpyro.plate | None = None,
) -> Array:
    """Daily shocks shared by the series of a group, summing to zero within each period.

    The training days are cut into periods of ``period`` days that end at the forecast origin (the first may be
    partial), and each group draws ``scale`` times a ``ZeroSumNormal(1)`` over every period. A zero-sum period cannot
    shift the level, so the shocks carry only the day-to-day variation the group's series share. The forecast days
    get fresh periods, starting at the origin.

    Sites: ``"<name>/shock"``, shape ``(n_periods, n_groups, period)``, and with a forecast ``"<name>/future"``, under
    the plates ``"<name>/periods"`` (``dim=-2``) and ``"<name>/groups"`` (``dim=-1``). Call it outside the series plate.

    Parameters
    ----------
    name : str
        Prefix of the block's sites and plates.
    scale : Array | float
        Scale of the shocks, a scalar or one per group, shape ``(n_groups,)``.
    horizon : Horizon
        The train / forecast split.
    groups : Array | np.ndarray | Level
        Group of every series, shape ``(n_series,)``, or a hierarchy level.
    period : int
        Days per zero-sum period, e.g. 7 for a week.
    series : numpyro.plate | None
        The model's plate over series, possibly subsampled, to return the shocks of its batch only. By default all
        the series.

    Returns
    -------
    Array
        The shock on every day for every series of the batch, shape ``(horizon.duration, n_batch)``: the same for
        the series of a group.
    """
    if period < 1:
        raise ValueError(f"period must be at least 1, got {period}")
    group_idx, n_groups = _group_index(groups)
    scale = jnp.asarray(scale)
    n_periods = -(-horizon.n_obs // period)
    with _scope(name), numpyro.plate(f"{name}/groups", n_groups, dim=-1):
        with numpyro.plate(f"{name}/periods", n_periods, dim=-2):
            shock = numpyro.sample("shock", dist.ZeroSumNormal(1.0, event_shape=(period,)))
        if horizon.future > 0:
            with numpyro.plate(f"{name}/future_periods", -(-horizon.future // period), dim=-2):
                future = numpyro.sample("future", dist.ZeroSumNormal(1.0, event_shape=(period,)))
            shock = jnp.concatenate([shock, future], axis=0)

    # (periods, groups, period) -> (periods * period, groups), consecutive days; then drop the leading days of the
    # partial first period so that the periods end at the origin.
    days = jnp.swapaxes(jnp.expand_dims(scale, -1) * shock, 1, 2).reshape(-1, n_groups)
    days = days[n_periods * period - horizon.n_obs :][: horizon.duration]
    return days[:, _batch(group_idx, series)]


def origin_level(
    y: Array | np.ndarray,
    gate: Array | np.ndarray | None = None,
    days: int = 28,
    fallback_days: int | None = None,
    floor: float = 0.05,
) -> Array:
    """Log of every series' mean over its listed days just before the forecast origin.

    The mean is taken over the days of the last ``days`` on which ``gate`` lists the series. A series not listed on
    any of them falls back to the last ``fallback_days``, and a series listed on none of those to a mean of zero.
    ``floor`` is added to every mean before the log, so that a series with a zero mean gets ``log(floor)``.

    Parameters
    ----------
    y : Array | np.ndarray
        Training counts, shape ``(n_days, n_series)``, ending at the origin.
    gate : Array | np.ndarray | None
        ``1`` on the days a series is listed (can sell), ``0`` otherwise, shape of ``y``. ``None`` lists every day.
    days : int
        Length of the recent window.
    fallback_days : int | None
        Length of the fallback window; ``None`` for all of ``y``.
    floor : float
        Added to the mean before the log.

    Returns
    -------
    Array
        The log level of each series, shape ``(n_series,)``.
    """
    y = jnp.asarray(y, dtype=jnp.float32)
    gate = jnp.ones_like(y) if gate is None else jnp.asarray(gate, dtype=y.dtype)
    if y.ndim != 2 or gate.shape != y.shape:
        raise ValueError(f"y must be (n_days, n_series) and gate of the same shape, got {y.shape} and {gate.shape}")
    if days < 1 or (fallback_days is not None and fallback_days < 1):
        raise ValueError(f"days and fallback_days must be at least 1, got {days} and {fallback_days}")

    def window_mean(n: int) -> tuple[Array, Array]:
        listed = (gate[-n:] > 0).astype(y.dtype)
        n_listed = listed.sum(0)
        return (y[-n:] * listed).sum(0) / jnp.maximum(n_listed, 1.0), n_listed

    recent, n_recent = window_mean(days)
    fallback, n_fallback = window_mean(y.shape[0] if fallback_days is None else fallback_days)
    mean = jnp.where(n_recent > 0, recent, jnp.where(n_fallback > 0, fallback, 0.0))
    return jnp.log(mean + floor)
