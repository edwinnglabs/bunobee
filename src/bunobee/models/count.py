"""A pooled count model of many series: :class:`HierarchicalCount`, built from terms on the log scale.

The model is the general form of Juan Orduz's hierarchical state space model of the M5 series (the notebook "M5
forecasting II", https://juanitorduz.github.io/numpyro_forecast/docs/examples/m5_hierarchical_state_space.html),
assembled from the blocks of :mod:`bunobee.models.blocks`, the families of :mod:`bunobee.models.obs` and the
inference of :mod:`bunobee.models.inference`. For ``n_series`` count series observed over ``n_obs`` days::

    eta[t, i]  = level[t, i] + sum of the components' terms[t, i]
    mean[t, i] = exp(clip(eta[t, i], *log_mean_bounds))
    y[t, i]    ~ obs.dist(mean[t, i], params[i], gate[t, i])

The level is the series' recent mean at the forecast origin plus an intercept, an optional shift per group and an
optional walk over blocks of days that is zero on the last training block (:class:`Level`). The components add their
terms: a seasonal pattern (:class:`Seasonal`), the day of the month (:class:`DayOfMonth`), events (:class:`Events`),
Fourier harmonics (:class:`Fourier`), shocks shared by a group (:class:`GroupShocks`) and covariates such as SNAP
(:class:`Regressor`).

Components
----------
A component is any object with a ``name``, the covariates it reads (``keys``) and ``contribute(ctx)``, which samples
its sites and returns its term, shape ``(rows, n_batch)``: the days of :attr:`Context.rows` and the series of the
batch of the series plate. ``prepare(covariates)``, called by :meth:`HierarchicalCount.fit` with the training
covariates, returns the component with whatever it sizes from the data (the number of events), and
``check(covariates)`` checks the forecast covariates; both are optional. A component's sites are named
``"<name>/<site>"``.

Pooling
-------
A component's ``pool_by`` names a level of the model's :class:`~bunobee.hierarchy.Hierarchy` (or is a
:class:`~bunobee.hierarchy.Level`): the series of a group share the effect. ``pool_by=None`` gives each series its
own. A ``scale_by`` names the level whose groups share the scale of a term, ``HalfNormal(scale)``; ``None`` gives one
scale for all the series.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import xarray as xr
from numpyro import handlers
from numpyro.infer import Predictive, init_to_median

from bunobee.hierarchy import DRAWS_VAR, SAMPLE_DIM, SERIES_DIM, Hierarchy
from bunobee.hierarchy import Level as HierarchyLevel
from bunobee.models.base import Forecaster
from bunobee.models.blocks import Horizon, _batch, _group_index, block_walk, group_shocks, origin_level, pooled_effect
from bunobee.models.inference import MAP, NUTS, SVI, Inference
from bunobee.models.obs import NegBin2, ObsFamily

__all__ = [
    "BlockWalk",
    "Component",
    "Context",
    "DayOfMonth",
    "Events",
    "Fourier",
    "GroupShocks",
    "HierarchicalCount",
    "Level",
    "LinearOneCycle",
    "RecentMean",
    "Regressor",
    "Seasonal",
]

#: A level of the hierarchy, by name or as a :class:`~bunobee.hierarchy.Level`, or ``None``.
PoolBy = str | HierarchyLevel | None

#: Series per SVI step of the default inference; a panel of fewer series is fitted whole.
DEFAULT_SUBSAMPLE = 1_500

#: SVI steps of the default inference, those of Orduz's M5 run.
DEFAULT_STEPS = 3_000

#: Name of the site of the observations, and the prefix of the observation family's sites.
OBS_SITE = "y"
OBS_NAME = "obs"

#: The deterministic site of the forecast-only pass: the mean over the forecast days, before the gate.
MEAN_FUTURE_SITE = "mean_future"


# ---------------------------------------------------------------------------------------------------------------------
# The context a component reads
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Context:
    """What a component reads in one call of the model.

    Attributes
    ----------
    horizon : Horizon
        The train / forecast split: ``future=0`` while fitting.
    rows : slice
        The days the terms cover: every day while fitting, the forecast days in the forecast-only pass.
    covariates : Mapping[str, Array]
        The covariates over every day of ``horizon``, ``(duration,)`` or ``(duration, n_series)`` each.
    series : numpyro.plate
        The model's plate over series, possibly subsampled.
    offset : Array
        The log level of every series at the forecast origin, shape ``(n_series,)``.
    hierarchy : Hierarchy | None
        The hierarchy the components' ``pool_by`` names refer to.
    """

    horizon: Horizon
    rows: slice
    covariates: Mapping[str, Any]
    series: Any
    offset: Any
    hierarchy: Hierarchy | None = None

    @property
    def n_series(self) -> int:
        """Number of series, all of them, not only the batch."""
        return int(self.offset.shape[0])

    @property
    def n_rows(self) -> int:
        """Number of rows, the days the terms cover."""
        return len(range(*self.rows.indices(self.horizon.duration)))

    @property
    def t(self) -> jax.Array:
        """Day of each row, counted from the first training day, shape ``(rows,)``."""
        return jnp.arange(self.horizon.duration)[self.rows]

    def level(self, spec: PoolBy) -> HierarchyLevel | None:
        """The hierarchy level ``spec`` names, or ``spec`` itself when it is a level; ``None`` stays ``None``."""
        if spec is None:
            return None
        if isinstance(spec, HierarchyLevel):
            level = spec
        elif self.hierarchy is None:
            raise ValueError(f"pool_by={spec!r} names a hierarchy level, but the model has no hierarchy=")
        else:
            level = self.hierarchy[spec]
        if level.n_series != self.n_series:
            raise ValueError(f"level {level.name!r} groups {level.n_series} series but the model has {self.n_series}")
        return level

    def batch(self, values: Any) -> Any:
        """``values`` over all the series, shape ``(n_series, ...)``, cut to the batch; call it outside the plate."""
        return _batch(values, self.series)

    def covariate(self, key: str) -> jax.Array:
        """Covariate ``key`` over the rows, ``(rows,)``, or ``(rows, n_batch)`` for one with a series axis."""
        values = jnp.asarray(self.covariates[key])[self.rows]
        if values.ndim == 1:
            return values
        with self.series:
            return numpyro.subsample(values, event_dim=0)


@runtime_checkable
class Component(Protocol):
    """A term of the log mean of :class:`HierarchicalCount`."""

    #: Prefix of the component's sites; distinct within a model.
    name: str

    @property
    def keys(self) -> tuple[str, ...]:
        """The covariates the component reads."""
        ...

    def contribute(self, ctx: Context) -> jax.Array:
        """Sample the component's sites and return its term on the log scale, shape ``(rows, n_batch)``."""
        ...


def _group_scale(site: str, scale: float, level: HierarchyLevel | None) -> Any:
    """``HalfNormal(scale)`` per group of ``level``, returned per series, ``(n_series,)``; one scalar with ``None``."""
    if level is None:
        return numpyro.sample(site, dist.HalfNormal(scale))
    group_idx, n_groups = _group_index(level)
    with numpyro.plate(f"{site}_groups", n_groups):
        value = numpyro.sample(site, dist.HalfNormal(scale))
    return value[group_idx]


def _group_effect(site: str, prior: dist.Distribution, level: HierarchyLevel | None, ctx: Context) -> jax.Array:
    """One draw of ``prior`` per group of ``level`` (or per series with ``None``), returned for the batch's series."""
    if level is None:
        with ctx.series:
            return numpyro.sample(site, prior)
    group_idx, n_groups = _group_index(level)
    with numpyro.plate(f"{site}_groups", n_groups):
        value = numpyro.sample(site, prior)
    return value[ctx.batch(group_idx)]


def _check_positive(owner: str, **values: float) -> None:
    for key, value in values.items():
        if not value > 0:
            raise ValueError(f"{owner}: {key} must be positive, got {value}")


def _check_index(owner: str, key: str, values: np.ndarray, low: int, high: int) -> None:
    """Check covariate ``key`` holds integers in ``[low, high]``, one per day, the same for every series."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"{owner}: covariate {key!r} must be one index per day, shape (n_days,), got {values.shape}")
    if values.size and (not np.all(np.mod(values, 1) == 0) or values.min() < low or values.max() > high):
        raise ValueError(f"{owner}: covariate {key!r} must hold integers in [{low}, {high}]")


# ---------------------------------------------------------------------------------------------------------------------
# The level
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RecentMean:
    """The anchor of the level: the log of each series' mean over its listed days just before the origin.

    See :func:`~bunobee.models.blocks.origin_level`.

    Parameters
    ----------
    days : int
        Length of the recent window, by default 28.
    fallback_days : int | None
        Length of the window of a series not listed in the recent one; ``None`` for all the training days.
    floor : float
        Added to every mean before the log, by default 0.05.
    """

    days: int = 28
    fallback_days: int | None = None
    floor: float = 0.05

    def __call__(self, y: np.ndarray, gate: np.ndarray) -> np.ndarray:
        """The log level of each series, ``(n_series,)``, from the training counts and gate ``(n_obs, n_series)``."""
        return np.asarray(origin_level(y, gate, self.days, self.fallback_days, self.floor))


@dataclass(frozen=True)
class BlockWalk:
    """The walk of the level: a random walk over blocks of ``block`` days, zero on the last training block.

    See :func:`~bunobee.models.blocks.block_walk`. A step's scale is ``HalfNormal(scale)``, one per group of
    ``scale_by``.

    Parameters
    ----------
    block : int
        Days per block, by default 28.
    n_blocks : int
        Training blocks; the days before them share the first. By default 13, a year of 28-day blocks.
    scale_by : str | Level | None
        The level whose groups share the step scale; ``None`` for one for all.
    scale : float
        Scale of the ``HalfNormal`` prior on the step scale, by default 0.3.
    """

    block: int = 28
    n_blocks: int = 13
    scale_by: PoolBy = None
    scale: float = 0.3

    def __post_init__(self) -> None:
        if self.block < 1 or self.n_blocks < 1:
            raise ValueError(f"BlockWalk: block and n_blocks must be at least 1, got {self.block} and {self.n_blocks}")
        _check_positive("BlockWalk", scale=self.scale)


@dataclass(frozen=True)
class Level:
    """The level of every series: its recent mean, an intercept, a shift per group and a walk over blocks.

    The level of series ``i`` on day ``t`` is ``anchor[i] + intercept[i] + shift[group(i)] + walk[t, i]``:

    - ``anchor`` is fixed from the training counts by ``anchor``, e.g. :class:`RecentMean`;
    - ``intercept[i] ~ Normal(0, sigma)``, with ``sigma ~ HalfNormal(scale)`` per group of ``scale_by``;
    - ``shift[g] ~ Normal(0, shift_scale)`` per group of ``shift_by``, none with ``shift_by=None``;
    - ``walk`` is :class:`BlockWalk`, zero on the last training block, none with ``walk=None``.

    The walk is zero at the origin, so the level there is the anchor plus the intercept and the shift.

    Sites: ``"<name>/intercept_scale"``, ``"<name>/intercept"``, ``"<name>/shift"``, ``"<name>/walk_scale"`` and the
    walk's ``"<name>/walk/step"`` and ``"<name>/walk/future"``.

    Parameters
    ----------
    anchor : RecentMean
        The log level at the origin around which the intercept is centred, a callable of ``(y, gate)``.
    walk : BlockWalk | None
        The walk over blocks, by default ``BlockWalk()``.
    scale_by : str | Level | None
        The level whose groups share the intercept's scale; ``None`` for one for all.
    scale : float
        Scale of the ``HalfNormal`` prior on the intercept's scale, by default 0.5.
    shift_by : str | Level | None
        The level whose groups shift their series' level together; by default ``None``, no shift.
    shift_scale : float
        Scale of the ``Normal`` prior on a group's shift, by default 0.3.
    name : str
        Prefix of the sites, by default ``"level"``.
    """

    anchor: Any = field(default_factory=RecentMean)
    walk: BlockWalk | None = field(default_factory=BlockWalk)
    scale_by: PoolBy = None
    scale: float = 0.5
    shift_by: PoolBy = None
    shift_scale: float = 0.3
    name: str = "level"

    def __post_init__(self) -> None:
        _check_positive("Level", scale=self.scale, shift_scale=self.shift_scale)

    @property
    def keys(self) -> tuple[str, ...]:
        """The level reads no covariates."""
        return ()

    def contribute(self, ctx: Context) -> jax.Array:
        """The level on the rows, ``(rows, n_batch)``."""
        name = self.name
        sigma = ctx.batch(_group_scale(f"{name}/intercept_scale", self.scale, ctx.level(self.scale_by)))
        term = ctx.batch(ctx.offset)
        shift_level = ctx.level(self.shift_by)
        if shift_level is not None:
            term = term + _group_effect(f"{name}/shift", dist.Normal(0.0, self.shift_scale), shift_level, ctx)
        walk_scale = None
        if self.walk is not None:
            walk_scale = _group_scale(f"{name}/walk_scale", self.walk.scale, ctx.level(self.walk.scale_by))
            walk_scale = ctx.batch(walk_scale)
        with ctx.series:
            term = (term + numpyro.sample(f"{name}/intercept", dist.Normal(0.0, sigma)))[None]
            if self.walk is not None:
                walk = block_walk(f"{name}/walk", walk_scale, ctx.horizon, self.walk.block, self.walk.n_blocks)
                term = term + walk[ctx.rows]
        return term


# ---------------------------------------------------------------------------------------------------------------------
# The components
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Seasonal:
    """A zero-sum seasonal pattern of ``period`` days, e.g. the weekday, per series or pooled over a level.

    With ``pool_by=None`` (the default) each series has its own pattern, ``ZeroSumNormal(scale)``. With ``pool_by``
    the series of a group share one, and with ``deviation`` each series adds its own zero-sum deviation, scaled by
    ``HalfNormal(deviation_scale)`` per group of ``deviation``. See :func:`~bunobee.models.blocks.pooled_effect`.

    Sites: ``"<name>/series"``, or ``"<name>/group"`` with ``"<name>/deviation_scale"`` and ``"<name>/deviation"``.

    Parameters
    ----------
    period : int
        Days per cycle, by default 7.
    pool_by : str | Level | None
        The level whose groups share the pattern; ``None`` for one per series.
    deviation : str | Level | None
        The level whose groups share the scale of the series' deviations from their group's pattern; needs
        ``pool_by``. By default ``None``, no deviations.
    scale : float
        Scale of the pattern, by default 0.3.
    deviation_scale : float
        Scale of the ``HalfNormal`` prior on the deviations' scale, by default 0.2.
    index : str | None
        The covariate holding each day's position in the cycle, integers in ``[0, period)``, e.g. the weekday. By
        default ``None``: the day's number since the first training day, modulo ``period``.
    name : str
        Prefix of the sites, by default ``"seasonal"``.
    """

    period: int = 7
    pool_by: PoolBy = None
    deviation: PoolBy = None
    scale: float = 0.3
    deviation_scale: float = 0.2
    index: str | None = None
    name: str = "seasonal"

    def __post_init__(self) -> None:
        if not (isinstance(self.period, (int, np.integer)) and self.period >= 2):
            raise ValueError(f"Seasonal: period must be an int of at least 2, got {self.period!r}")
        if self.deviation is not None and self.pool_by is None:
            raise ValueError("Seasonal: deviation needs pool_by, the groups whose pattern the series deviate from")
        _check_positive("Seasonal", scale=self.scale, deviation_scale=self.deviation_scale)

    @property
    def keys(self) -> tuple[str, ...]:
        """``(index,)``, or nothing when the position is counted from the first training day."""
        return () if self.index is None else (self.index,)

    def check(self, covariates: Mapping[str, np.ndarray]) -> None:
        """Check the index covariate lies in ``[0, period)``."""
        if self.index is not None:
            _check_index(self.name, self.index, covariates[self.index], 0, self.period - 1)

    def prepare(self, covariates: Mapping[str, np.ndarray]) -> Seasonal:
        """Check the training covariates; nothing to size."""
        self.check(covariates)
        return self

    def contribute(self, ctx: Context) -> jax.Array:
        """The pattern on the rows, ``(rows, n_batch)``."""
        index = ctx.t % self.period if self.index is None else ctx.covariate(self.index)
        deviation_scale = None
        if self.deviation is not None:
            deviation_scale = _group_scale(
                f"{self.name}/deviation_scale", self.deviation_scale, ctx.level(self.deviation)
            )
        return pooled_effect(
            self.name,
            index,
            self.period,
            group=ctx.level(self.pool_by),
            scale=self.scale,
            deviation_scale=deviation_scale,
            series=ctx.series,
        )


@dataclass(frozen=True)
class DayOfMonth:
    """A zero-sum effect of the day of the month, per series or pooled over a level.

    Sites: ``"<name>/series"`` or ``"<name>/group"``; see :func:`~bunobee.models.blocks.pooled_effect`.

    Parameters
    ----------
    pool_by : str | Level | None
        The level whose groups share the effect; ``None`` for one per series.
    scale : float
        Scale of the ``ZeroSumNormal`` effect over the 31 days, by default 0.2.
    key : str
        The covariate holding each day's day of the month, ``1`` to ``31`` (``DatetimeIndex.day``). By default
        ``"day_of_month"``.
    name : str
        Prefix of the sites, by default ``"day_of_month"``.
    """

    pool_by: PoolBy = None
    scale: float = 0.2
    key: str = "day_of_month"
    name: str = "day_of_month"

    N_DAYS: ClassVar[int] = 31

    def __post_init__(self) -> None:
        _check_positive("DayOfMonth", scale=self.scale)

    @property
    def keys(self) -> tuple[str, ...]:
        """``(key,)``."""
        return (self.key,)

    def check(self, covariates: Mapping[str, np.ndarray]) -> None:
        """Check the day of the month lies in ``[1, 31]``."""
        _check_index(self.name, self.key, covariates[self.key], 1, self.N_DAYS)

    def prepare(self, covariates: Mapping[str, np.ndarray]) -> DayOfMonth:
        """Check the training covariates; nothing to size."""
        self.check(covariates)
        return self

    def contribute(self, ctx: Context) -> jax.Array:
        """The effect on the rows, ``(rows, n_batch)``."""
        index = ctx.covariate(self.key).astype(jnp.int32) - 1
        return pooled_effect(
            self.name, index, self.N_DAYS, group=ctx.level(self.pool_by), scale=self.scale, series=ctx.series
        )


@dataclass(frozen=True)
class Events:
    """Effects of named events, e.g. holidays, per series or pooled over a level.

    Each event covariate holds ``0`` on a day without an event and ``k`` in ``[1, n_events]`` on a day of event
    ``k``. Each group of ``pool_by`` (or each series) draws the scale of its effects, ``HalfNormal(scale)``, and an
    effect of every event, ``Normal(0, its scale)``; no event has no effect. The term is the sum of the effects of
    the events of every covariate in ``keys``, so two columns of events on the same day add up.

    Sites: ``"<name>/scale"`` and ``"<name>/effect"``, shape ``(..., n_events)``.

    Parameters
    ----------
    pool_by : str | Level | None
        The level whose groups share the effects; ``None`` for one set per series.
    keys : Sequence[str]
        The event covariates, by default ``("event",)``.
    n_events : int | None
        Number of events. By default ``None``, the largest index of the training covariates; a forecast day may then
        only hold events seen in training.
    scale : float
        Scale of the ``HalfNormal`` prior on the effects' scale, by default 0.3.
    name : str
        Prefix of the sites, by default ``"events"``.
    """

    pool_by: PoolBy = None
    keys: tuple[str, ...] = ("event",)
    n_events: int | None = None
    scale: float = 0.3
    name: str = "events"

    def __post_init__(self) -> None:
        keys = (self.keys,) if isinstance(self.keys, str) else tuple(self.keys)
        if not keys:
            raise ValueError("Events: keys must name at least one covariate")
        object.__setattr__(self, "keys", keys)
        if self.n_events is not None and not (isinstance(self.n_events, (int, np.integer)) and self.n_events >= 1):
            raise ValueError(f"Events: n_events must be a positive int or None, got {self.n_events!r}")
        _check_positive("Events", scale=self.scale)

    def check(self, covariates: Mapping[str, np.ndarray]) -> None:
        """Check every event index lies in ``[0, n_events]``."""
        for key in self.keys:
            _check_index(self.name, key, covariates[key], 0, self.n_events)

    def prepare(self, covariates: Mapping[str, np.ndarray]) -> Events:
        """Size ``n_events`` from the training covariates when it is ``None``, then check them."""
        events = self
        if self.n_events is None:
            largest = max(int(np.max(covariates[key], initial=0)) for key in self.keys)
            events = replace(self, n_events=max(largest, 1))
        events.check(covariates)
        return events

    def contribute(self, ctx: Context) -> jax.Array:
        """The effect of the rows' events, ``(rows, n_batch)``."""
        if self.n_events is None:
            raise ValueError("Events: n_events is not set; HierarchicalCount.fit sizes it from the covariates")
        level = ctx.level(self.pool_by)
        ones = jnp.ones(self.n_events)
        if level is None:
            with ctx.series:
                sigma = numpyro.sample(f"{self.name}/scale", dist.HalfNormal(self.scale))
                effect = numpyro.sample(f"{self.name}/effect", dist.Normal(0.0, sigma[..., None] * ones).to_event(1))
        else:
            group_idx, n_groups = _group_index(level)
            with numpyro.plate(f"{self.name}/groups", n_groups):
                sigma = numpyro.sample(f"{self.name}/scale", dist.HalfNormal(self.scale))
                effect = numpyro.sample(f"{self.name}/effect", dist.Normal(0.0, sigma[..., None] * ones).to_event(1))
            effect = effect[ctx.batch(group_idx)]
        # Index 0, no event, has no effect.
        effect = jnp.concatenate([jnp.zeros_like(effect[..., :1]), effect], axis=-1)
        term = 0.0
        for key in self.keys:
            term = term + effect[:, ctx.covariate(key).astype(jnp.int32)].T
        return term


@dataclass(frozen=True)
class Fourier:
    """Fourier harmonics of a ``period``, e.g. the year, with weights per series or pooled over a level.

    The features are ``sin(2 pi k t / period)`` for ``k = 1..order``, then the cosines, of the day ``t`` counted from
    the first training day, and the weights are ``Normal(0, scale)``.

    Sites: ``"<name>/weight"``, shape ``(..., 2 * order)``.

    Parameters
    ----------
    period : float
        Days per cycle, by default 365.25.
    order : int
        Number of harmonics, by default 4.
    pool_by : str | Level | None
        The level whose groups share the weights; ``None`` for one set per series.
    scale : float
        Scale of the weights, by default 0.3.
    name : str
        Prefix of the sites, by default ``"fourier"``.
    """

    period: float = 365.25
    order: int = 4
    pool_by: PoolBy = None
    scale: float = 0.3
    name: str = "fourier"

    def __post_init__(self) -> None:
        if not (isinstance(self.order, (int, np.integer)) and 1 <= self.order <= self.period / 2):
            raise ValueError(f"Fourier: order must be an int in [1, period / 2], got {self.order!r}")
        _check_positive("Fourier", period=self.period, scale=self.scale)

    @property
    def keys(self) -> tuple[str, ...]:
        """The harmonics read no covariates."""
        return ()

    def features(self, t: jax.Array) -> jax.Array:
        """The sines then the cosines of the days ``t``, shape ``(len(t), 2 * order)``."""
        angle = 2.0 * jnp.pi * jnp.asarray(t, dtype=jnp.float32)[:, None] * jnp.arange(1, self.order + 1) / self.period
        return jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)

    def contribute(self, ctx: Context) -> jax.Array:
        """The harmonics on the rows, ``(rows, n_batch)``."""
        prior = dist.Normal(0.0, self.scale).expand([2 * self.order]).to_event(1)
        weight = _group_effect(f"{self.name}/weight", prior, ctx.level(self.pool_by), ctx)
        return self.features(ctx.t) @ weight.T


@dataclass(frozen=True)
class GroupShocks:
    """Daily shocks shared by the series of a group, zero-sum within each period of ``period`` days.

    The scale of a group's shocks is ``HalfNormal(scale)``. See :func:`~bunobee.models.blocks.group_shocks`.

    Sites: ``"<name>/scale"``, ``"<name>/shock"`` and, forecasting, ``"<name>/future"``.

    Parameters
    ----------
    by : str | Level
        The level whose groups share the shocks, e.g. the store.
    period : int
        Days per zero-sum period, by default 7.
    scale : float
        Scale of the ``HalfNormal`` prior on a group's shock scale, by default 0.2.
    name : str
        Prefix of the sites, by default ``"shocks"``.
    """

    by: str | HierarchyLevel
    period: int = 7
    scale: float = 0.2
    name: str = "shocks"

    def __post_init__(self) -> None:
        if self.by is None:
            raise ValueError("GroupShocks: by must name the level whose groups share the shocks")
        if not (isinstance(self.period, (int, np.integer)) and self.period >= 1):
            raise ValueError(f"GroupShocks: period must be a positive int, got {self.period!r}")
        _check_positive("GroupShocks", scale=self.scale)

    @property
    def keys(self) -> tuple[str, ...]:
        """The shocks read no covariates."""
        return ()

    def contribute(self, ctx: Context) -> jax.Array:
        """The shocks on the rows, ``(rows, n_batch)``."""
        level = ctx.level(self.by)
        with numpyro.plate(f"{self.name}/groups", level.n_groups):
            scale = numpyro.sample(f"{self.name}/scale", dist.HalfNormal(self.scale))
        return group_shocks(self.name, scale, ctx.horizon, level, self.period, series=ctx.series)[ctx.rows]


@dataclass(frozen=True)
class Regressor:
    """A covariate times a coefficient per series or pooled over a level, e.g. the SNAP days of each series' state.

    The coefficient is ``Normal(0, scale)``. The covariate is ``(n_days,)``, the same for every series, or
    ``(n_days, n_series)``.

    Sites: ``"<name>/coef"``.

    Parameters
    ----------
    name : str
        Prefix of the sites, and the covariate's key unless ``key`` is given.
    pool_by : str | Level | None
        The level whose groups share the coefficient; ``None`` for one per series.
    scale : float
        Scale of the coefficient, by default 0.3.
    key : str | None
        The covariate, by default ``name``.
    """

    name: str
    pool_by: PoolBy = None
    scale: float = 0.3
    key: str | None = None

    def __post_init__(self) -> None:
        _check_positive("Regressor", scale=self.scale)

    @property
    def keys(self) -> tuple[str, ...]:
        """``(key,)``, by default ``(name,)``."""
        return (self.name if self.key is None else self.key,)

    def contribute(self, ctx: Context) -> jax.Array:
        """The covariate times its coefficient on the rows, ``(rows, n_batch)``."""
        coef = _group_effect(f"{self.name}/coef", dist.Normal(0.0, self.scale), ctx.level(self.pool_by), ctx)
        x = ctx.covariate(self.keys[0])
        return (x if x.ndim == 2 else x[:, None]) * coef


# ---------------------------------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearOneCycle:
    """Orduz's learning rate schedule, ``optax.linear_onecycle_schedule``, built when it is first called.

    It is the default ``schedule`` of :class:`HierarchicalCount`'s :class:`~bunobee.models.inference.SVI`, and needs
    ``optax`` only once the fit runs.

    Parameters
    ----------
    steps : int
        Steps of the cycle.
    peak_lr : float
        Peak learning rate, by default 0.03.
    pct_start, pct_final : float
        Fractions of the steps at which the rate peaks and at which it has come back down, by default 0.2 and 0.8.
    div_factor, final_div_factor : float
        The rate starts at ``peak_lr / div_factor`` and ends ``final_div_factor`` times lower again, by default 10 each.
    """

    steps: int
    peak_lr: float = 0.03
    pct_start: float = 0.2
    pct_final: float = 0.8
    div_factor: float = 10.0
    final_div_factor: float = 10.0

    def __call__(self, count: Any) -> Any:
        import optax

        schedule = optax.linear_onecycle_schedule(
            transition_steps=self.steps,
            peak_value=self.peak_lr,
            pct_start=self.pct_start,
            pct_final=self.pct_final,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
        )
        return schedule(count)


def default_inference() -> SVI:
    """The default inference of :class:`HierarchicalCount`: the SVI of Orduz's M5 run.

    ``SVI(subsample=1500, steps=3000)`` on :class:`LinearOneCycle`, peaking at 0.03.
    """
    return SVI(subsample=DEFAULT_SUBSAMPLE, steps=DEFAULT_STEPS, schedule=LinearOneCycle(DEFAULT_STEPS))


class HierarchicalCount(Forecaster):
    """A pooled count model of many series: a level plus terms on the log scale, and a count likelihood.

    See the module docstring for the model. Fitted by subsampled SVI by default, it scales to tens of thousands of
    series: each step sees a batch of the series, and the forecast is drawn in chunks of posterior draws.

    Parameters
    ----------
    level : Level | None
        The level of every series, by default ``Level()``: the recent mean, an intercept and a 13-block walk.
    components : Sequence[Component]
        The other terms of the log mean, e.g. ``[Seasonal(7), GroupShocks("store")]``. Their names, and the level's,
        must be distinct.
    obs : ObsFamily | None
        The observation family, by default ``NegBin2()``; it must have count support. Its sites are
        ``"obs/<param>"``.
    hierarchy : Hierarchy | None
        The hierarchy whose levels the components' ``pool_by`` names refer to. Needed when any of them names one.
    inference : NUTS | SVI | MAP | None
        How :meth:`fit` fits the model. By default :func:`default_inference`, SVI on batches of 1,500 series; a panel
        of fewer series is fitted whole at every step.
    log_mean_bounds : tuple[float, float | None]
        Bounds the log mean is clipped to, by default ``(-12, None)``. ``None`` for the upper bound is
        ``log(10 * (max(y) + 1))`` of the training counts.

    Examples
    --------
    >>> hierarchy = Hierarchy.from_frame(meta, {"store": ["store_id"], "dept": ["dept_id"]})  # doctest: +SKIP
    >>> model = HierarchicalCount(
    ...     components=[Seasonal(7, pool_by="store"), GroupShocks("store"), Regressor("snap", pool_by="dept")],
    ...     hierarchy=hierarchy,
    ... )  # doctest: +SKIP
    >>> model.fit(y, {"snap": snap}, gate=listed)  # doctest: +SKIP
    >>> forecast = model.predict({"snap": snap_future}, gate_future=listed_future)  # doctest: +SKIP
    >>> draw_quantiles(aggregate_draws(forecast, hierarchy, "store"), [0.05, 0.5, 0.95])  # doctest: +SKIP
    """

    inference_methods = (NUTS, SVI, MAP)
    series_plate = "series"

    def __init__(
        self,
        level: Level | None = None,
        components: Sequence[Component] = (),
        obs: ObsFamily | None = None,
        hierarchy: Hierarchy | None = None,
        inference: Inference | None = None,
        *,
        log_mean_bounds: tuple[float, float | None] = (-12.0, None),
    ) -> None:
        self._default_inference = inference is None
        super().__init__(default_inference() if inference is None else inference)
        self.level = Level() if level is None else level
        self.components = tuple(components)
        self.obs = NegBin2() if obs is None else obs
        if getattr(self.obs, "support", None) != "count":
            raise ValueError(
                f"HierarchicalCount models counts, so obs must have count support, got {self.obs!r} with support "
                f"{getattr(self.obs, 'support', None)!r}"
            )
        names = [term.name for term in self.terms]
        if len(set(names)) != len(names) or OBS_NAME in names:
            raise ValueError(f"the level and the components need distinct names other than {OBS_NAME!r}, got {names}")
        if hierarchy is not None and not isinstance(hierarchy, Hierarchy):
            raise ValueError(f"hierarchy must be a Hierarchy or None, got {hierarchy!r}")
        self.hierarchy = hierarchy
        low, high = log_mean_bounds
        if high is not None and not high > low:
            raise ValueError(f"log_mean_bounds must be (low, high) with high > low, got {log_mean_bounds}")
        self.log_mean_bounds = (float(low), None if high is None else float(high))
        self._n_fit_series = 0
        self._fitted: dict[str, Any] = {}

    @property
    def terms(self) -> tuple[Any, ...]:
        """The level, then the components."""
        return (self.level, *self.components)

    def _init_strategy(self):
        """``init_to_median``, as in Orduz's run."""
        return init_to_median

    def _inference_for(self, mcmc_args: Mapping[str, Any] | None) -> Inference:
        """The default SVI fits a panel of no more series than its batch whole, at every step."""
        inference = super()._inference_for(mcmc_args)
        if self._default_inference and inference.subsample is not None and self._n_fit_series <= inference.subsample:
            inference = copy.copy(inference)
            inference.subsample = None
        return inference

    def model(
        self,
        y: Any,
        gate: Any,
        offset: Any,
        covariates: Mapping[str, Any],
        horizon: Horizon,
        terms: Sequence[Any],
        bounds: tuple[float, float],
        forecast: bool = False,
    ) -> None:
        """The NumPyro model.

        While fitting (``forecast=False``) the counts ``y``, ``(n_obs, n_series)``, are observed at the site ``"y"``
        under the plates ``"time"`` and ``"series"``. With ``forecast=True`` the terms cover the forecast days only
        and the mean over them, before the gate, is the deterministic site ``"mean_future"``, ``(future,
        n_series)``; nothing is observed. ``terms`` are the level and the components, as :meth:`fit` prepared them.
        """
        series = numpyro.plate(SERIES_DIM, offset.shape[0])
        rows = slice(horizon.n_obs, None) if forecast else slice(None)
        ctx = Context(horizon, rows, covariates, series, offset, self.hierarchy)
        # Every row gets the time axis, also when no term varies over time.
        eta = jnp.zeros((ctx.n_rows, 1))
        for term in terms:
            eta = eta + term.contribute(ctx)
        params = self.obs.sample_params(OBS_NAME, series=series)
        mean = jnp.exp(jnp.clip(eta, *bounds))
        if forecast:
            numpyro.deterministic(MEAN_FUTURE_SITE, mean)
            return
        with series:
            y_batch = numpyro.subsample(y, event_dim=0)
            gate_batch = numpyro.subsample(gate, event_dim=0)
        with numpyro.plate("time", horizon.n_obs, dim=-2), series:
            numpyro.sample(OBS_SITE, self.obs.dist(mean, params, gate_batch), obs=y_batch)

    def fit(
        self,
        y: np.ndarray | pd.DataFrame,
        covariates: Mapping[str, np.ndarray] | None = None,
        *,
        gate: np.ndarray | None = None,
        mcmc_args: Mapping[str, Any] | None = None,
        seed: int = 0,
    ) -> HierarchicalCount:
        """Fit the model to the counts ``y`` and store the posterior on ``idata_``.

        Parameters
        ----------
        y : np.ndarray | pd.DataFrame
            Non-negative integer counts, ``(n_obs, n_series)``, ending at the forecast origin. A DataFrame's columns
            label the series. NaN is a day the series is not listed, as if ``gate`` were 0 there.
        covariates : Mapping[str, np.ndarray] | None
            The covariates the components read, by key, each ``(n_obs,)`` or ``(n_obs, n_series)``, e.g.
            ``{"day_of_month": dates.day, "event": event_index, "snap": snap}``.
        gate : np.ndarray | None
            ``1`` on the days a series is listed (it can sell), ``0`` where its mean is structurally zero, e.g. no
            listed price or the store is closed, shape of ``y``. ``None`` lists every day. A count on a gated day is
            taken as 0. The gate also picks the days of the level's anchor.
        mcmc_args : Mapping | None
            On top of the model's ``NUTS`` settings for this fit; only for a model fitted by NUTS.
        seed : int
            Seed of the fit's ``jax.random.PRNGKey``, by default 0.

        Returns
        -------
        HierarchicalCount
            ``self``, fitted. ``idata_`` holds every latent site, and the deterministic ones, with dims ``(chain,
            draw, ...)``; a site's axis over the series is the ``series`` dim.
        """
        labels = y.columns.to_numpy() if isinstance(y, pd.DataFrame) else None
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 2 or 0 in y.shape:
            raise ValueError(f"y must be (n_obs, n_series) with at least one day and series, got shape {y.shape}")
        n_obs, n_series = y.shape
        gate = np.ones_like(y) if gate is None else np.asarray(gate, dtype=np.float64)
        if gate.shape != y.shape or not np.isfinite(gate).all() or (gate < 0).any() or (gate > 1).any():
            raise ValueError(f"gate must be finite in [0, 1] with the shape of y {y.shape}, got shape {gate.shape}")
        missing = np.isnan(y)
        gate = np.where(missing, 0.0, gate)
        y = np.where(missing | (gate == 0), 0.0, y)
        if not np.isfinite(y).all() or (y < 0).any() or not np.all(np.mod(y, 1) == 0):
            raise ValueError("y must hold non-negative integer counts, or NaN")
        if self.hierarchy is not None and self.hierarchy.n_series != n_series:
            raise ValueError(f"the hierarchy groups {self.hierarchy.n_series} series but y has {n_series}")

        covariates = _check_covariates(covariates, self._keys(self.terms), n_obs, n_series, "covariates")
        terms = tuple(term.prepare(covariates) if hasattr(term, "prepare") else term for term in self.terms)
        offset = np.asarray(self.level.anchor(y, gate), dtype=np.float32)
        low, high = self.log_mean_bounds
        bounds = (low, float(np.log(10.0 * (y.max() + 1.0))) if high is None else high)

        inputs = {
            "y": jnp.asarray(y, dtype=jnp.int32),
            "gate": jnp.asarray(gate, dtype=jnp.float32),
            "offset": jnp.asarray(offset),
            "covariates": {key: jnp.asarray(value) for key, value in covariates.items()},
            "horizon": Horizon(n_obs),
            "terms": terms,
            "bounds": bounds,
        }
        trace = handlers.trace(handlers.seed(self.model, jax.random.PRNGKey(0))).get_trace(**inputs)
        dims = _site_dims(trace, n_series)
        latent = [name for name, site in trace.items() if site["type"] == "sample" and not site["is_observed"]]

        self._n_fit_series = n_series
        self._fit(inputs, mcmc_args=mcmc_args, seed=seed, context={"dims": dims, "labels": labels})
        self._fitted = {
            "covariates": covariates,
            "offset": offset,
            "terms": terms,
            "bounds": bounds,
            "n_obs": n_obs,
            "n_series": n_series,
            "labels": labels,
            "latent": latent,
        }
        return self

    @staticmethod
    def _keys(terms: Sequence[Any]) -> tuple[str, ...]:
        keys: list[str] = []
        for term in terms:
            keys.extend(key for key in term.keys if key not in keys)
        return tuple(keys)

    def _to_idata(self, samples, inputs, context) -> xr.Dataset:
        data_vars = {
            name: (("chain", "draw", *context["dims"][name]), np.asarray(value)) for name, value in samples.items()
        }
        idata = xr.Dataset(data_vars)
        if context["labels"] is not None:
            idata = idata.assign_coords({SERIES_DIM: context["labels"]})
        return idata

    def predict(
        self,
        covariates_future: Mapping[str, np.ndarray] | None = None,
        *,
        gate_future: np.ndarray | None = None,
        horizon: int | None = None,
        num_draws: int | None = None,
        chunk: int = 50,
        seed: int = 0,
    ) -> xr.Dataset:
        """Draw count forecasts of every series from the fitted posterior.

        The posterior draws are taken ``chunk`` at a time. For each chunk a forecast-only pass of the model gives
        the mean and the observation parameters over the forecast days (drawing the walk's and the shocks' future
        sites afresh), and the family's :meth:`~bunobee.models.obs.ObsFamily.sample` draws the counts with NumPy.

        Parameters
        ----------
        covariates_future : Mapping[str, np.ndarray] | None
            The covariates of the forecast days, by the keys of :meth:`fit`, each ``(future,)`` or ``(future,
            n_series)``. ``None`` when the components read none.
        gate_future : np.ndarray | None
            The gate of the forecast days, ``(future, n_series)``; ``None`` lists every day.
        horizon : int | None
            Number of forecast days; by default the length of the future covariates or gate.
        num_draws : int | None
            Number of forecast draws, by default one per posterior draw. More draws than the posterior has reuse its
            draws in turn.
        chunk : int
            Posterior draws per forecast-only pass, by default 50.
        seed : int
            Seed of the pass's ``jax.random.PRNGKey`` and of the NumPy generator of the counts, by default 0.

        Returns
        -------
        xr.Dataset
            ``forecast_samples``, the counts as float32, dims ``(sample, time, series)``; ``time`` counts the
            forecast days from 0. It goes through :func:`~bunobee.hierarchy.aggregate_draws` and
            :func:`~bunobee.hierarchy.draw_quantiles` as it is.
        """
        idata = self.idata_
        fitted = self._fitted
        n_obs, n_series = fitted["n_obs"], fitted["n_series"]
        future = _forecast_length(covariates_future, gate_future, horizon)
        if not (isinstance(chunk, int) and chunk >= 1):
            raise ValueError(f"chunk must be a positive int, got {chunk!r}")
        keys = tuple(fitted["covariates"])
        covariates_future = _check_covariates(covariates_future, keys, future, n_series, "covariates_future")
        for term in fitted["terms"]:
            if hasattr(term, "check"):
                term.check(covariates_future)
        gate_future = np.ones((future, n_series)) if gate_future is None else np.asarray(gate_future, np.float64)
        if gate_future.shape != (future, n_series) or not np.isfinite(gate_future).all():
            raise ValueError(f"gate_future must be finite with shape {(future, n_series)}, got {gate_future.shape}")

        n_posterior = idata.sizes["chain"] * idata.sizes["draw"]
        num_draws = n_posterior if num_draws is None else num_draws
        if not (isinstance(num_draws, (int, np.integer)) and num_draws >= 1):
            raise ValueError(f"num_draws must be a positive int or None, got {num_draws!r}")
        order = np.arange(num_draws) % n_posterior
        latent = {name: idata[name].values.reshape(n_posterior, *idata[name].shape[2:]) for name in fitted["latent"]}
        covariates = {
            key: jnp.asarray(np.concatenate([fitted["covariates"][key], covariates_future[key]], axis=0))
            for key in keys
        }
        inputs = {
            "y": None,
            "gate": None,
            "offset": jnp.asarray(fitted["offset"]),
            "covariates": covariates,
            "horizon": Horizon(n_obs, future),
            "terms": fitted["terms"],
            "bounds": fitted["bounds"],
            "forecast": True,
        }
        param_sites = {param: f"{OBS_NAME}/{param}" for param in self.obs.param_names}
        key = jax.random.PRNGKey(seed)
        rng = np.random.default_rng(seed)
        draws = np.empty((num_draws, future, n_series), dtype=np.float32)
        for start in range(0, num_draws, chunk):
            picked = order[start : start + chunk]
            predictive = Predictive(
                self.model,
                posterior_samples={name: value[picked] for name, value in latent.items()},
                return_sites=[MEAN_FUTURE_SITE, *param_sites.values()],
            )
            pass_ = predictive(jax.random.fold_in(key, start), **inputs)
            mean = np.asarray(pass_[MEAN_FUTURE_SITE], dtype=np.float64)
            params = {param: _with_time_axis(np.asarray(pass_[site])) for param, site in param_sites.items()}
            draws[start : start + picked.size] = self.obs.sample(rng, mean, params, gate=gate_future[None])

        coords = {SAMPLE_DIM: np.arange(num_draws), "time": np.arange(future)}
        if fitted["labels"] is not None:
            coords[SERIES_DIM] = fitted["labels"]
        return xr.Dataset({DRAWS_VAR: ((SAMPLE_DIM, "time", SERIES_DIM), draws)}, coords=coords)


def _with_time_axis(value: np.ndarray) -> np.ndarray:
    """A parameter of a chunk, ``(chunk,)`` or ``(chunk, n_series)``, given a forecast-day axis after the chunk's."""
    return value.reshape(value.shape[0], 1, *value.shape[1:]) if value.ndim >= 2 else value[:, None, None]


def _forecast_length(covariates: Mapping[str, np.ndarray] | None, gate: np.ndarray | None, horizon: int | None) -> int:
    """The number of forecast days, from ``horizon``, the gate or the covariates, which must agree."""
    lengths = {np.shape(value)[0] for value in (covariates or {}).values()}
    if gate is not None:
        lengths.add(np.shape(gate)[0])
    if horizon is not None:
        lengths.add(horizon)
    if len(lengths) != 1:
        if not lengths:
            raise ValueError("pass horizon=, covariates_future or gate_future: the forecast length is not known")
        raise ValueError(f"horizon, covariates_future and gate_future give different forecast lengths: {lengths}")
    future = lengths.pop()
    if not (isinstance(future, (int, np.integer)) and future >= 1):
        raise ValueError(f"the forecast must cover at least one day, got {future!r}")
    return int(future)


def _check_covariates(
    covariates: Mapping[str, np.ndarray] | None, keys: Sequence[str], n_days: int, n_series: int, label: str
) -> dict[str, np.ndarray]:
    """The covariates of ``keys`` as finite NumPy arrays of ``n_days`` rows, ``(n_days,)`` or ``(n_days, n_series)``."""
    covariates = dict(covariates or {})
    missing = [key for key in keys if key not in covariates]
    if missing:
        raise KeyError(f"{label} has no {missing}; the components read {list(keys)}")
    out = {}
    for key in keys:
        value = np.asarray(covariates[key])
        if value.shape not in ((n_days,), (n_days, n_series)) or not np.isfinite(value).all():
            raise ValueError(
                f"{label}[{key!r}] must be finite with shape ({n_days},) or ({n_days}, {n_series}), got {value.shape}"
            )
        out[key] = value.astype(np.int32) if np.issubdtype(value.dtype, np.integer) else value.astype(np.float32)
    return out


def _site_dims(trace: Mapping[str, Any], n_series: int) -> dict[str, list[str]]:
    """Dims of every site past ``(chain, draw)``: the axis of the series plate is ``series``, the others are numbered.

    A deterministic site has no plates, so a ``(n_series,)`` one gets the ``series`` dim.
    """
    dims: dict[str, list[str]] = {}
    for name, site in trace.items():
        if site["type"] not in ("sample", "deterministic") or site.get("is_observed"):
            continue
        shape = np.shape(site["value"])
        names = [f"{name}_dim_{i}" for i in range(len(shape))]
        if site["type"] == "sample":
            event_ndim = len(site["fn"].event_shape)
            for frame in site["cond_indep_stack"]:
                if frame.name == SERIES_DIM:
                    names[len(shape) - event_ndim + frame.dim] = SERIES_DIM
        elif shape == (n_series,):
            names = [SERIES_DIM]
        dims[name] = names
    return dims
