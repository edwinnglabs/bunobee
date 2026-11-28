"""Observation families: the likelihood of a model and the sampler of its forecasts.

A model picks its likelihood by taking an observation family, e.g. ``DLT(..., obs=StudentT())``, instead of
branching on a distribution name. A family implements the :class:`ObsFamily` protocol:

- :meth:`~ObsFamily.sample_params` registers the priors of the family's parameters in the NumPyro model and returns
  them, e.g. ``{"sigma": ...}``;
- :meth:`~ObsFamily.dist` is the NumPyro distribution of the observations given their mean and the parameters;
- :meth:`~ObsFamily.sample` draws forecasts with NumPy from the posterior parameters;
- :attr:`~ObsFamily.support` is ``"real"``, ``"positive"`` or ``"count"``, so a model can refuse a family it cannot
  use when it is constructed.

The families are :class:`Normal`, :class:`StudentT`, :class:`Poisson` and :class:`NegBin2`. Each is a frozen
dataclass; a user family needs no base class, only the members of the protocol.

Gate
----
``dist`` and ``sample`` take an optional ``gate``, ``1`` on the steps where the series can be observed and ``0`` where
its mean is structurally zero, e.g. the item has no listed price or the store is closed. The mean is multiplied by the
gate and :data:`GATE_FLOOR` is added (:func:`gate_mean`), so a gated step keeps a valid, tiny mean and contributes the
likelihood of a zero under it. This is not the NaN mask of a model's ``y``, which means "missing, skip this step".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from bunobee.hierarchy import Level
from bunobee.models.blocks import _batch, _group_index

if TYPE_CHECKING:
    from collections.abc import Mapping

    from jax import Array

__all__ = [
    "GATE_FLOOR",
    "SUPPORTS",
    "NegBin2",
    "Normal",
    "ObsFamily",
    "Poisson",
    "RealObsFamily",
    "StudentT",
    "gate_mean",
]

#: The supports a family may declare.
SUPPORTS = ("real", "positive", "count")

#: Added to the gated mean, so that a gated step has a small positive mean rather than an invalid zero.
GATE_FLOOR = 1e-6


@runtime_checkable
class ObsFamily(Protocol):
    """An observation family: the likelihood of a model and the sampler of its forecasts."""

    #: ``"real"``, ``"positive"`` or ``"count"``: the values the observations can take.
    support: str
    #: Names of the parameters :meth:`sample_params` returns. Each is also a site of the model, ``"<name>/<param>"``
    #: (or ``"<param>"`` with ``name=None``), so the parameters can be read back from the posterior samples.
    param_names: tuple[str, ...]

    def sample_params(self, name: str | None, series: numpyro.plate | None = None) -> dict[str, Any]:
        """Register the priors of the family's parameters under ``name`` and return the parameters by name."""
        ...

    def dist(self, mean: Any, params: Mapping[str, Any], gate: Any = None) -> dist.Distribution:
        """The NumPyro distribution of the observations given their ``mean``."""
        ...

    def sample(self, rng: np.random.Generator, mean: Any, params: Mapping[str, Any], gate: Any = None) -> np.ndarray:
        """Draw observations with NumPy given their ``mean``, broadcasting ``mean`` against ``params``."""
        ...


@runtime_checkable
class RealObsFamily(ObsFamily, Protocol):
    """A family with real support that also draws additive noise in JAX, as ``DLT`` needs for its state updates."""

    def noise(self, key: Array, params: Mapping[str, Any], shape: tuple[int, ...]) -> Array:
        """Zero-mean noise of ``shape`` with JAX, broadcasting ``params`` against ``shape``."""
        ...


def gate_mean(mean: Any, gate: Any = None, floor: float = GATE_FLOOR) -> Any:
    """The mean of a gated likelihood: ``mean * gate + floor``, or ``mean`` unchanged when ``gate`` is ``None``.

    Parameters
    ----------
    mean : Array | np.ndarray
        Mean of the observations.
    gate : Array | np.ndarray | None
        ``1`` where the series can be observed, ``0`` where its mean is structurally zero; broadcast against ``mean``.
    floor : float
        Added to the gated mean, by default :data:`GATE_FLOOR`.

    Returns
    -------
    Array | np.ndarray
        The gated mean, of the array type of ``mean``.
    """
    if gate is None:
        return mean
    return mean * gate + floor


def _site(name: str | None, param: str) -> str:
    """Site name of ``param``: ``"<name>/<param>"``, or ``param`` alone when ``name`` is ``None``."""
    return param if name is None else f"{name}/{param}"


def _check_positive(family: str, **values: float) -> None:
    for key, value in values.items():
        if not value > 0:
            raise ValueError(f"{family}: {key} must be positive, got {value}")


def _np_params(mean: Any, params: Mapping[str, Any], gate: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The gated ``mean`` and ``params`` as float64 NumPy arrays."""
    mean = np.asarray(gate_mean(np.asarray(mean, dtype=np.float64), None if gate is None else np.asarray(gate)))
    return mean, {key: np.asarray(value, dtype=np.float64) for key, value in params.items()}


@dataclass(frozen=True)
class Normal:
    """Normal observations, ``Normal(mean, sigma)`` with ``sigma ~ HalfNormal(sigma_scale)``.

    Sites: ``"<name>/sigma"``.

    Parameters
    ----------
    sigma_scale : float
        Scale of the ``HalfNormal`` prior on ``sigma``, by default 5.
    """

    sigma_scale: float = 5.0

    support: ClassVar[str] = "real"
    param_names: ClassVar[tuple[str, ...]] = ("sigma",)

    def __post_init__(self) -> None:
        _check_positive("Normal", sigma_scale=self.sigma_scale)

    def sample_params(self, name: str | None, series: numpyro.plate | None = None) -> dict[str, Any]:
        """Sample ``sigma``, one for all the series; ``series`` is accepted for the protocol and not used."""
        return {"sigma": numpyro.sample(_site(name, "sigma"), dist.HalfNormal(self.sigma_scale))}

    def dist(self, mean: Any, params: Mapping[str, Any], gate: Any = None) -> dist.Distribution:
        """``Normal(gate_mean(mean, gate), sigma)``."""
        return dist.Normal(loc=gate_mean(mean, gate), scale=params["sigma"])

    def sample(self, rng: np.random.Generator, mean: Any, params: Mapping[str, Any], gate: Any = None) -> np.ndarray:
        """``mean + sigma * z`` with standard normal ``z``, float64."""
        mean, params = _np_params(mean, params, gate)
        shape = np.broadcast_shapes(mean.shape, params["sigma"].shape)
        return mean + params["sigma"] * rng.standard_normal(shape)

    def noise(self, key: Array, params: Mapping[str, Any], shape: tuple[int, ...]) -> Array:
        """``sigma * z`` with standard normal ``z`` drawn by ``jax.random.normal``."""
        return jax.random.normal(key, shape=shape) * params["sigma"]


@dataclass(frozen=True)
class StudentT:
    """Student-t observations, ``StudentT(nu, mean, sigma)``, a heavier-tailed backup to :class:`Normal`.

    ``sigma ~ HalfNormal(sigma_scale)`` and the degrees of freedom ``nu ~ Uniform(min_nu, max_nu)``, sampled in that
    order. Sites: ``"<name>/sigma"`` and ``"<name>/nu"``.

    Parameters
    ----------
    min_nu, max_nu : float
        Bounds of the uniform prior on ``nu``, by default 5 and 40; need ``0 < min_nu < max_nu``.
    sigma_scale : float
        Scale of the ``HalfNormal`` prior on ``sigma``, by default 5.
    """

    min_nu: float = 5.0
    max_nu: float = 40.0
    sigma_scale: float = 5.0

    support: ClassVar[str] = "real"
    param_names: ClassVar[tuple[str, ...]] = ("sigma", "nu")

    def __post_init__(self) -> None:
        if not 0 < self.min_nu < self.max_nu:
            raise ValueError(f"need 0 < min_nu < max_nu, got min_nu={self.min_nu}, max_nu={self.max_nu}")
        _check_positive("StudentT", sigma_scale=self.sigma_scale)

    def sample_params(self, name: str | None, series: numpyro.plate | None = None) -> dict[str, Any]:
        """Sample ``sigma`` then ``nu``, one each for all the series; ``series`` is not used."""
        sigma = numpyro.sample(_site(name, "sigma"), dist.HalfNormal(self.sigma_scale))
        nu = numpyro.sample(_site(name, "nu"), dist.Uniform(self.min_nu, self.max_nu))
        return {"sigma": sigma, "nu": nu}

    def dist(self, mean: Any, params: Mapping[str, Any], gate: Any = None) -> dist.Distribution:
        """``StudentT(nu, gate_mean(mean, gate), sigma)``."""
        return dist.StudentT(df=params["nu"], loc=gate_mean(mean, gate), scale=params["sigma"])

    def sample(self, rng: np.random.Generator, mean: Any, params: Mapping[str, Any], gate: Any = None) -> np.ndarray:
        """``mean + sigma * t`` with standard Student-t ``t`` of ``nu`` degrees of freedom, float64."""
        mean, params = _np_params(mean, params, gate)
        shape = np.broadcast_shapes(mean.shape, params["sigma"].shape, params["nu"].shape)
        return mean + params["sigma"] * rng.standard_t(np.broadcast_to(params["nu"], shape), size=shape)

    def noise(self, key: Array, params: Mapping[str, Any], shape: tuple[int, ...]) -> Array:
        """``sigma * t`` with standard Student-t ``t`` drawn by ``jax.random.t``."""
        return jax.random.t(key, df=params["nu"], shape=shape) * params["sigma"]


@dataclass(frozen=True)
class Poisson:
    """Poisson counts, ``Poisson(mean)``. No parameters; ``mean`` must be positive.

    Its variance equals its mean, so overdispersed counts want :class:`NegBin2`.
    """

    support: ClassVar[str] = "count"
    param_names: ClassVar[tuple[str, ...]] = ()

    def sample_params(self, name: str | None, series: numpyro.plate | None = None) -> dict[str, Any]:
        """Nothing to sample; returns ``{}``."""
        return {}

    def dist(self, mean: Any, params: Mapping[str, Any], gate: Any = None) -> dist.Distribution:
        """``Poisson(gate_mean(mean, gate))``."""
        return dist.Poisson(rate=gate_mean(mean, gate))

    def sample(self, rng: np.random.Generator, mean: Any, params: Mapping[str, Any], gate: Any = None) -> np.ndarray:
        """Poisson counts of rate ``mean``, as float64."""
        mean, _ = _np_params(mean, params, gate)
        return rng.poisson(mean).astype(np.float64)


@dataclass(frozen=True)
class NegBin2:
    """Negative binomial counts in the mean parametrisation, ``NegativeBinomial2(mean, concentration)``.

    The variance is ``mean + mean**2 / concentration``: the concentration is the inverse of the dispersion, and a
    large concentration tends to :class:`Poisson`. Its log is ``Normal(log_concentration_loc,
    log_concentration_scale)``:

    - with ``dispersion_pool_by=None``, one log-concentration per series under the ``series`` plate, or one for all
      when no plate is given;
    - with ``dispersion_pool_by`` a :class:`~bunobee.hierarchy.Level`, one per group of that level, under the plate
      ``"<name>/dispersion_groups"``, shared by the group's series;
    - with ``deviation_by`` a level as well, one per series again, ``Normal(group_log_concentration, tau)`` around its
      group's (the site ``"<name>/log_concentration_group"``), where ``tau ~ HalfNormal(deviation_scale)`` is shared
      by the series of a group of ``deviation_by`` (the site ``"<name>/deviation_scale"``, under the plate
      ``"<name>/deviation_groups"``). This is the dispersion of Orduz's M5 model: per item, pooled per store ×
      department, with a spread per department.

    Sites: ``"<name>/log_concentration"`` and the deterministic ``"<name>/concentration"``, the concentration of each
    series of the batch (or the one for all). Call :meth:`sample_params` outside the series plate.

    Parameters
    ----------
    dispersion_pool_by : Level | None
        The hierarchy level whose groups share a dispersion, e.g. ``hierarchy["store"]``; ``None`` for one per series.
    log_concentration_loc, log_concentration_scale : float
        Location and scale of the ``Normal`` prior on the log-concentration (of a group, with ``dispersion_pool_by``),
        by default 0 and 1.
    deviation_by : Level | None
        The level whose groups share the spread of the series' log-concentrations around their group's; needs
        ``dispersion_pool_by`` and the ``series`` plate. By default ``None``, no series-level dispersion.
    deviation_scale : float
        Scale of the ``HalfNormal`` prior on that spread, by default 1.
    """

    dispersion_pool_by: Level | None = None
    log_concentration_loc: float = 0.0
    log_concentration_scale: float = 1.0
    deviation_by: Level | None = None
    deviation_scale: float = 1.0

    support: ClassVar[str] = "count"
    param_names: ClassVar[tuple[str, ...]] = ("concentration",)

    def __post_init__(self) -> None:
        if self.dispersion_pool_by is not None and not isinstance(self.dispersion_pool_by, Level):
            raise ValueError(f"dispersion_pool_by must be a hierarchy Level or None, got {self.dispersion_pool_by!r}")
        if self.deviation_by is not None:
            if not isinstance(self.deviation_by, Level):
                raise ValueError(f"deviation_by must be a hierarchy Level or None, got {self.deviation_by!r}")
            if self.dispersion_pool_by is None:
                raise ValueError("deviation_by needs dispersion_pool_by, the groups the series deviate from")
        _check_positive(
            "NegBin2", log_concentration_scale=self.log_concentration_scale, deviation_scale=self.deviation_scale
        )

    def sample_params(self, name: str | None, series: numpyro.plate | None = None) -> dict[str, Any]:
        """Sample the log-concentration and return the ``concentration`` of each series of the ``series`` batch.

        The shape is ``(n_batch,)`` with a ``series`` plate or ``dispersion_pool_by``, else ``()``.
        """
        prior = dist.Normal(self.log_concentration_loc, self.log_concentration_scale)
        site = _site(name, "log_concentration")
        if self.deviation_by is not None:
            if series is None:
                raise ValueError("NegBin2 with deviation_by samples one log-concentration per series; pass series=")
            group_idx, n_groups = _group_index(self.dispersion_pool_by)
            dev_idx, n_dev = _group_index(self.deviation_by)
            with numpyro.plate(_site(name, "dispersion_groups"), n_groups):
                log_group = numpyro.sample(_site(name, "log_concentration_group"), prior)
            with numpyro.plate(_site(name, "deviation_groups"), n_dev):
                tau = numpyro.sample(_site(name, "deviation_scale"), dist.HalfNormal(self.deviation_scale))
            loc = log_group[_batch(group_idx, series)]
            scale = tau[_batch(dev_idx, series)]
            with series:
                log_concentration = numpyro.sample(site, dist.Normal(loc, scale))
        elif self.dispersion_pool_by is not None:
            group_idx, n_groups = _group_index(self.dispersion_pool_by)
            with numpyro.plate(_site(name, "dispersion_groups"), n_groups):
                log_group = numpyro.sample(site, prior)
            log_concentration = log_group[_batch(group_idx, series)]
        elif series is not None:
            with series:
                log_concentration = numpyro.sample(site, prior)
        else:
            log_concentration = numpyro.sample(site, prior)
        concentration = numpyro.deterministic(_site(name, "concentration"), jnp.exp(log_concentration))
        return {"concentration": concentration}

    def dist(self, mean: Any, params: Mapping[str, Any], gate: Any = None) -> dist.Distribution:
        """``NegativeBinomial2(gate_mean(mean, gate), concentration)``."""
        return dist.NegativeBinomial2(mean=gate_mean(mean, gate), concentration=params["concentration"])

    def sample(self, rng: np.random.Generator, mean: Any, params: Mapping[str, Any], gate: Any = None) -> np.ndarray:
        """Negative binomial counts of mean ``mean``, as float64.

        Drawn as ``rng.negative_binomial(n=concentration, p=concentration / (concentration + mean))``, which has the
        mean ``mean`` and the variance ``mean + mean**2 / concentration`` of ``NegativeBinomial2``.
        """
        mean, params = _np_params(mean, params, gate)
        concentration = params["concentration"]
        shape = np.broadcast_shapes(mean.shape, concentration.shape)
        n = np.broadcast_to(concentration, shape)
        return rng.negative_binomial(n, n / (n + np.broadcast_to(mean, shape))).astype(np.float64)
