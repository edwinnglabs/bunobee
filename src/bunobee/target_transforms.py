"""Target transforms: the scale a series is fitted on, and the way back to units.

The MAP fitters in :mod:`bunobee.models.ssp.map_fit` fit a linear-Gaussian state-space model to the
transformed values of each series, and their predictors map the Gaussian forecast back to a point forecast in
units. A :class:`Transform` owns both directions, so trying another target needs a new transform, not an edit
of the fitter. :class:`LogTransform` is the default; :class:`PowerTransform` fits a Box-Cox or Yeo-Johnson
power per series.

The transforms are written for non-negative series such as sales, counts or demand: values below ``floor`` are
floored, and ``to_units`` and ``inverse`` never return a negative value.

Unlike :mod:`bunobee.models.ssp.map_fit`, this module needs only ``numpy`` and ``scipy``.
:class:`PowerTransform` fits its powers with ``scikit-learn``, part of the optional ``optim`` extra, which it
imports only when it fits.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy import special


def _per_series(value: Any) -> Any:
    """Add a trailing day axis to a per-series parameter, so it broadcasts over the days.

    A scalar is returned as it is, so a Python float keeps its weak dtype and a float32 forecast stays float32.
    """
    return value if np.isscalar(value) else np.asarray(value)[..., None]


class Transform(Protocol):
    """A per-series map from values to the scale an SSP model is fitted on, and back to units.

    ``fit`` fits each series' parameters on its fit days, ``forward`` maps values to the transformed scale and
    ``inverse`` maps a Gaussian forecast on that scale back to a point forecast in units. The inverse of the
    forecast mean is not the mean of the values (``exp(mu)`` is a lognormal's median), so ``inverse`` takes the
    forecast variance too and carries the bias correction.

    Arrays hold one series per row along their leading dimensions and its days along the last one.
    """

    def fit(self, values: np.ndarray) -> dict[str, np.ndarray]:
        """Fit the per-series parameters on the fit days.

        Parameters
        ----------
        values : np.ndarray
            Non-negative observations over the fit days, of shape ``(..., n_steps)``.

        Returns
        -------
        dict of str to np.ndarray
            The parameters by name, each of shape ``(...)``: one value per series. The fitters return them
            among their outputs, so a name must not be one of the fitter's own keys (``at``, ``sigma_h``, ...).
        """
        ...

    def forward(self, values: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Map values to the transformed scale.

        Parameters
        ----------
        values : np.ndarray
            Non-negative observations, of shape ``(..., n_steps)``.
        params : Mapping of str to Any
            Holds at least the parameters ``fit`` returned, for the same series.

        Returns
        -------
        np.ndarray
            The transformed values, of shape ``(..., n_steps)``.
        """
        ...

    def inverse(self, mu: np.ndarray, var: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Map a Gaussian forecast on the transformed scale to a point forecast in units.

        Parameters
        ----------
        mu : np.ndarray
            Forecast mean on the transformed scale, of shape ``(..., horizon)``.
        var : np.ndarray
            Forecast variance on the transformed scale, of the shape of ``mu``.
        params : Mapping of str to Any
            Holds at least the parameters ``fit`` returned, for the same series, e.g. a fitter's output.

        Returns
        -------
        np.ndarray
            Point forecasts in units, of the shape of ``mu``.
        """
        ...

    def to_units(self, z: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Map values on the transformed scale to units: the inverse of ``forward``, kept within the admissible range.

        Unlike ``inverse`` it maps each value as it is, not a Gaussian forecast, so it takes draws or quantiles of the
        forecast distribution to the same distribution in units. It is monotone in ``z``.

        Parameters
        ----------
        z : np.ndarray
            Values on the transformed scale, of shape ``(..., n_steps)``. Leading dimensions beyond the series'
            own, e.g. one of draws, broadcast against the parameters.
        params : Mapping of str to Any
            Holds at least the parameters ``fit`` returned, for the same series.

        Returns
        -------
        np.ndarray
            Values in units, of the shape of ``z``, never negative.
        """
        ...

    def quantile(self, mu: np.ndarray, var: np.ndarray, u: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Quantiles in units of a Gaussian forecast on the transformed scale.

        The transform is monotone, so the ``u`` quantile of the forecast in units is ``to_units`` of the ``u``
        quantile ``mu + sqrt(var) * ndtri(u)`` of the Gaussian: these are the exact quantiles of the distribution
        ``inverse`` takes the mean of, floored at 0.

        Parameters
        ----------
        mu, var : np.ndarray
            Forecast mean and variance on the transformed scale, of shape ``(..., horizon)``.
        u : np.ndarray
            Quantile levels in (0, 1), of shape ``(n_quantiles,)``.
        params : Mapping of str to Any
            Holds at least the parameters ``fit`` returned, for the same series.

        Returns
        -------
        np.ndarray
            Quantiles in units, of shape ``(..., n_quantiles, horizon)``, increasing along the quantile axis.
        """
        ...


def _gaussian_quantile(
    transform: Transform, mu: np.ndarray, var: np.ndarray, u: np.ndarray, params: Mapping[str, Any]
) -> np.ndarray:
    """Quantiles of a Gaussian forecast in units; see :meth:`Transform.quantile`.

    The quantile axis leads while the quantiles are mapped, so the per-series parameters broadcast as they do for
    a forecast, and moves to second-to-last at the end.
    """
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 1 or not ((u > 0) & (u < 1)).all():
        raise ValueError("u must be a 1-D array of quantile levels strictly between 0 and 1")
    mu, var = np.asarray(mu), np.asarray(var)
    sd = np.sqrt(np.maximum(var, 0.0))
    z = mu + sd * special.ndtri(u).reshape((-1,) + (1,) * mu.ndim)  # (n_quantiles, ..., horizon)
    return np.moveaxis(transform.to_units(z, params), 0, -2)


@dataclass(frozen=True)
class LogTransform:
    """Log of the floored values over their fit-days mean, inverted as a lognormal mean.

    ``fit`` sets ``response_norm``, the mean of ``max(values, floor)`` over the fit days. ``forward`` maps values to
    ``log(max(values, floor) / response_norm)``, so a zero day becomes ``log(floor / response_norm)``.
    ``inverse`` returns ``exp(mu + var / 2) * response_norm``, the mean of a lognormal: the Jensen correction
    ``var / 2`` makes it an estimate of ``E[Y]`` rather than ``exp(E[log Y])``.

    Parameters
    ----------
    floor : float
        Floor applied to the values before the log, in the units of the data, by default 0.1.
    """

    floor: float = 0.1

    def _floored(self, values: np.ndarray) -> np.ndarray:
        return np.clip(values, self.floor, None).astype(np.float32)

    def fit(self, values: np.ndarray) -> dict[str, np.ndarray]:
        """Set ``response_norm``; see :meth:`Transform.fit`."""
        return {"response_norm": self._floored(values).mean(axis=-1)}

    def forward(self, values: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return ``log(max(values, floor) / response_norm)``; see :meth:`Transform.forward`."""
        return np.log(self._floored(values) / _per_series(params["response_norm"]))

    def inverse(self, mu: np.ndarray, var: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return ``exp(mu + var / 2) * response_norm``; see :meth:`Transform.inverse`."""
        return np.exp(mu + 0.5 * var) * _per_series(params["response_norm"])

    def to_units(self, z: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return ``exp(z) * response_norm``; see :meth:`Transform.to_units`."""
        return np.exp(z) * _per_series(params["response_norm"])

    def quantile(self, mu: np.ndarray, var: np.ndarray, u: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return the lognormal's quantiles; see :meth:`Transform.quantile`."""
        return _gaussian_quantile(self, mu, var, u, params)


#: Methods of :class:`PowerTransform`, as scikit-learn's ``PowerTransformer`` names them.
POWER_METHODS = ("box-cox", "yeo-johnson")


@dataclass(frozen=True)
class PowerTransform:
    """Box-Cox or Yeo-Johnson power of the values with a fitted λ per series, inverted as the mean of its forecast.

    The power applies to ``x = max(values, floor) + shift``. ``fit`` fits each series' ``lmbda`` by maximum
    likelihood on its fit days with scikit-learn's ``PowerTransformer``, then clips it to ``lmbda_bounds``. With
    ``standardize`` it sets ``loc`` and ``scale`` to the mean and standard deviation of the powered fit days, as
    ``PowerTransformer(standardize=True)`` does; without it they are 0 and 1. It also sets ``max_value``, the
    series' largest fit-day observation. ``forward`` returns ``(power(x, lmbda) - loc) / scale``.

    Box-Cox needs ``x > 0``, so ``floor + shift`` must be positive. Yeo-Johnson accepts zeros. The observations are
    never negative, and on ``x >= 0`` Yeo-Johnson is the Box-Cox power of ``x + 1``, with the same maximum-likelihood
    λ: ``method="yeo-johnson"`` fits what ``method="box-cox"`` fits with one more unit of ``shift``.

    ``inverse`` maps a Gaussian forecast ``z`` back by ``loc`` and ``scale`` and returns the mean of the observations it
    implies, not the inverse power of its mean, which is their median. A Gaussian puts mass where no observation
    maps to: below the power of zero and, with ``lmbda < 0``, whose power is bounded above, beyond the power of
    any observation. The inverse power of ``z`` is therefore censored to the series' admissible range, from 0 to
    ``max_value``, and the forecast is the mean of the censored observations, ``w_0 + ∫ P(W > w) dw`` over that range,
    ``W`` the censored inverse power and ``w_0`` its least value. The integral is a Gauss-Legendre quadrature
    in ``log w``, where the integrand is smooth, on either side of the median, where it falls through 1/2. The
    forecast is never negative, nor below ``floor``.

    Parameters
    ----------
    method : str
        ``"box-cox"`` or ``"yeo-johnson"``, by default ``"box-cox"``.
    shift : float
        Added to the floored values before the power and subtracted after its inverse, by default 0.
    floor : float
        Floor applied to the values before the shift, as :class:`LogTransform` floors them, by default 0.
    standardize : bool
        Centre and scale each series' powered values on its fit days, by default True.
    lmbda_bounds : tuple of float, optional
        Bounds the fitted λ is clipped to, by default ``(-1.0, 2.0)``, the range R's ``forecast::BoxCox.lambda``
        searches. The profile log-likelihood of λ is concave, so the clipped λ is the maximum-likelihood λ within
        the bounds. ``None`` leaves λ unbounded: on intermittent retail sales, for one, most series' λ is then below
        -1 and some below -30, a power that maps every day with sales to almost one value and whose inverse loses all
        precision.
    n_nodes : int
        Gauss-Legendre nodes on each side of the median in ``inverse``, by default 24.
    """

    method: str = "box-cox"
    shift: float = 0.0
    floor: float = 0.0
    standardize: bool = True
    lmbda_bounds: tuple[float, float] | None = (-1.0, 2.0)
    n_nodes: int = 24

    def __post_init__(self) -> None:
        if self.method not in POWER_METHODS:
            raise ValueError(f"method must be one of {POWER_METHODS}, got {self.method!r}")
        if self.floor < 0 or self.shift < 0:
            raise ValueError(f"floor and shift must not be negative, got floor={self.floor}, shift={self.shift}")
        if self.method == "box-cox" and self.floor + self.shift <= 0:
            raise ValueError("Box-Cox needs positive input: set floor or shift above 0")

    @property
    def _offset(self) -> float:
        """What Yeo-Johnson adds to ``x`` before its Box-Cox power: 1, and 0 for Box-Cox."""
        return float(self.method == "yeo-johnson")

    def _input(self, values: Any) -> np.ndarray:
        """The power's input ``x = max(values, floor) + shift``, in float64."""
        return np.maximum(np.asarray(values, dtype=np.float64), self.floor) + self.shift

    def fit(self, values: np.ndarray) -> dict[str, np.ndarray]:
        """Set ``lmbda``, ``loc``, ``scale`` and ``max_value``; see :meth:`Transform.fit`.

        A series whose fit days all have the same value has no maximum-likelihood λ (scipy's Box-Cox raises on
        it), so its ``lmbda`` is 1 before the bounds.
        """
        try:
            from sklearn.preprocessing import PowerTransformer
        except ModuleNotFoundError as exc:  # pragma: no cover - only hit without the optim extra
            raise ModuleNotFoundError(
                "PowerTransform requires scikit-learn, part of the optional 'optim' extra. "
                "Install it with: pip install 'bunobee[optim]'"
            ) from exc

        x = self._input(values)
        rows = x.reshape(-1, x.shape[-1])  # (n_series, n_steps)
        lmbda = np.ones(rows.shape[0])
        varying = np.ptp(rows, axis=1) > 0
        if varying.any():
            # PowerTransformer fits one λ per column, so each series is a column.
            lmbda[varying] = PowerTransformer(method=self.method, standardize=False).fit(rows[varying].T).lambdas_
        if self.lmbda_bounds is not None:
            lmbda = np.clip(lmbda, *self.lmbda_bounds)

        loc, scale = np.zeros_like(lmbda), np.ones_like(lmbda)
        if self.standardize:
            powered = special.boxcox(rows + self._offset, lmbda[:, None])
            loc, scale = powered.mean(axis=1), powered.std(axis=1)
            scale[scale == 0] = 1.0  # as StandardScaler leaves a constant column unscaled
        max_value = np.asarray(values, dtype=np.float64).reshape(rows.shape).max(axis=1)
        shape = x.shape[:-1]
        return {
            "lmbda": lmbda.reshape(shape),
            "loc": loc.reshape(shape),
            "scale": scale.reshape(shape),
            "max_value": max_value.reshape(shape),
        }

    def forward(self, values: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return ``(power(max(values, floor) + shift, lmbda) - loc) / scale``; see :meth:`Transform.forward`."""
        powered = special.boxcox(self._input(values) + self._offset, _per_series(params["lmbda"]))
        return ((powered - _per_series(params["loc"])) / _per_series(params["scale"])).astype(np.float32)

    def inverse(self, mu: np.ndarray, var: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return the mean of the observations the Gaussian forecast implies; see :meth:`Transform.inverse`."""
        lmbda, loc, scale, max_value = (
            _per_series(np.asarray(params[key], dtype=np.float64)) for key in ("lmbda", "loc", "scale", "max_value")
        )
        # W, the Box-Cox argument the Gaussian z maps back to, censored to the admissible range: 0 to max_value.
        w_low, w_high = self._input(0.0) + self._offset, self._input(max_value) + self._offset
        centre = loc + scale * np.asarray(mu, dtype=np.float64)
        spread = np.maximum(scale * np.sqrt(np.asarray(var, dtype=np.float64)), np.finfo(np.float64).tiny)
        z_low, z_high = special.boxcox(w_low, lmbda), special.boxcox(w_high, lmbda)
        median = np.clip(special.inv_boxcox(np.clip(centre, z_low, z_high), lmbda), w_low, w_high)

        # E[W] = w_low + the integral of P(W > w) = P(z > boxcox(w)) over (w_low, w_high), taken over u = log w.
        nodes, weights = np.polynomial.legendre.leggauss(self.n_nodes)
        mean = np.full(np.broadcast_shapes(centre.shape, spread.shape, np.shape(w_high)), w_low)
        # A near-zero spread overflows the standardised distance to ±inf, which ndtr maps to 0 or 1 as it should.
        with np.errstate(over="ignore"):
            for start, stop in ((np.log(w_low), np.log(median)), (np.log(median), np.log(w_high))):
                half = (stop - start) / 2
                # One node at a time, so no (..., n_steps, n_nodes) array is built for an in-sample fit.
                for node, weight in zip(nodes, weights):
                    w = np.exp(start + half * (1 + node))
                    mean += weight * half * w * special.ndtr((centre - special.boxcox(w, lmbda)) / spread)
        return np.maximum(mean - self._offset - self.shift, 0.0).astype(np.result_type(mu, var))

    def to_units(self, z: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return the inverse power of ``z``, censored to the series' admissible range; see :meth:`Transform.to_units`.

        As in :meth:`inverse`, ``z`` is censored to the power of zero and, where the power is bounded above, of
        the largest observation, so a value beyond them maps to 0 or ``max_value`` and nothing maps to a NaN.
        """
        lmbda, loc, scale, max_value = (
            _per_series(np.asarray(params[key], dtype=np.float64)) for key in ("lmbda", "loc", "scale", "max_value")
        )
        w_low, w_high = self._input(0.0) + self._offset, self._input(max_value) + self._offset
        z_low, z_high = special.boxcox(w_low, lmbda), special.boxcox(w_high, lmbda)
        centre = loc + scale * np.asarray(z, dtype=np.float64)
        w = np.clip(special.inv_boxcox(np.clip(centre, z_low, z_high), lmbda), w_low, w_high)
        return np.maximum(w - self._offset - self.shift, 0.0).astype(np.result_type(z, np.float32))

    def quantile(self, mu: np.ndarray, var: np.ndarray, u: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return the quantiles of the censored observations the Gaussian implies; see :meth:`Transform.quantile`."""
        return _gaussian_quantile(self, mu, var, u, params)
