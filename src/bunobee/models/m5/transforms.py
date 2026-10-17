"""Target transforms of the M5 SSP models: the scale the sales are fitted on, and the way back to units.

The MAP fitters in :mod:`bunobee.models.m5.m5_ssp_optim` fit a linear-Gaussian state-space model to the
transformed sales of each series, and their predictors map the Gaussian forecast back to a point forecast in
units. A :class:`Transform` owns both directions, so trying another target needs a new transform, not an edit
of the fitter. :class:`LogTransform` is the default.

Unlike :mod:`bunobee.models.m5.m5_ssp_optim`, this module needs only ``numpy``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


def _per_series(value: Any) -> Any:
    """Add a trailing day axis to a per-series parameter, so it broadcasts over the days.

    A scalar is returned as it is, so a Python float keeps its weak dtype and a float32 forecast stays float32.
    """
    return value if np.isscalar(value) else np.asarray(value)[..., None]


class Transform(Protocol):
    """A per-series map from sales to the scale an SSP model is fitted on, and back to units.

    ``fit`` fits each series' parameters on its fit days, ``forward`` maps sales to the transformed scale and
    ``inverse`` maps a Gaussian forecast on that scale back to a point forecast in units. The inverse of the
    forecast mean is not the mean of the sales (``exp(mu)`` is a lognormal's median), so ``inverse`` takes the
    forecast variance too and carries the bias correction.

    Arrays hold one series per row along their leading dimensions and its days along the last one.
    """

    def fit(self, sales: np.ndarray) -> dict[str, np.ndarray]:
        """Fit the per-series parameters on the fit days.

        Parameters
        ----------
        sales : np.ndarray
            Daily unit sales over the fit days, of shape ``(..., n_steps)``.

        Returns
        -------
        dict of str to np.ndarray
            The parameters by name, each of shape ``(...)``: one value per series. The fitters return them
            among their outputs, so a name must not be one of the fitter's own keys (``at``, ``sigma_h``, ...).
        """
        ...

    def forward(self, sales: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Map sales to the transformed scale.

        Parameters
        ----------
        sales : np.ndarray
            Daily unit sales, of shape ``(..., n_steps)``.
        params : Mapping of str to Any
            Holds at least the parameters ``fit`` returned, for the same series.

        Returns
        -------
        np.ndarray
            The transformed sales, of shape ``(..., n_steps)``.
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


@dataclass(frozen=True)
class LogTransform:
    """Log of the floored sales over their fit-days mean, inverted as a lognormal mean.

    ``fit`` sets ``response_norm``, the mean of ``max(sales, floor)`` over the fit days. ``forward`` maps sales to
    ``log(max(sales, floor) / response_norm)``, so a zero-sales day becomes ``log(floor / response_norm)``.
    ``inverse`` returns ``exp(mu + var / 2) * response_norm``, the mean of a lognormal: the Jensen correction
    ``var / 2`` makes it an estimate of ``E[Y]`` rather than ``exp(E[log Y])``.

    Parameters
    ----------
    floor : float
        Floor applied to the sales before the log, by default 0.1.
    """

    floor: float = 0.1

    def _floored(self, sales: np.ndarray) -> np.ndarray:
        return np.clip(sales, self.floor, None).astype(np.float32)

    def fit(self, sales: np.ndarray) -> dict[str, np.ndarray]:
        """Set ``response_norm``; see :meth:`Transform.fit`."""
        return {"response_norm": self._floored(sales).mean(axis=-1)}

    def forward(self, sales: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return ``log(max(sales, floor) / response_norm)``; see :meth:`Transform.forward`."""
        return np.log(self._floored(sales) / _per_series(params["response_norm"]))

    def inverse(self, mu: np.ndarray, var: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return ``exp(mu + var / 2) * response_norm``; see :meth:`Transform.inverse`."""
        return np.exp(mu + 0.5 * var) * _per_series(params["response_norm"])
