"""The shared shell of the bunobee models: configure, ``fit``, then ``predict``.

Every model is configured at construction, fitted with :meth:`Forecaster.fit` and forecast with
:meth:`Forecaster.predict`. What the models share lives here: fitting the model's NumPyro model with its inference
method (see :mod:`bunobee.models.inference`), keeping the posterior as an ``xr.Dataset`` on ``idata_`` and what the
method leaves behind on ``mcmc_`` (NUTS) or ``guide_`` / ``params_`` / ``losses_`` (SVI, MAP), and refusing to
forecast before a fit.

The numerical core stays functional. The filters in :mod:`bunobee.models.ssp` and the recursion in
:mod:`bunobee.models.dlt` are pure JAX functions, so ``jit``, ``vmap`` and ``grad`` work on them unchanged, and a
model class only calls them. :meth:`Forecaster.model` must be pure in its arguments: NumPyro traces it, so the data
comes in as arguments and the instance holds configuration only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import jax
import xarray as xr
from numpyro.infer import MCMC, init_to_uniform

from bunobee.models.inference import DEFAULT_MCMC_ARGS, MAP, NUTS, SVI, Inference, Posterior

# The inference names are re-exported so they can be imported next to ``Forecaster``.
__all__ = ["DEFAULT_MCMC_ARGS", "Forecaster", "Inference", "MAP", "NUTS", "NotFittedError", "Posterior", "SVI"]


class NotFittedError(RuntimeError):
    """Raised when a model is asked for its posterior or a forecast before :meth:`Forecaster.fit`."""


class Forecaster(ABC):
    """Base class of the bunobee models.

    A subclass supplies four things:

    - :meth:`model`, the NumPyro model, pure in its arguments.
    - :meth:`fit`, with the model's own data arguments. It turns them into the model's inputs and hands those to
      :meth:`_fit`, which runs the inference method and stores the posterior.
    - :meth:`_to_idata`, which turns the chain-grouped samples into the posterior ``xr.Dataset``.
    - :meth:`predict`, the forecast from the stored posterior.

    It may also override :meth:`_init_strategy`, by default NumPyro's ``init_to_uniform``, and declare
    :attr:`inference_methods` and :attr:`series_plate`.

    Parameters
    ----------
    inference : Inference | None
        How :meth:`fit` fits the model, by default ``NUTS()``; see :mod:`bunobee.models.inference`. It must be an
        instance of one of :attr:`inference_methods`, and may subsample only a model with a :attr:`series_plate`.

    Raises
    ------
    ValueError
        If the model does not accept ``inference``.
    """

    #: The inference classes the model accepts; any other raises at construction.
    inference_methods: ClassVar[tuple[type, ...]] = (NUTS,)

    #: The name of the model's plate over series, or ``None`` when it has none it can subsample. A model with one
    #: opens it as ``with numpyro.plate(series_plate, n_series) as idx:`` and indexes its per-series data with
    #: ``idx``, so ``SVI(subsample=k)`` can draw ``k`` of the series at each step.
    series_plate: ClassVar[str | None] = None

    def __init__(self, inference: Inference | None = None) -> None:
        inference = NUTS() if inference is None else inference
        self._check_inference(inference)
        self.inference = inference
        self._idata: xr.Dataset | None = None
        self._posterior: Posterior | None = None

    def _check_inference(self, inference: Inference) -> None:
        name = type(self).__name__
        if not isinstance(inference, self.inference_methods):
            accepted = ", ".join(cls.__name__ for cls in self.inference_methods)
            raise ValueError(f"{name} accepts inference by {accepted}, got {inference!r}")
        if isinstance(inference, SVI) and inference.subsample is not None and self.series_plate is None:
            raise ValueError(
                f"{name} has no series plate to subsample, so it fits by {inference.name} with subsample=None only"
            )

    @abstractmethod
    def model(self, **inputs: Any) -> None:
        """The NumPyro model. Its keyword arguments are the inputs :meth:`fit` builds."""

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> Forecaster:
        """Fit the model to data and return ``self``."""

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> xr.Dataset:
        """Forecast from the fitted posterior."""

    @abstractmethod
    def _to_idata(
        self, samples: Mapping[str, jax.Array], inputs: Mapping[str, Any], context: Mapping[str, Any]
    ) -> xr.Dataset:
        """Turn the chain-grouped samples of a :class:`~bunobee.models.inference.Posterior` into an ``xr.Dataset``.

        ``inputs`` are the model's inputs and ``context`` is whatever else :meth:`fit` passed to :meth:`_fit`.
        """

    def _init_strategy(self) -> Callable:
        """The init strategy of NUTS and of the SVI guides, by default ``init_to_uniform``."""
        return init_to_uniform

    def _inference_for(self, mcmc_args: Mapping[str, Any] | None) -> Inference:
        """The inference of one fit: :attr:`inference`, with ``mcmc_args`` on top when it is NUTS."""
        if mcmc_args is None:
            return self.inference
        if not isinstance(self.inference, NUTS):
            raise ValueError(
                f"mcmc_args only applies to NUTS, but this {type(self).__name__} fits by {self.inference.name}; "
                "configure it through inference= instead"
            )
        return self.inference.updated(mcmc_args)

    def _fit(
        self,
        inputs: Mapping[str, Any],
        *,
        mcmc_args: Mapping[str, Any] | None,
        seed: int,
        context: Mapping[str, Any] | None = None,
    ) -> Forecaster:
        """Fit :meth:`model` to ``inputs`` with the model's inference, then store the posterior.

        ``mcmc_args``, when given, is put on top of the model's ``NUTS`` settings for this fit only, and is an error
        under any other inference. ``context`` holds what :meth:`_to_idata` needs beyond the model's inputs, e.g.
        coordinate labels. The posterior is stored only once the fit and :meth:`_to_idata` both succeed, so a failed
        refit keeps the previous fit.
        """
        inference = self._inference_for(mcmc_args)
        posterior = inference.run(self, inputs, jax.random.PRNGKey(seed))
        idata = self._to_idata(posterior.samples, inputs, dict(context or {}))
        idata = idata.assign_attrs(inference=posterior.method)
        self._posterior, self._idata = posterior, idata
        return self

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has stored a posterior."""
        return self._idata is not None

    @property
    def idata_(self) -> xr.Dataset:
        """The posterior from the last :meth:`fit`, with leading ``(chain, draw)`` dims.

        ``attrs["inference"]`` names the method that fitted it, e.g. ``"NUTS"``.
        """
        if self._idata is None:
            raise NotFittedError(f"{type(self).__name__} is not fitted yet; call fit() first")
        return self._idata

    def _extra(self, key: str) -> Any:
        if self._posterior is None:
            raise NotFittedError(f"{type(self).__name__} is not fitted yet; call fit() first")
        if key not in self._posterior.extras:
            raise AttributeError(f"{type(self).__name__} was fitted by {self._posterior.method}, which has no {key}_")
        return self._posterior.extras[key]

    @property
    def mcmc_(self) -> MCMC:
        """The ``numpyro.infer.MCMC`` of the last NUTS :meth:`fit`, for diagnostics such as ``print_summary()``."""
        return self._extra("mcmc")

    @property
    def guide_(self) -> Callable:
        """The autoguide of the last SVI or MAP :meth:`fit`."""
        return self._extra("guide")

    @property
    def params_(self) -> dict[str, jax.Array]:
        """The fitted guide parameters of the last SVI or MAP :meth:`fit`."""
        return self._extra("params")

    @property
    def losses_(self) -> jax.Array:
        """The ELBO loss at each step of the last SVI or MAP :meth:`fit`."""
        return self._extra("losses")
