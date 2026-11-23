"""The shared shell of the bunobee models: configure, ``fit``, then ``predict``.

Every model is configured at construction, fitted with :meth:`Forecaster.fit` and forecast with
:meth:`Forecaster.predict`. What the models share lives here: running NUTS on the model's NumPyro model, keeping the
posterior as an ``xr.Dataset`` on ``idata_`` and the sampler on ``mcmc_``, and refusing to forecast before a fit.

The numerical core stays functional. The filters in :mod:`bunobee.models.ssp` and the recursion in
:mod:`bunobee.models.dlt` are pure JAX functions, so ``jit``, ``vmap`` and ``grad`` work on them unchanged, and a
model class only calls them. :meth:`Forecaster.model` must be pure in its arguments: NumPyro traces it, so the data
comes in as arguments and the instance holds configuration only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

import jax
import xarray as xr
from numpyro.infer import MCMC, NUTS, init_to_uniform

#: Sampler settings :meth:`Forecaster.fit` uses for any key ``mcmc_args`` leaves out.
DEFAULT_MCMC_ARGS: dict[str, Any] = {"num_warmup": 500, "num_samples": 500, "num_chains": 1, "progress_bar": False}


class NotFittedError(RuntimeError):
    """Raised when a model is asked for its posterior or a forecast before :meth:`Forecaster.fit`."""


class Forecaster(ABC):
    """Base class of the bunobee models.

    A subclass supplies four things:

    - :meth:`model`, the NumPyro model, pure in its arguments.
    - :meth:`fit`, with the model's own data arguments. It turns them into the model's inputs and hands those to
      :meth:`_fit`, which runs NUTS and stores the posterior.
    - :meth:`_to_idata`, which turns the chain-grouped samples into the posterior ``xr.Dataset``.
    - :meth:`predict`, the forecast from the stored posterior.

    It may also override :meth:`_init_strategy`, by default NumPyro's ``init_to_uniform``.
    """

    def __init__(self) -> None:
        self._idata: xr.Dataset | None = None
        self._mcmc: MCMC | None = None

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
        """Turn ``mcmc.get_samples(group_by_chain=True)`` into the posterior ``xr.Dataset``.

        ``inputs`` are the model's inputs and ``context`` is whatever else :meth:`fit` passed to :meth:`_fit`.
        """

    def _init_strategy(self) -> Callable:
        """The NUTS init strategy, by default ``init_to_uniform``."""
        return init_to_uniform

    def _fit(
        self,
        inputs: Mapping[str, Any],
        *,
        mcmc_args: Mapping[str, Any] | None,
        seed: int,
        context: Mapping[str, Any] | None = None,
    ) -> Forecaster:
        """Run NUTS on :meth:`model` with ``inputs``, then store the sampler and the posterior.

        ``mcmc_args`` is passed to ``numpyro.infer.MCMC`` on top of :data:`DEFAULT_MCMC_ARGS`. ``context`` holds
        what :meth:`_to_idata` needs beyond the model's inputs, e.g. coordinate labels. The posterior is stored only
        once sampling and :meth:`_to_idata` both succeed, so a failed refit keeps the previous fit.
        """
        args = {**DEFAULT_MCMC_ARGS, **dict(mcmc_args or {})}
        mcmc = MCMC(NUTS(self.model, init_strategy=self._init_strategy()), **args)
        mcmc.run(jax.random.PRNGKey(seed), **inputs)
        idata = self._to_idata(mcmc.get_samples(group_by_chain=True), inputs, dict(context or {}))
        self._mcmc, self._idata = mcmc, idata
        return self

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has stored a posterior."""
        return self._idata is not None

    @property
    def idata_(self) -> xr.Dataset:
        """The posterior from the last :meth:`fit`, with leading ``(chain, draw)`` dims."""
        if self._idata is None:
            raise NotFittedError(f"{type(self).__name__} is not fitted yet; call fit() first")
        return self._idata

    @property
    def mcmc_(self) -> MCMC:
        """The ``numpyro.infer.MCMC`` of the last :meth:`fit`, for diagnostics such as ``print_summary()``."""
        if self._mcmc is None:
            raise NotFittedError(f"{type(self).__name__} is not fitted yet; call fit() first")
        return self._mcmc
