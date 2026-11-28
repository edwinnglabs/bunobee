"""Swappable inference for the bunobee models: :class:`NUTS`, :class:`SVI` or :class:`MAP`.

A model is fitted by the inference method it was constructed with (``SSP(prior, inference=SVI())``). Each method
implements the :class:`Inference` protocol: ``run(model, inputs, key)`` fits ``model.model`` to ``inputs`` and returns
a :class:`Posterior`, whose samples are grouped by chain for every method, so the posterior always has leading
``(chain, draw)`` dims. SVI and MAP draws come back as one chain.

- :class:`NUTS` is the sampler :meth:`~bunobee.models.base.Forecaster.fit` has always run, with the model's
  ``_init_strategy()``.
- :class:`SVI` fits an ``AutoNormal`` (or ``AutoDelta``) guide with a one-cycle Adam schedule and global-norm gradient
  clipping, then draws the posterior from the guide. With ``subsample=k`` each step sees ``k`` series, drawn from the
  model's series plate (:attr:`~bunobee.models.base.Forecaster.series_plate`), which is how a pooled model over
  tens of thousands of series fits when NUTS cannot.
- :class:`MAP` is SVI with an ``AutoDelta`` guide and one draw.

SVI and MAP need ``optax``, from the optional ``optim`` extra: ``pip install 'bunobee[optim]'``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import jax
from numpyro import handlers
from numpyro.infer import MCMC, Predictive, Trace_ELBO
from numpyro.infer import NUTS as _NUTSKernel
from numpyro.infer import SVI as _SVI
from numpyro.infer.autoguide import AutoDelta, AutoNormal
from numpyro.primitives import Messenger

if TYPE_CHECKING:
    from bunobee.models.base import Forecaster

#: Sampler settings :class:`NUTS` uses for any key it is not given.
DEFAULT_MCMC_ARGS: dict[str, Any] = {"num_warmup": 500, "num_samples": 500, "num_chains": 1, "progress_bar": False}

#: The guides :class:`SVI` accepts, by name.
GUIDES: dict[str, type] = {"normal": AutoNormal, "delta": AutoDelta}


@dataclass(frozen=True)
class Posterior:
    """What an :class:`Inference` run returns.

    Attributes
    ----------
    samples : dict[str, jax.Array]
        Every latent and deterministic site of the model, each with leading ``(chain, draw)`` dims, as
        ``mcmc.get_samples(group_by_chain=True)`` returns them.
    method : str
        The name of the inference method, e.g. ``"NUTS"``; the fitted model stores it as ``idata_.attrs["inference"]``.
    extras : dict[str, Any]
        What is specific to the method: ``mcmc`` for NUTS; ``guide``, ``params`` and ``losses`` for SVI and MAP.
    """

    samples: dict[str, jax.Array]
    method: str
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Inference(Protocol):
    """An inference method: fits a model's NumPyro model and returns its :class:`Posterior`."""

    #: Short name of the method, stored as ``idata_.attrs["inference"]``.
    name: str

    def run(self, model: Forecaster, inputs: Mapping[str, Any], key: jax.Array) -> Posterior:
        """Fit ``model.model`` to ``inputs`` with the random ``key`` and return the posterior."""
        ...


class NUTS:
    """NUTS through ``numpyro.infer.MCMC``, started from the model's ``_init_strategy()``.

    Parameters
    ----------
    num_warmup, num_samples, num_chains : int
        Passed to ``numpyro.infer.MCMC``, by default 500, 500 and 1.
    progress_bar : bool
        Show the sampler's progress bar, by default ``False``.
    kernel_args : Mapping | None
        Keyword arguments for the ``numpyro.infer.NUTS`` kernel, e.g. ``target_accept_prob``.
    **mcmc_args
        Any other keyword argument of ``numpyro.infer.MCMC``, e.g. ``thinning`` or ``chain_method``.

    Examples
    --------
    >>> NUTS(num_warmup=200, num_samples=200).mcmc_args["num_samples"]
    200
    """

    name = "NUTS"

    def __init__(
        self,
        num_warmup: int = DEFAULT_MCMC_ARGS["num_warmup"],
        num_samples: int = DEFAULT_MCMC_ARGS["num_samples"],
        num_chains: int = DEFAULT_MCMC_ARGS["num_chains"],
        *,
        progress_bar: bool = DEFAULT_MCMC_ARGS["progress_bar"],
        kernel_args: Mapping[str, Any] | None = None,
        **mcmc_args: Any,
    ) -> None:
        #: The keyword arguments of ``numpyro.infer.MCMC``.
        self.mcmc_args: dict[str, Any] = {
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_chains": num_chains,
            "progress_bar": progress_bar,
            **mcmc_args,
        }
        #: The keyword arguments of the ``numpyro.infer.NUTS`` kernel.
        self.kernel_args: dict[str, Any] = dict(kernel_args or {})

    def updated(self, mcmc_args: Mapping[str, Any]) -> NUTS:
        """Return a copy with ``mcmc_args`` on top of this one's ``mcmc_args``."""
        return NUTS(kernel_args=self.kernel_args, **{**self.mcmc_args, **dict(mcmc_args)})

    def run(self, model: Forecaster, inputs: Mapping[str, Any], key: jax.Array) -> Posterior:
        """Sample ``model.model`` by NUTS; the posterior keeps the ``numpyro.infer.MCMC`` as ``extras["mcmc"]``."""
        kernel = _NUTSKernel(model.model, init_strategy=model._init_strategy(), **self.kernel_args)
        mcmc = MCMC(kernel, **self.mcmc_args)
        mcmc.run(key, **inputs)
        return Posterior(mcmc.get_samples(group_by_chain=True), self.name, {"mcmc": mcmc})

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.mcmc_args.items())
        return f"NUTS({args}, kernel_args={self.kernel_args!r})"


class SVI:
    """Stochastic variational inference with an autoguide, then posterior draws from the guide.

    The ELBO is minimised by Adam on a one-cycle cosine schedule that peaks at ``peak_lr`` (or on ``schedule``), with
    gradients clipped to a global norm of ``clip_norm``. The guide starts from the model's ``_init_strategy()``.

    Parameters
    ----------
    guide : {"normal", "delta"}
        ``"normal"`` for ``AutoNormal`` (the default), ``"delta"`` for ``AutoDelta``, a point estimate.
    subsample : int | None
        Series per step. When set, each step subsamples the model's series plate
        (:attr:`~bunobee.models.base.Forecaster.series_plate`) to ``subsample`` series, and the likelihood inside it
        is scaled up to all of them. The posterior is still drawn for every series. By default ``None``, every series
        at every step.
    steps : int
        Optimisation steps, by default 2000.
    peak_lr : float
        Peak learning rate of the one-cycle schedule, by default 0.03.
    clip_norm : float
        Global norm the gradients are clipped to, by default 10.
    num_draws : int
        Posterior draws taken from the fitted guide, by default 500. They come back as one chain.
    progress_bar : bool
        Show the optimiser's progress bar, by default ``False``.
    schedule : Callable | None
        The learning rate of each step, an ``optax`` schedule such as ``optax.linear_onecycle_schedule(steps, 0.03)``.
        By default ``None``, ``optax.cosine_onecycle_schedule(steps, peak_lr)``; ``peak_lr`` is not used otherwise.

    Examples
    --------
    >>> SVI(subsample=1500, steps=5000)  # doctest: +ELLIPSIS
    SVI(guide='normal', subsample=1500, steps=5000, ...)
    """

    name = "SVI"

    def __init__(
        self,
        guide: str = "normal",
        subsample: int | None = None,
        steps: int = 2000,
        peak_lr: float = 0.03,
        clip_norm: float = 10.0,
        num_draws: int = 500,
        *,
        progress_bar: bool = False,
        schedule: Callable[[Any], Any] | None = None,
    ) -> None:
        if guide not in GUIDES:
            raise ValueError(f"guide must be one of {sorted(GUIDES)}, got {guide!r}")
        if subsample is not None and not (isinstance(subsample, int) and subsample >= 1):
            raise ValueError(f"subsample must be a positive int or None, got {subsample!r}")
        if not (isinstance(steps, int) and steps >= 1):
            raise ValueError(f"steps must be a positive int, got {steps!r}")
        if not (isinstance(num_draws, int) and num_draws >= 1):
            raise ValueError(f"num_draws must be a positive int, got {num_draws!r}")
        if not peak_lr > 0:
            raise ValueError(f"peak_lr must be positive, got {peak_lr}")
        if not clip_norm > 0:
            raise ValueError(f"clip_norm must be positive, got {clip_norm}")
        if schedule is not None and not callable(schedule):
            raise ValueError(f"schedule must be an optax schedule (a callable of the step) or None, got {schedule!r}")
        self.guide = guide
        self.subsample = subsample
        self.steps = steps
        self.peak_lr = float(peak_lr)
        self.clip_norm = float(clip_norm)
        self.num_draws = num_draws
        self.progress_bar = progress_bar
        self.schedule = schedule

    def run(self, model: Forecaster, inputs: Mapping[str, Any], key: jax.Array) -> Posterior:
        """Fit the guide to ``model.model``, then draw the posterior from it.

        The posterior keeps the guide, its fitted parameters and the loss per step as ``extras["guide"]``,
        ``extras["params"]`` and ``extras["losses"]``.
        """
        optim = self._optimizer()
        fit_key, draw_key = jax.random.split(key)
        init_loc_fn = model._init_strategy()
        guide_cls = GUIDES[self.guide]

        train_model = model.model
        if self.subsample is not None:
            if model.series_plate is None:
                raise ValueError(f"{type(model).__name__} has no series plate, so SVI cannot subsample it")
            train_model = _subsample_plate(model.model, model.series_plate, self.subsample)
        guide = guide_cls(train_model, init_loc_fn=init_loc_fn)
        result = _SVI(train_model, guide, optim, Trace_ELBO()).run(
            fit_key, self.steps, progress_bar=self.progress_bar, **inputs
        )

        # A guide fitted on subsamples draws only a subsample; a fresh guide over the full plate reads the same
        # full-size parameters and draws every series.
        draw_guide = guide if self.subsample is None else guide_cls(model.model, init_loc_fn=init_loc_fn)
        draws = Predictive(
            model.model,
            guide=draw_guide,
            params=result.params,
            num_samples=self.num_draws,
            return_sites=_posterior_sites(model.model, inputs),
        )(draw_key, **inputs)
        samples = {name: value[None] for name, value in draws.items()}
        return Posterior(samples, self.name, {"guide": guide, "params": result.params, "losses": result.losses})

    def _optimizer(self) -> Any:
        try:
            import optax
            from numpyro.optim import optax_to_numpyro
        except ModuleNotFoundError as exc:  # pragma: no cover - only hit without the optim extra
            raise ModuleNotFoundError(
                f"{type(self).__name__} requires the optional 'optim' extra (missing: {exc.name}). "
                "Install it with: pip install 'bunobee[optim]'"
            ) from exc
        schedule = self.schedule
        if schedule is None:
            schedule = optax.cosine_onecycle_schedule(transition_steps=self.steps, peak_value=self.peak_lr)
        return optax_to_numpyro(optax.chain(optax.clip_by_global_norm(self.clip_norm), optax.adam(schedule)))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(guide={self.guide!r}, subsample={self.subsample!r}, steps={self.steps}, "
            f"peak_lr={self.peak_lr}, clip_norm={self.clip_norm}, num_draws={self.num_draws})"
        )


class MAP(SVI):
    """Maximum a posteriori: :class:`SVI` with an ``AutoDelta`` guide and one draw.

    The posterior has ``chain=1`` and ``draw=1``: the point estimate of every latent site, with the deterministic
    sites computed at it.

    Parameters
    ----------
    steps, peak_lr, clip_norm, subsample, progress_bar, schedule
        As for :class:`SVI`.

    Examples
    --------
    >>> MAP(steps=500).num_draws
    1
    """

    name = "MAP"

    def __init__(
        self,
        steps: int = 2000,
        peak_lr: float = 0.03,
        clip_norm: float = 10.0,
        subsample: int | None = None,
        *,
        progress_bar: bool = False,
        schedule: Callable[[Any], Any] | None = None,
    ) -> None:
        super().__init__(
            guide="delta",
            subsample=subsample,
            steps=steps,
            peak_lr=peak_lr,
            clip_norm=clip_norm,
            num_draws=1,
            progress_bar=progress_bar,
            schedule=schedule,
        )

    def __repr__(self) -> str:
        return (
            f"MAP(steps={self.steps}, peak_lr={self.peak_lr}, clip_norm={self.clip_norm}, "
            f"subsample={self.subsample!r})"
        )


class _subsample_plate(Messenger):
    """Run ``fn`` with the plate ``name`` subsampled to ``size``, as ``numpyro.plate(name, n, subsample_size=size)``."""

    def __init__(self, fn: Callable, name: str, size: int) -> None:
        self.name = name
        self.size = size
        super().__init__(fn)

    def process_message(self, msg: dict) -> None:
        if msg["type"] != "plate" or msg["name"] != self.name:
            return
        full_size = msg["args"][0]
        if self.size > full_size:
            raise ValueError(f"subsample={self.size} is more than the {full_size} series of plate {self.name!r}")
        msg["args"] = (full_size, self.size)
        if self.size != full_size:
            # None makes the outer seed handler draw the subsample indices.
            msg["value"] = None


def _posterior_sites(model: Callable, inputs: Mapping[str, Any]) -> list[str]:
    """The sites ``mcmc.get_samples()`` would return for ``model``: its latent sample sites and deterministic ones."""
    trace = handlers.trace(handlers.seed(model, jax.random.PRNGKey(0))).get_trace(**inputs)
    return [
        name
        for name, site in trace.items()
        if (site["type"] == "sample" and not site["is_observed"]) or site["type"] == "deterministic"
    ]
