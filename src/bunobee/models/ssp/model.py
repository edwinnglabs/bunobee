"""The multi-series state-space model behind ``fit`` / ``predict``.

:class:`SSP` packages the NumPyro model the ``ssp_04`` to ``ssp_06`` prototypes wrote by hand: a shared latent state
filtered by one of the multi-series Kalman filters, with the filter's log likelihood as the likelihood, and priors on
the observation noise ``sigma_h`` and the process noise ``sigma_q``. The prior is the natural-scale SSP prior
(:class:`~bunobee.models.ssp.prior.SspPrior` or a bare ``xr.Dataset``); :meth:`SSP.predict` forecasts with
:func:`~bunobee.models.ssp.forecast.forecast_ssp`.

The filter is picked by name (see :data:`FILTERS`), and the name fixes everything that has to agree with it:

- ``"ekf"``: :func:`~bunobee.models.ssp.kalman_1d_st_ekf.kalman_filter_1d_ekf_st`. A positivity state is a
  log-intensity ``a`` whose natural value is ``exp(exponent * a)``, and the forecast link is ``"exp"``.
- ``"linear"``: :func:`~bunobee.models.ssp.kalman_1d_st.kalman_filter_1d_st`. A positivity state is on the natural
  scale and floored by the filter, and the forecast link is ``"identity"``.

So a posterior can no longer be forecast under the wrong positivity mechanism, the mistake ``forecast_ssp``'s
``link`` argument exists to catch.

A filter of your own is passed as a :class:`FilterSpec` instance, ``SSP(prior, filter=FilterSpec(...))``, without
touching :data:`FILTERS`. Its ``fn`` must follow :class:`SspFilter`. Give it a ``forecast`` (see
:class:`SspForecast`) whenever its state dynamics or observation noise differ from the random walk with Gaussian
noise that :func:`~bunobee.models.ssp.forecast.forecast_ssp` assumes, so the forecast follows the model that was
fitted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import jax.numpy as jnp
import numpy as np
import numpyro
import xarray as xr
from numpyro import distributions as dist

from bunobee.models.base import Forecaster
from bunobee.models.inference import MAP, NUTS, SVI, Inference
from bunobee.models.ssp.forecast import EXP_CLIP, FORECAST_LINKS, forecast_ssp
from bunobee.models.ssp.kalman_1d_st import kalman_filter_1d_st
from bunobee.models.ssp.kalman_1d_st_ekf import kalman_filter_1d_ekf_st
from bunobee.models.ssp.posterior import posterior_to_xarray
from bunobee.models.ssp.prior import SspPrior
from bunobee.models.ssp.transforms import _unwrap, transform_to_ekf_st, validate_prior


class SspFilter(Protocol):
    """The call :class:`SSP` makes to a filter inside its NumPyro model.

    Every argument is passed by keyword. A filter whose spec has ``nonlinear=True`` is also passed
    ``exponent=<float>``, the ``k`` of ``exp(k * a)``, so it must accept that keyword too.

    Parameters
    ----------
    a0 : jnp.ndarray
        Initial state mean, ``(n_states,)``, on the filter's scale (a-space for a nonlinear spec).
    P0 : jnp.ndarray
        Initial state covariance, ``(n_states, n_states)``.
    Z : jnp.ndarray
        Design, ``(n_steps, n_series, n_states)``.
    sigma_h : jnp.ndarray
        Observation noise scale per series, ``(n_series,)``.
    sigma_q : jnp.ndarray
        Process noise scale per state, ``(n_states,)``.
    y : jnp.ndarray
        Observations, ``(n_steps, n_series)``; NaN where ``mask`` is ``False``.
    logp : bool
        Always ``True``: the filter must return its log likelihood first.
    positivity : jnp.ndarray
        Boolean mask of the positivity states, ``(n_states,)``.
    a_obs, P_obs : jnp.ndarray | None
        The prior's disclosed states and their variances, ``(n_steps, n_states)``, or ``None``.
    mask : jnp.ndarray | None
        ``(n_steps, n_series)``, ``True`` where ``y`` is observed, or ``None`` when every entry is.

    Returns
    -------
    tuple
        ``(lp, at, ...)``: the scalar log likelihood, then the filtered states ``(n_steps, n_states)``. Anything
        after them is ignored.
    """

    def __call__(
        self,
        *,
        a0: jnp.ndarray,
        P0: jnp.ndarray,
        Z: jnp.ndarray,
        sigma_h: jnp.ndarray,
        sigma_q: jnp.ndarray,
        y: jnp.ndarray,
        logp: bool,
        positivity: jnp.ndarray,
        a_obs: jnp.ndarray | None,
        P_obs: jnp.ndarray | None,
        mask: jnp.ndarray | None,
    ) -> tuple[Any, ...]: ...


class SspForecast(Protocol):
    """The call :meth:`SSP.predict` makes to a :class:`FilterSpec`'s ``forecast``.

    Parameters
    ----------
    idata : xr.Dataset
        The fitted posterior, ``SSP.idata_``: every model site over ``(chain, draw, ...)``, the ``positivity``
        mask, and the ``exponent``, ``link``, ``filter`` and ``inference`` attrs.
    Z_future : np.ndarray
        Future design, ``(horizon, n_series, n_states)``.
    method : str
        ``predict``'s ``method``, passed through as given; the forecast decides which values it supports.
    noise_embed : bool
        Whether ``forecast_samples`` should include the observation noise.
    transform_callback : Callable | None
        To apply to ``forecast_samples`` only.
    seed : int
        Seed of the forecast's random draws.

    Returns
    -------
    xr.Dataset
        What ``predict`` returns, by convention ``forecast_samples`` and ``mu_samples`` (plus ``eps_samples`` with
        ``noise_embed``) over ``(sample, time, series)``, as :func:`~bunobee.models.ssp.forecast.forecast_ssp` does.
    """

    def __call__(
        self,
        idata: xr.Dataset,
        Z_future: np.ndarray,
        *,
        method: str,
        noise_embed: bool,
        transform_callback: Callable | None,
        seed: int,
    ) -> xr.Dataset: ...


@dataclass(frozen=True)
class FilterSpec:
    """One filter :class:`SSP` can fit with, how its positivity states are read, and how it is forecast.

    Without a ``forecast``, :meth:`SSP.predict` uses :func:`~bunobee.models.ssp.forecast.forecast_ssp` with
    ``link``. That is only valid for a filter whose state is a driftless random walk with ``N(0, sigma_q**2)``
    steps and whose observation noise is ``N(0, sigma_h**2)``: a filter with other dynamics (a transition matrix,
    seasonality) or another observation model needs its own ``forecast``, or it is fitted with one model and
    forecast with another.

    Examples
    --------
    >>> damped = functools.partial(kalman_filter_1d_st, T=0.9 * np.eye(3))  # doctest: +SKIP
    >>> spec = FilterSpec(damped, "identity", False, forecast=damped_forecast, name="damped")  # doctest: +SKIP
    >>> SSP(prior, filter=spec).fit(y, Z).predict(Z_future)  # doctest: +SKIP
    """

    #: The multi-series filter, called with ``logp=True`` as the likelihood; see :class:`SspFilter`.
    fn: SspFilter
    #: The ``forecast_ssp`` link of its posterior: ``"exp"`` or ``"identity"``. Any label when ``forecast`` is set.
    link: str
    #: Whether positivity states are log-intensities mapped by ``exp(exponent * a)``.
    nonlinear: bool
    #: Forecasts the fitted posterior in place of ``forecast_ssp``; see :class:`SspForecast`.
    forecast: SspForecast | None = None
    #: The filter's name in ``idata.attrs["filter"]``; by default the ``__name__`` of ``fn``.
    name: str | None = None

    @property
    def label(self) -> str:
        """The name recorded in ``idata.attrs["filter"]``: ``name``, else the name of ``fn``."""
        if self.name is not None:
            return self.name
        fn = getattr(self.fn, "func", self.fn)  # a functools.partial has no __name__ of its own
        return str(getattr(fn, "__name__", type(fn).__name__))


#: The filters :class:`SSP` accepts, by name.
FILTERS: dict[str, FilterSpec] = {
    "ekf": FilterSpec(fn=kalman_filter_1d_ekf_st, link="exp", nonlinear=True),
    "linear": FilterSpec(fn=kalman_filter_1d_st, link="identity", nonlinear=False),
}

#: The posterior dim of a compressed ``(2,)`` ``sigma_q_raw``: the first state's drift, then the shared one.
SIGMA_Q_GROUP_DIM = "sigma_q_group"

#: The variables each ``sigma_q`` family reads from the prior, optional ones included.
_SIGMA_Q_VARS: dict[str, tuple[str, ...]] = {
    "beta": ("sigma_q_alpha_prior", "sigma_q_beta_prior", "sigma_q_scale_prior"),
    "truncated_normal": ("sigma_q_loc_prior", "sigma_q_scale_prior", "sigma_q_low_prior", "sigma_q_high_prior"),
}


class SSP(Forecaster):
    """Multi-series state-space model: a shared latent state, one Kalman filter, fitted by NUTS, SVI or MAP.

    The model, for ``n_series`` series sharing ``n_states`` states::

        a_t     = a_{t-1} + eta_t                    eta_t ~ N(0, diag(sigma_q**2))
        y_{t,j} = Z_{t,j} . h(a_t) + eps_{t,j}       eps_{t,j} ~ N(0, sigma_h[j]**2)
        sigma_h = sigma_h_raw * sdy                  sigma_h_raw[j] ~ sigma_h_prior
        sigma_q ~ the prior's sigma_q block

    ``h`` is ``exp(exponent * a)`` on the positivity states under ``filter="ekf"`` and the identity under
    ``filter="linear"``, where the filter floors the positivity states instead. The prior's disclosed states
    ``a_obs`` / ``P_obs`` (``P_obs = inf`` where nothing is disclosed) are fused into the state at each step.

    Parameters
    ----------
    prior : SspPrior | xr.Dataset
        Natural-scale SSP prior: ``a0``, ``P0`` (diagonal ``(state,)`` or full ``(state, state_dual)``),
        ``positivity``, the ``sigma_q`` block of its ``attrs["sigma_q_family"]`` (see
        :func:`~bunobee.models.ssp.transforms.validate_prior`), and optionally ``a_obs`` / ``P_obs`` over
        ``(time, state)`` and ``sdy`` over ``(series,)``. The ``sigma_q`` block is per state, ``(n_states,)``, or
        compressed, ``(2,)``: the first state's drift and one drift the other states share. Under ``filter="ekf"``
        it is moved to a-space with :func:`~bunobee.models.ssp.transforms.transform_to_ekf_st`, so a positivity
        state's ``a0`` must be positive.
    filter : {"ekf", "linear"} | FilterSpec
        The filter, by default ``"ekf"``: a name in :data:`FILTERS` (see the module docstring), or a
        :class:`FilterSpec` of your own. The spec is resolved once, here, and kept on ``filter_spec``.
    exponent : float
        ``k`` in the EKF's ``exp(k * a)``, by default 0.5. Unused under ``filter="linear"``.
    match : {"mean", "median", "linearize"}
        Moment-matching mode of the a-space transform, by default ``"mean"``. Unused under ``filter="linear"``.
    sigma_h_prior : numpyro.distributions.Distribution | None
        Prior of each series' ``sigma_h / sdy``, by default ``Beta(2, 10)``. ``sdy`` is the prior's ``sdy`` when
        it has one, else the standard deviation of each series of ``y``, ignoring NaN.
    inference : NUTS | SVI | MAP | None
        How :meth:`fit` fits the model, by default ``NUTS()``; see :mod:`bunobee.models.inference`. The filter
        runs over all the series jointly, so ``SVI`` and ``MAP`` must leave ``subsample=None``.

    Attributes
    ----------
    filter_spec : FilterSpec
        The filter the model fits with and how it is forecast.
    filter_prior : xr.Dataset
        The prior the filter runs on: ``prior`` itself under ``filter="linear"``, its a-space transform under
        ``filter="ekf"``.

    Examples
    --------
    >>> model = SSP(prior, filter="ekf", exponent=1.0)  # doctest: +SKIP
    >>> model.fit(y, Z, mcmc_args={"num_warmup": 200, "num_samples": 200})  # doctest: +SKIP
    >>> forecast = model.predict(Z_future, noise_embed=True)  # doctest: +SKIP
    >>> SSP(prior, inference=SVI(steps=3000)).fit(y, Z).predict(Z_future)  # doctest: +SKIP
    >>> plot_states(model.idata_, dates, state_labels, states_key="at_nat")  # doctest: +SKIP
    """

    #: The filter needs every series at every step, so the series plate is not subsampled.
    inference_methods = (NUTS, SVI, MAP)

    def __init__(
        self,
        prior: SspPrior | xr.Dataset,
        *,
        filter: str | FilterSpec = "ekf",
        exponent: float = 0.5,
        match: str = "mean",
        sigma_h_prior: dist.Distribution | None = None,
        inference: Inference | None = None,
    ) -> None:
        super().__init__(inference)
        self.filter_spec = _resolve_filter(filter)
        prior_ds = _unwrap(prior)
        validate_prior(prior_ds, require_init=True, require_sigma_q=True)
        prior_ds = _with_full_P0(prior_ds)

        self.prior = prior_ds
        self.filter = filter
        self.exponent = float(exponent)
        self.match = match
        self.sigma_h_prior = dist.Beta(2.0, 10.0) if sigma_h_prior is None else sigma_h_prior
        self.sigma_q_family = str(prior_ds.attrs.get("sigma_q_family", "truncated_normal"))
        nonlinear = self.filter_spec.nonlinear
        self.filter_prior = transform_to_ekf_st(prior_ds, self.exponent, match) if nonlinear else prior_ds
        self._sigma_q_prior_dim = _check_sigma_q_block(self.filter_prior, self.sigma_q_family)
        self._single_series = False

    @property
    def n_states(self) -> int:
        """Number of latent states."""
        return int(self.prior.sizes["state"])

    def model(
        self,
        y: jnp.ndarray,
        Z: jnp.ndarray,
        a0: jnp.ndarray,
        P0: jnp.ndarray,
        positivity: jnp.ndarray,
        sdy: jnp.ndarray,
        sigma_q_params: Mapping[str, jnp.ndarray],
        a_obs: jnp.ndarray | None = None,
        P_obs: jnp.ndarray | None = None,
        mask: jnp.ndarray | None = None,
    ) -> None:
        """The NumPyro model. Sampled sites: ``sigma_h_raw`` and ``sigma_q_raw``; the rest is deterministic.

        Deterministic sites: ``sigma_h`` ``(series,)``, ``sigma_q`` ``(state,)``, ``at`` ``(time, state)``, ``mu``
        ``(time, series)`` and, for a nonlinear filter (``filter="ekf"``), ``at_nat``, the states on the natural scale.
        """
        n_states = a0.shape[0]
        with numpyro.plate("series", y.shape[1]):
            sigma_h_raw = numpyro.sample("sigma_h_raw", self.sigma_h_prior)
        sigma_h = numpyro.deterministic("sigma_h", sigma_h_raw * sdy)
        sigma_q = numpyro.deterministic("sigma_q", _expand_sigma_q(self._sample_sigma_q(sigma_q_params), n_states))

        spec = self.filter_spec
        kwargs: dict[str, Any] = {"exponent": self.exponent} if spec.nonlinear else {}
        lp, at, *_ = spec.fn(
            a0=a0,
            P0=P0,
            Z=Z,
            sigma_h=sigma_h,
            sigma_q=sigma_q,
            y=y,
            logp=True,
            positivity=positivity,
            a_obs=a_obs,
            P_obs=P_obs,
            mask=mask,
            **kwargs,
        )
        numpyro.factor("lp", lp)
        numpyro.deterministic("at", at)
        if spec.nonlinear:
            # The same clip the EKF applies inside the filter and forecast_ssp applies to the forecast.
            lam = jnp.exp(jnp.clip(self.exponent * at, -EXP_CLIP, EXP_CLIP))
            at = numpyro.deterministic("at_nat", jnp.where(positivity[None, :], lam, at))
        numpyro.deterministic("mu", jnp.einsum("tjs,ts->tj", Z, at))

    def _sample_sigma_q(self, params: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        if self.sigma_q_family == "beta":
            raw = numpyro.sample("sigma_q_raw", dist.Beta(params["sigma_q_alpha_prior"], params["sigma_q_beta_prior"]))
            return raw * params["sigma_q_scale_prior"]
        low = params.get("sigma_q_low_prior")
        return numpyro.sample(
            "sigma_q_raw",
            dist.TruncatedNormal(
                params["sigma_q_loc_prior"],
                params["sigma_q_scale_prior"],
                low=0.0 if low is None else low,
                high=params.get("sigma_q_high_prior"),
            ),
        )

    def fit(
        self,
        y: np.ndarray,
        Z: np.ndarray,
        *,
        mcmc_args: Mapping[str, Any] | None = None,
        seed: int = 0,
    ) -> SSP:
        """Fit the model to ``y`` with its inference method and store the posterior on ``idata_``.

        Parameters
        ----------
        y : np.ndarray
            Observations, ``(n_steps, n_series)``, or ``(n_steps,)`` for one series. NaN marks a missing entry: the
            filter makes it a predict-only step and leaves it out of the likelihood.
        Z : np.ndarray
            Design, ``(n_steps, n_series, n_states)``, or ``(n_steps, n_states)`` with a 1-D ``y``. It must be
            finite wherever ``y`` is observed.
        mcmc_args : Mapping | None
            Keyword arguments for ``numpyro.infer.MCMC`` for this fit, on top of the model's ``NUTS`` settings
            (:data:`~bunobee.models.base.DEFAULT_MCMC_ARGS` by default). Only for a model fitted by NUTS.
        seed : int
            Seed of the fit's ``jax.random.PRNGKey``, by default 0.

        Returns
        -------
        SSP
            ``self``, fitted. ``idata_`` holds every model site with dims ``(chain, draw, ...)`` (one chain under
            SVI or MAP), the ``positivity`` mask, the ``exponent``, ``link`` and ``filter`` attrs ``forecast_ssp``
            reads, and the ``inference`` attr.
        """
        y, Z, single = _as_panel(y, Z, self.n_states)
        n_steps, n_series = y.shape
        if "a_obs" in self.filter_prior and self.filter_prior.sizes["time"] != n_steps:
            raise ValueError(
                f"the prior's a_obs / P_obs cover {self.filter_prior.sizes['time']} steps but y has {n_steps}"
            )
        observed = ~np.isnan(y)
        if not np.isfinite(Z[observed]).all():
            raise ValueError("Z must be finite wherever y is observed")

        fp = self.filter_prior
        inputs: dict[str, Any] = {
            "y": jnp.asarray(y),
            "Z": jnp.asarray(Z),
            "a0": jnp.asarray(fp["a0"].values),
            "P0": jnp.asarray(fp["P0"].values),
            "positivity": jnp.asarray(fp["positivity"].values, dtype=bool),
            "sdy": jnp.asarray(self._sdy(y)),
            "sigma_q_params": {
                name: jnp.asarray(fp[name].values) for name in _SIGMA_Q_VARS[self.sigma_q_family] if name in fp
            },
            "a_obs": jnp.asarray(fp["a_obs"].values) if "a_obs" in fp else None,
            "P_obs": jnp.asarray(fp["P_obs"].values) if "P_obs" in fp else None,
            # None keeps the filter bitwise identical to the unmasked one when nothing is missing.
            "mask": None if observed.all() else jnp.asarray(observed),
        }
        self._fit(inputs, mcmc_args=mcmc_args, seed=seed, context={"n_steps": n_steps, "n_series": n_series})
        self._single_series = single
        return self

    def _sdy(self, y: np.ndarray) -> np.ndarray:
        """The per-series scale of ``sigma_h``: the prior's ``sdy``, else the NaN-ignoring std of ``y``."""
        n_series = y.shape[1]
        if "sdy" in self.prior:
            sdy = np.asarray(self.prior["sdy"].values, dtype=float).reshape(-1)
            if sdy.shape != (n_series,):
                raise ValueError(f"the prior's sdy has shape {sdy.shape} but y has {n_series} series")
        else:
            sdy = np.nanstd(y, axis=0)
        if not (np.isfinite(sdy).all() and (sdy > 0).all()):
            raise ValueError(f"sdy must be finite and positive for every series, got {sdy}; is a series constant?")
        return sdy

    def _to_idata(self, samples, inputs, context) -> xr.Dataset:
        dims = {
            "sigma_h_raw": ["series"],
            "sigma_h": ["series"],
            "sigma_q": ["state"],
            # A compressed block gets its own dim; the prior's name for it may be "sigma_q", a variable here.
            "sigma_q_raw": ["state"] if self._sigma_q_prior_dim is None else [SIGMA_Q_GROUP_DIM],
            "at": ["time", "state"],
            "at_nat": ["time", "state"],
            "mu": ["time", "series"],
        }
        coords = {}
        for dim, size in (("time", context["n_steps"]), ("state", self.n_states), ("series", context["n_series"])):
            if dim in self.prior.coords and self.prior.sizes.get(dim) == size:
                coords[dim] = self.prior.coords[dim].values
        if self._sigma_q_prior_dim is not None and self._sigma_q_prior_dim in self.prior.coords:
            coords[SIGMA_Q_GROUP_DIM] = self.prior.coords[self._sigma_q_prior_dim].values
        idata = posterior_to_xarray(samples, dims=dims, coords=coords)
        positivity = np.asarray(self.filter_prior["positivity"].values, dtype=bool)
        return idata.assign(positivity=(("state",), positivity)).assign_attrs(
            exponent=self.exponent, link=self.filter_spec.link, filter=self.filter_spec.label
        )

    def predict(
        self,
        Z_future: np.ndarray,
        *,
        method: str = "replay",
        noise_embed: bool = False,
        transform_callback: Callable | None = None,
        seed: int = 0,
    ) -> xr.Dataset:
        """Forecast the fitted posterior with the filter spec's ``forecast``, else ``forecast_ssp``.

        With ``filter_spec.forecast`` set, ``predict`` returns what that callable returns for
        ``(idata_, Z_future, method=..., noise_embed=..., transform_callback=..., seed=...)``; see
        :class:`SspForecast`. Otherwise it is :func:`~bunobee.models.ssp.forecast.forecast_ssp` with the spec's
        ``link``, a random-walk state with Gaussian observation noise.

        Parameters
        ----------
        Z_future : np.ndarray
            Future design, ``(horizon, n_series, n_states)``, or ``(horizon, n_states)`` after a one-series fit.
            :func:`~bunobee.models.ssp.forecast.build_forecast_design` continues a training design.
        method : {"replay", "filter"}
            How the future state path is propagated, by default ``"replay"``.
        noise_embed : bool
            Add the observation noise to ``forecast_samples``, by default ``False``.
        transform_callback : Callable | None
            Applied to ``forecast_samples`` only, e.g. to undo a normalisation.
        seed : int
            Seed of the forecast's ``numpy`` generator, by default 0.

        Returns
        -------
        xr.Dataset
            ``forecast_samples`` and ``mu_samples`` (plus ``eps_samples`` with ``noise_embed``), dims
            ``(sample, time, series)``.
        """
        Z_future = np.asarray(Z_future, dtype=float)
        if self._single_series and Z_future.ndim == 2:
            Z_future = Z_future[:, None, :]
        spec = self.filter_spec
        if spec.forecast is not None:
            return spec.forecast(
                self.idata_,
                Z_future,
                method=method,
                noise_embed=noise_embed,
                transform_callback=transform_callback,
                seed=seed,
            )
        return forecast_ssp(
            self.idata_,
            Z_future,
            method=method,
            link=spec.link,
            exponent=self.exponent,
            noise_embed=noise_embed,
            transform_callback=transform_callback,
            seed=seed,
        )


def _resolve_filter(filter: str | FilterSpec) -> FilterSpec:
    """Return the :class:`FilterSpec` of ``filter``, a name in :data:`FILTERS` or a spec, after checking it.

    Raises
    ------
    ValueError
        If the name is not in :data:`FILTERS`, ``fn`` or a given ``forecast`` is not callable, or ``link`` is not in
        ``FORECAST_LINKS`` while ``forecast`` is left out.
    TypeError
        If ``filter`` is neither a name nor a :class:`FilterSpec`.
    """
    if isinstance(filter, str):
        if filter not in FILTERS:
            raise ValueError(f"filter must be one of {sorted(FILTERS)} or a FilterSpec, got {filter!r}")
        spec = FILTERS[filter]
    elif isinstance(filter, FilterSpec):
        spec = filter
    else:
        raise TypeError(f"filter must be a name in FILTERS or a FilterSpec, got {type(filter).__name__}")
    if not callable(spec.fn):
        raise ValueError(f"FilterSpec.fn must be callable, got {type(spec.fn).__name__}")
    if spec.forecast is None:
        if spec.link not in FORECAST_LINKS:
            raise ValueError(
                f"FilterSpec.link must be one of {FORECAST_LINKS} when no forecast is given, got {spec.link!r}; "
                "pass FilterSpec.forecast to forecast a filter forecast_ssp does not cover"
            )
    elif not callable(spec.forecast):
        raise ValueError(f"FilterSpec.forecast must be callable, got {type(spec.forecast).__name__}")
    return spec


def _with_full_P0(prior: xr.Dataset) -> xr.Dataset:
    """Return ``prior`` with a full ``(state, state_dual)`` ``P0``; a diagonal ``(state,)`` one is expanded."""
    P0 = np.asarray(prior["P0"].values, dtype=float)
    if P0.ndim == 2:
        return prior
    state = prior.coords["state"].values if "state" in prior.coords else np.arange(P0.shape[0])
    return prior.assign(P0=(("state", "state_dual"), np.diag(P0))).assign_coords(state_dual=state)


def _check_sigma_q_block(prior: xr.Dataset, family: str) -> str | None:
    """Check the ``sigma_q`` block is per state or compressed ``(2,)``.

    Returns ``None`` for a per-state block, else the prior's dim of the compressed block.
    """
    n_states = int(prior.sizes["state"])
    present = [name for name in _SIGMA_Q_VARS[family] if name in prior]
    shapes = {name: prior[name].shape for name in present}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"the sigma_q prior variables must share one shape, got {shapes}")
    shape = next(iter(shapes.values()))
    if shape not in ((n_states,), (2,)):
        raise ValueError(f"the sigma_q prior must have shape ({n_states},) or compressed (2,), got {shape}")
    if shape == (n_states,):
        return None
    return str(prior[present[0]].dims[0])


def _expand_sigma_q(sigma_q: jnp.ndarray, n_states: int) -> jnp.ndarray:
    """Expand a compressed ``(2,)`` ``sigma_q`` to ``(n_states,)``: the first state's, then the shared one."""
    if sigma_q.shape[-1] == n_states:
        return sigma_q
    return jnp.concatenate([sigma_q[..., :1], jnp.broadcast_to(sigma_q[..., 1:2], (*sigma_q.shape[:-1], n_states - 1))])


def _as_panel(y: np.ndarray, Z: np.ndarray, n_states: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return ``(y, Z, single)`` as ``(n_steps, n_series)`` and ``(n_steps, n_series, n_states)`` arrays."""
    y = np.asarray(y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    single = y.ndim == 1
    if single:
        y = y[:, None]
        if Z.ndim == 2:
            Z = Z[:, None, :]
    if y.ndim != 2:
        raise ValueError(f"y must be (n_steps,) or (n_steps, n_series), got shape {y.shape}")
    if Z.ndim != 3 or Z.shape[:2] != y.shape:
        raise ValueError(f"Z must be (n_steps, n_series, n_states) matching y {y.shape}, got shape {Z.shape}")
    if Z.shape[2] != n_states:
        raise ValueError(f"Z has {Z.shape[2]} states but the prior has {n_states}")
    return y, Z, single
