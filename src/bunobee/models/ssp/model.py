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
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro
import xarray as xr
from numpyro import distributions as dist

from bunobee.models.base import Forecaster
from bunobee.models.ssp.forecast import EXP_CLIP, forecast_ssp
from bunobee.models.ssp.kalman_1d_st import kalman_filter_1d_st
from bunobee.models.ssp.kalman_1d_st_ekf import kalman_filter_1d_ekf_st
from bunobee.models.ssp.posterior import posterior_to_xarray
from bunobee.models.ssp.prior import SspPrior
from bunobee.models.ssp.transforms import _unwrap, transform_to_ekf_st, validate_prior


@dataclass(frozen=True)
class FilterSpec:
    """One filter :class:`SSP` can fit with, and how its positivity states are read."""

    #: The multi-series filter, called with ``logp=True`` as the likelihood.
    fn: Callable
    #: The ``forecast_ssp`` link of its posterior: ``"exp"`` or ``"identity"``.
    link: str
    #: Whether positivity states are log-intensities mapped by ``exp(exponent * a)``.
    nonlinear: bool


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
    """Multi-series state-space model: a shared latent state, one Kalman filter, NUTS.

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
    filter : {"ekf", "linear"}
        The filter, by default ``"ekf"``; see the module docstring.
    exponent : float
        ``k`` in the EKF's ``exp(k * a)``, by default 0.5. Unused under ``filter="linear"``.
    match : {"mean", "median", "linearize"}
        Moment-matching mode of the a-space transform, by default ``"mean"``. Unused under ``filter="linear"``.
    sigma_h_prior : numpyro.distributions.Distribution | None
        Prior of each series' ``sigma_h / sdy``, by default ``Beta(2, 10)``. ``sdy`` is the prior's ``sdy`` when
        it has one, else the standard deviation of each series of ``y``, ignoring NaN.

    Attributes
    ----------
    filter_prior : xr.Dataset
        The prior the filter runs on: ``prior`` itself under ``filter="linear"``, its a-space transform under
        ``filter="ekf"``.

    Examples
    --------
    >>> model = SSP(prior, filter="ekf", exponent=1.0)  # doctest: +SKIP
    >>> model.fit(y, Z, mcmc_args={"num_warmup": 200, "num_samples": 200})  # doctest: +SKIP
    >>> forecast = model.predict(Z_future, noise_embed=True)  # doctest: +SKIP
    >>> plot_states(model.idata_, dates, state_labels, states_key="at_nat")  # doctest: +SKIP
    """

    def __init__(
        self,
        prior: SspPrior | xr.Dataset,
        *,
        filter: str = "ekf",
        exponent: float = 0.5,
        match: str = "mean",
        sigma_h_prior: dist.Distribution | None = None,
    ) -> None:
        super().__init__()
        if filter not in FILTERS:
            raise ValueError(f"filter must be one of {sorted(FILTERS)}, got {filter!r}")
        prior_ds = _unwrap(prior)
        validate_prior(prior_ds, require_init=True, require_sigma_q=True)
        prior_ds = _with_full_P0(prior_ds)

        self.prior = prior_ds
        self.filter = filter
        self.exponent = float(exponent)
        self.match = match
        self.sigma_h_prior = dist.Beta(2.0, 10.0) if sigma_h_prior is None else sigma_h_prior
        self.sigma_q_family = str(prior_ds.attrs.get("sigma_q_family", "truncated_normal"))
        self.filter_prior = transform_to_ekf_st(prior_ds, self.exponent, match) if self._spec.nonlinear else prior_ds
        self._sigma_q_prior_dim = _check_sigma_q_block(self.filter_prior, self.sigma_q_family)
        self._single_series = False

    @property
    def _spec(self) -> FilterSpec:
        return FILTERS[self.filter]

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
        ``(time, series)`` and, under ``filter="ekf"``, ``at_nat``, the states on the natural scale.
        """
        n_states = a0.shape[0]
        with numpyro.plate("series", y.shape[1]):
            sigma_h_raw = numpyro.sample("sigma_h_raw", self.sigma_h_prior)
        sigma_h = numpyro.deterministic("sigma_h", sigma_h_raw * sdy)
        sigma_q = numpyro.deterministic("sigma_q", _expand_sigma_q(self._sample_sigma_q(sigma_q_params), n_states))

        spec = self._spec
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
        """Fit the model to ``y`` by NUTS and store the posterior on ``idata_``.

        Parameters
        ----------
        y : np.ndarray
            Observations, ``(n_steps, n_series)``, or ``(n_steps,)`` for one series. NaN marks a missing entry: the
            filter makes it a predict-only step and leaves it out of the likelihood.
        Z : np.ndarray
            Design, ``(n_steps, n_series, n_states)``, or ``(n_steps, n_states)`` with a 1-D ``y``. It must be
            finite wherever ``y`` is observed.
        mcmc_args : Mapping | None
            Keyword arguments for ``numpyro.infer.MCMC``, on top of
            :data:`~bunobee.models.base.DEFAULT_MCMC_ARGS`.
        seed : int
            Seed of the sampler's ``jax.random.PRNGKey``, by default 0.

        Returns
        -------
        SSP
            ``self``, fitted. ``idata_`` holds every model site with dims ``(chain, draw, ...)``, the
            ``positivity`` mask, and the ``exponent``, ``link`` and ``filter`` attrs ``forecast_ssp`` reads.
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
            exponent=self.exponent, link=self._spec.link, filter=self._spec.fn.__name__
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
        """Forecast with :func:`~bunobee.models.ssp.forecast.forecast_ssp` from the fitted posterior.

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
        return forecast_ssp(
            self.idata_,
            Z_future,
            method=method,
            link=self._spec.link,
            exponent=self.exponent,
            noise_embed=noise_embed,
            transform_callback=transform_callback,
            seed=seed,
        )


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
