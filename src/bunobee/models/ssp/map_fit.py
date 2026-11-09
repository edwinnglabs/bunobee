"""MAP fits of the linear state-space model, one series at a time or many in parallel.

:func:`fit_map` fits one series and :func:`fit_map_batch` fits ``B`` independent series with ``jax.vmap``. Both
maximise the Kalman filter's log likelihood plus the log priors of the noise and the drifts with Adam, much faster
than NUTS and without posterior uncertainty, and return the parameters of the lowest loss over their steps. The
design matrix ``Z`` of shape ``(n_steps, n_states)`` is shared by all series: a local level, a seasonal pattern, a
trend or regression columns are whatever the caller puts in it. The first state drifts on its own; the other states
share one drift, and ``fixed_states`` gives the states it names no drift, for coefficients that do not move.

The fitters take the target transform as an argument (see :mod:`bunobee.target_transforms`), by default
:class:`~bunobee.target_transforms.LogTransform`, and the predictors invert the transform a fit used. They take
the priors of the noise and the drifts as arguments too (see :class:`SigmaPrior`), by default the uniform priors
:data:`SIGMA_H_PRIOR` and :data:`SIGMA_Q_PRIOR`. Their filtered covariances are diagonal, so the forecast variance
the predictors use is approximate.

The ``predict_*`` helpers here are deprecated in favor of :func:`bunobee.models.ssp.forecast.forecast_ssp`; they are
retained because their closed-form lognormal-mean correction has no posterior-replay equivalent. See each
function's deprecation note.

This module is the only part of ``bunobee`` that needs ``optax`` and ``tqdm``. Those live behind the optional
``optim`` extra, so importing this module requires ``pip install 'bunobee[optim]'``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from jax import jit, value_and_grad
import numpy as np
import jax.numpy as jnp
from numpyro import distributions as dist

try:
    import optax
    from tqdm.auto import tqdm
except ModuleNotFoundError as exc:  # pragma: no cover - only hit without the optim extra
    raise ModuleNotFoundError(
        f"bunobee.models.ssp.map_fit requires the optional 'optim' extra (missing: {exc.name}). "
        "Install it with: pip install 'bunobee[optim]'"
    ) from exc

from .kalman_1d import kalman_filter_1d
from .vfactory import make_kalman_batch
from bunobee.target_transforms import LogTransform, Transform

logger = logging.getLogger(__name__)

#: Batched Kalman filter over independent series, built from the single-series filter.
kalman_filter_1d_batch = make_kalman_batch(kalman_filter_1d)

#: Bounds of the default uniform priors of the MAP fitters: ``sigma_h``, the observation noise, and ``sigma_q``, the
#: drifts of the states. They suit the default log target, which divides each series by its mean, so the noise is on
#: the scale of the log of a unit-mean series whatever the series' own units.
SIGMA_H_BOUNDS = (0.1, 0.5)
SIGMA_Q_BOUNDS = (0.01, 0.1)


@dataclass(frozen=True)
class SigmaPrior:
    """Prior of a sigma of the MAP fitters: a positive distribution family and its parameters.

    A uniform prior bounds the sigma: the fit never returns one outside ``params``. The half-normal and
    log-normal priors are soft: they pull the sigma toward small values or toward ``exp(loc)`` but allow any
    positive value. The fit starts each sigma at its prior's mean, so the default uniform priors start
    ``sigma_h`` at 0.3 and the drifts at 0.055.

    Parameters
    ----------
    family : str
        ``"uniform"``, ``"half_normal"`` or ``"log_normal"``.
    params : tuple of float
        The family's parameters, as ``numpyro.distributions`` takes them: ``(low, high)`` for ``"uniform"``,
        ``(scale,)`` for ``"half_normal"`` and ``(loc, scale)`` of the sigma's log for ``"log_normal"``.
    """

    family: str
    params: tuple[float, ...]

    #: Each family's distribution, and its mean as a function of its parameters: the sigma's starting point.
    FAMILIES: ClassVar[dict[str, tuple[type[dist.Distribution], Callable[..., float]]]] = {
        "uniform": (dist.Uniform, lambda low, high: (low + high) / 2),
        "half_normal": (dist.HalfNormal, lambda scale: scale * math.sqrt(2 / math.pi)),
        "log_normal": (dist.LogNormal, lambda loc, scale: math.exp(loc + scale**2 / 2)),
    }

    def __post_init__(self) -> None:
        if self.family not in self.FAMILIES:
            raise ValueError(f"family must be one of {sorted(self.FAMILIES)}, got {self.family!r}")

    @property
    def distribution(self) -> dist.Distribution:
        """The prior as a ``numpyro`` distribution."""
        return self.FAMILIES[self.family][0](*self.params)

    @property
    def init(self) -> float:
        """The prior's mean, where the fit starts the sigma."""
        return float(self.FAMILIES[self.family][1](*self.params))

    @property
    def bounds(self) -> tuple[float, float] | None:
        """The ``(low, high)`` bounds of a uniform prior; None for a soft prior, which allows any positive sigma."""
        return (float(self.params[0]), float(self.params[1])) if self.family == "uniform" else None


#: Default priors of the MAP fitters: ``sigma_h`` ~ U(0.1, 0.5) and each drift ~ U(0.01, 0.1).
SIGMA_H_PRIOR = SigmaPrior("uniform", SIGMA_H_BOUNDS)
SIGMA_Q_PRIOR = SigmaPrior("uniform", SIGMA_Q_BOUNDS)


def _softplus(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log1p(jnp.exp(x))


def _softplus_inv(v: float) -> float:
    """Softplus inverse: maps a positive value to the unconstrained initialisation point."""
    return float(np.log(np.expm1(float(v)) + 1e-12))


def _drift_mask(n_states: int, fixed_states: np.ndarray | None) -> np.ndarray:
    """Which states of a design drift, from the ``fixed_states`` argument of the MAP fitters.

    Parameters
    ----------
    n_states : int
        Number of states of the design.
    fixed_states : np.ndarray | None
        Boolean mask of shape ``(n_states,)``, True where a state's drift is 0. ``None`` fixes no state.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(n_states,)``, True where a state drifts.

    Raises
    ------
    ValueError
        If ``fixed_states`` does not have shape ``(n_states,)``.
    """
    if fixed_states is None:
        return np.ones(n_states, dtype=bool)
    fixed = np.asarray(fixed_states, dtype=bool)
    if fixed.shape != (n_states,):
        raise ValueError(f"fixed_states must have shape ({n_states},), got {fixed.shape}")
    return ~fixed


def _state_drifts(sigma_q_first: jnp.ndarray, sigma_q_rest: jnp.ndarray, drifts: np.ndarray) -> jnp.ndarray:
    """Per-state drifts: the first state's, then the one the other states share, and 0 on the fixed states.

    Parameters
    ----------
    sigma_q_first, sigma_q_rest : jnp.ndarray
        The two drifts, of shape ``()`` or ``(B,)``.
    drifts : np.ndarray
        Boolean mask of the states that drift (:func:`_drift_mask`), of shape ``(n_states,)``.

    Returns
    -------
    jnp.ndarray
        The drifts of shape ``(n_states,)`` or ``(B, n_states)``.
    """
    sigma_q = jnp.concatenate(
        [sigma_q_first[..., None], jnp.repeat(sigma_q_rest[..., None], drifts.size - 1, axis=-1)], axis=-1
    )
    return sigma_q * drifts


def fit_map(
    values: np.ndarray,
    n_iter: int = 500,
    lr: float = 3e-2,
    Z: jnp.ndarray | None = None,
    patience: int | None = 50,
    tol: float = 1e-5,
    log_every: int | None = None,
    transform: Transform = LogTransform(),
    sigma_h_prior: SigmaPrior = SIGMA_H_PRIOR,
    sigma_q_prior: SigmaPrior = SIGMA_Q_PRIOR,
    fixed_states: np.ndarray | None = None,
) -> dict:
    """Fit a linear state-space model to one series via MAP (Adam optimiser).

    Gradient-based MAP estimation of the Kalman filter's noise and drifts — much faster
    than NUTS, no posterior uncertainty. The model is fitted to
    ``transform.forward(values)``, with the transform's parameters fitted on ``values``
    first. The returned parameters are those of the lowest loss over the steps run.

    Parameters
    ----------
    values : np.ndarray
        1-D array of the series' non-negative observations (n_steps,).
    n_iter : int
        Maximum number of Adam optimisation steps, by default 500.
    lr : float
        Adam learning rate, by default 3e-2.
    Z : jnp.ndarray | None
        (n_steps, n_states) pre-built design matrix. Build it once and reuse it
        when fitting many series of the same length.
    patience : int | None
        Stop early if the loss does not improve by more than ``tol`` for this
        many consecutive steps. Set to ``None`` to disable early stopping.
    tol : float
        Minimum loss improvement to reset the patience counter, by default 1e-5.
    log_every : int | None
        Log loss at this step interval via DEBUG. ``None`` disables logging.
    transform : Transform
        Target transform, by default :class:`LogTransform`: the model is fitted to
        ``log(max(values, floor) / mean)``, floor 0.1.
    sigma_h_prior : SigmaPrior
        Prior of the observation noise ``sigma_h``, by default :data:`SIGMA_H_PRIOR`,
        U(0.1, 0.5).
    sigma_q_prior : SigmaPrior
        Prior of each of the two drifts: the first state's and the one the other
        states share (with one state, of its drift alone). By default
        :data:`SIGMA_Q_PRIOR`, U(0.01, 0.1). A drift no state uses has no prior term.
    fixed_states : np.ndarray | None
        Boolean mask of shape (n_states,), True for a state whose drift is 0: a fixed
        coefficient, zero-centered by its ``a0 = 0``, ``P0 = 1`` Gaussian prior and
        updated by the filter only. ``None``, the default, fixes no state.

    Returns
    -------
    dict
        Keys: ``at`` (n_steps, n_states), ``sigma_h`` (float), ``sigma_q``
        (n_states,), the transform's parameters (``response_norm`` for the log
        transform), ``transform``, ``Z``, ``a0``, ``P0``, ``losses``
        (list[float], the loss before each step's update), ``final_loss`` (float,
        the lowest of ``losses``, which the returned sigmas achieve), ``n_iter_run``
        (int).
    """
    transform_params = transform.fit(values)
    y = jnp.array(transform.forward(values, transform_params))

    n_states = Z.shape[1]
    drifts = _drift_mask(n_states, fixed_states)

    a0 = jnp.zeros(n_states)
    P0 = jnp.ones(n_states)

    def neg_log_posterior(params: jnp.ndarray) -> jnp.ndarray:
        sigma_h = _softplus(params[0])
        sigma_q_level = _softplus(params[1])
        sigma_q_seas = _softplus(params[2])
        sigma_q = _state_drifts(sigma_q_level, sigma_q_seas, drifts)

        lp, _, _, _, _, _ = kalman_filter_1d(
            a0=a0,
            P0=P0,
            sigma_h=sigma_h,
            sigma_q=sigma_q,
            y=y,
            Z=Z,
            logp=True,
        )

        log_prior = sigma_h_prior.distribution.log_prob(sigma_h)
        # A drift no state uses must not enter the loss: the term would change which step is kept as the best.
        if drifts[0]:
            log_prior += sigma_q_prior.distribution.log_prob(sigma_q_level)
        if drifts[1:].any():
            log_prior += sigma_q_prior.distribution.log_prob(sigma_q_seas)
        return -(lp + log_prior)

    # Each sigma starts at its prior's mean: for the default priors, the midpoints of U(0.1, 0.5) and U(0.01, 0.1).
    params = jnp.array(
        [
            _softplus_inv(sigma_h_prior.init),  # sigma_h
            _softplus_inv(sigma_q_prior.init),  # sigma_q_level
            _softplus_inv(sigma_q_prior.init),  # sigma_q_seas
        ]
    )

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jit
    def _step(params, opt_state):
        loss, grads = value_and_grad(neg_log_posterior)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), new_opt_state, loss

    losses: list[float] = []
    best_params, best_loss = params, float("inf")
    patience_loss = float("inf")
    no_improve = 0

    for i in range(n_iter):
        new_params, opt_state, loss = _step(params, opt_state)
        loss_val = float(loss)
        losses.append(loss_val)
        # The loss is that of params, before this step's update: they, not new_params, achieved it. A sigma
        # pushed past a uniform prior's bound has an infinite loss, so the best parameters stay inside the bounds.
        if loss_val < best_loss:
            best_params, best_loss = params, loss_val
        params = new_params

        if log_every is not None and i % log_every == 0:
            logger.debug("step %d  loss=%.6f", i, loss_val)

        if patience is not None:
            if patience_loss - loss_val > tol:
                patience_loss = loss_val
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                logger.debug("early stop at step %d  loss=%.6f", i, loss_val)
                break

    sigma_h = float(_softplus(best_params[0]))
    sigma_q = _state_drifts(_softplus(best_params[1]), _softplus(best_params[2]), drifts)

    _, at, Pt, _, _, _ = kalman_filter_1d(
        a0=a0,
        P0=P0,
        sigma_h=sigma_h,
        sigma_q=sigma_q,
        y=y,
        Z=Z,
        logp=False,
    )

    return {
        "at": np.array(at),
        "Pt": np.array(Pt),  # (n_steps, n_states) filtered diagonal covariances
        "sigma_h": sigma_h,
        "sigma_q": np.array(sigma_q),
        **transform_params,
        "transform": transform,
        "Z": Z,
        "a0": a0,
        "P0": P0,
        "losses": losses,
        "final_loss": best_loss,
        "n_iter_run": len(losses),
    }


def predict_map(
    fit_result: dict,
    Z_future: np.ndarray | None = None,
) -> np.ndarray:
    """Generate point forecasts from a MAP-fitted single-series model.

    The forecast mean and variance on the transformed scale are mapped back to
    units by ``transform.inverse`` of the fit's transform. The default log transform
    applies the Jensen / lognormal correction (+0.5·Var) so that the back-
    transformed forecast is an unbiased estimate of E[Y] rather than exp(E[log Y]).

    .. deprecated::
        Prefer :func:`bunobee.models.ssp.forecast.forecast_ssp` for new code. This
        function is kept for MAP fits only: they have no posterior draws to
        replay, and its closed-form variance ``Z_future**2 @ P_last + sigma_h**2``
        comes from the terminal filtered covariance rather than from the forward
        process-noise accumulation ``forecast_ssp`` samples, so the two are not
        interchangeable. To migrate, fit with NUTS and call ``forecast_ssp``, then
        take the mean of ``forecast_samples`` for the same E[Y] target.

    Parameters
    ----------
    fit_result : dict
        Output of ``fit_map()``. Without a ``transform`` key the
        forecast is inverted as :class:`LogTransform` inverts it.
    Z_future : np.ndarray | None
        (horizon, n_states) pre-built future design matrix.

    Returns
    -------
    np.ndarray
        Point forecasts of shape (horizon,).
    """
    at = fit_result["at"]
    Pt = fit_result["Pt"]  # (n_steps, n_states) diagonal covariances
    sigma_h = float(fit_result["sigma_h"])
    transform = fit_result.get("transform", LogTransform())

    Z_future = np.asarray(Z_future)  # (horizon, n_states)
    a_last = at[-1]  # (n_states,)
    P_last = Pt[-1]  # (n_states,) diagonal
    mu_future = Z_future @ a_last  # (horizon,)

    # Use P_last (terminal filtered variance) + σ_h² for Jensen correction.
    # The daily random-walk drift h·σ_q² is excluded: the state transitions run
    # every day but the seasonal states are only active once per week, so the
    # daily accumulation inflates week-4 forecasts relative to week-1 with no
    # basis in the observations. P_last already encodes σ_q implicitly via the
    # converged Kalman filter, so the correction remains meaningful but horizon-flat.
    Z_sq = Z_future**2  # (horizon, n_states)
    var_future = Z_sq @ P_last + sigma_h**2

    return transform.inverse(mu_future, var_future, fit_result)


def fit_map_batch(
    values: np.ndarray,
    n_iter: int = 500,
    lr: float = 3e-2,
    Z: jnp.ndarray | None = None,
    chunk_size: int | None = 4096,
    log_every: int | None = None,
    show_progress: bool = False,
    transform: Transform = LogTransform(),
    sigma_h_prior: SigmaPrior = SIGMA_H_PRIOR,
    sigma_q_prior: SigmaPrior = SIGMA_Q_PRIOR,
    fixed_states: np.ndarray | None = None,
) -> dict:
    """Fit a linear state-space model to B series in parallel via MAP (Adam optimiser).

    Uses ``jax.vmap`` to batch the Kalman filter across series and runs Adam
    optimisation for a fixed number of steps, tracking the best parameters
    per series: those of the lowest loss over the steps. Each series is fitted
    to ``transform.forward`` of its values, with its transform parameters fitted
    on its values first.

    Parameters
    ----------
    values : np.ndarray
        2-D array of the series' non-negative observations, shape (B, n_steps).
    n_iter : int
        Number of Adam optimisation steps, by default 500.
    lr : float
        Adam learning rate, by default 3e-2.
    Z : jnp.ndarray | None
        (n_steps, n_states) pre-built design matrix, shared across all series.
    chunk_size : int | None
        Maximum series per ``vmap`` call to limit memory. ``None`` processes
        all series in one call. Default 4096.
    log_every : int | None
        Log mean loss at this step interval via DEBUG. ``None`` disables.
    show_progress : bool
        Display a ``tqdm`` progress bar over optimisation steps, by default False.
    transform : Transform
        Target transform, by default :class:`LogTransform`: each series is fitted
        to ``log(max(values, floor) / mean)``, floor 0.1.
    sigma_h_prior : SigmaPrior
        Prior of the observation noise ``sigma_h``, by default :data:`SIGMA_H_PRIOR`,
        U(0.1, 0.5).
    sigma_q_prior : SigmaPrior
        Prior of each of the two drifts: the first state's and the one the other
        states share (with one state, of its drift alone). By default
        :data:`SIGMA_Q_PRIOR`, U(0.01, 0.1). A drift no state uses has no prior term.
    fixed_states : np.ndarray | None
        Boolean mask of shape (n_states,), True for a state whose drift is 0: a fixed
        coefficient, zero-centered by its ``a0 = 0``, ``P0 = 1`` Gaussian prior and
        updated by the filter only. ``None``, the default, fixes no state.

    Returns
    -------
    dict
        Keys: ``at`` (B, n_steps, n_states), ``sigma_h`` (B,),
        ``sigma_q`` (B, n_states), the transform's parameters (B,) each
        (``response_norm`` for the log transform), ``transform``, ``Z``,
        ``a0``, ``P0``, ``final_loss`` (B,), the lowest loss over the steps,
        which the returned sigmas achieve.
    """
    B = values.shape[0]

    transform_params = transform.fit(values)  # (B,) each
    y = jnp.array(transform.forward(values, transform_params))  # (B, n_steps)

    n_states = Z.shape[1]
    drifts = _drift_mask(n_states, fixed_states)
    a0 = jnp.zeros(n_states)
    P0 = jnp.ones(n_states)

    # Each sigma starts at its prior's mean: for the default priors, the midpoints of U(0.1, 0.5) and U(0.01, 0.1).
    init_params = jnp.tile(
        jnp.array(
            [
                _softplus_inv(sigma_h_prior.init),  # sigma_h
                _softplus_inv(sigma_q_prior.init),  # sigma_q_level
                _softplus_inv(sigma_q_prior.init),  # sigma_q_seas
            ]
        ),
        (B, 1),
    )  # (B, 3)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(init_params)

    def _loss_fn(params: jnp.ndarray) -> jnp.ndarray:
        """Batched neg-log-posterior. params: (B, 3) -> scalar (sum over series)."""
        sigma_h = _softplus(params[:, 0])  # (B,)
        sigma_q_level = _softplus(params[:, 1])  # (B,)
        sigma_q_seas = _softplus(params[:, 2])  # (B,)
        sigma_q = _state_drifts(sigma_q_level, sigma_q_seas, drifts)  # (B, n_states)

        log_p, _, _, _, _, _ = kalman_filter_1d_batch(
            a0=a0,
            P0=P0,
            Z=Z,
            sigma_h=sigma_h,
            sigma_q=sigma_q,
            y=y,
            logp=True,
            chunk_size=chunk_size,
        )

        log_prior = sigma_h_prior.distribution.log_prob(sigma_h)  # (B,)
        # A drift no state uses must not enter the loss: the term would change which step is kept as the best.
        if drifts[0]:
            log_prior += sigma_q_prior.distribution.log_prob(sigma_q_level)
        if drifts[1:].any():
            log_prior += sigma_q_prior.distribution.log_prob(sigma_q_seas)

        per_series = -(log_p + log_prior)  # (B,)
        return jnp.sum(per_series), per_series

    @jit
    def _step(params, opt_state, best_params, best_loss):
        (_, per_series_loss), grads = value_and_grad(_loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        # The loss is that of params, before this step's update: they, not new_params, achieved it. A sigma pushed
        # past a uniform prior's bound has an infinite loss, so the best parameters stay inside the bounds.
        improved = per_series_loss < best_loss
        new_best_loss = jnp.where(improved, per_series_loss, best_loss)
        new_best_params = jnp.where(improved[:, None], params, best_params)
        return new_params, new_opt_state, new_best_params, new_best_loss, per_series_loss

    best_params = init_params
    best_loss = jnp.full(B, float("inf"))
    params = init_params

    steps = tqdm(range(n_iter), desc=f"fitting {B} series", unit="step") if show_progress else range(n_iter)
    for i in steps:
        params, opt_state, best_params, best_loss, per_series_loss = _step(
            params,
            opt_state,
            best_params,
            best_loss,
        )
        mean_loss = float(jnp.mean(per_series_loss))
        if show_progress:
            steps.set_postfix(mean_loss=f"{mean_loss:.4f}")  # type: ignore[union-attr]
        if log_every is not None and i % log_every == 0:
            logger.debug("step %d  mean_loss=%.6f", i, mean_loss)

    # Recover MAP estimates from best params
    sigma_h = jnp.asarray(_softplus(best_params[:, 0]))
    sigma_q = _state_drifts(_softplus(best_params[:, 1]), _softplus(best_params[:, 2]), drifts)

    # Final forward pass to get filtered states and diagonal covariances
    _, at, Pt, _, _, _ = kalman_filter_1d_batch(
        a0=a0,
        P0=P0,
        Z=Z,
        sigma_h=sigma_h,
        sigma_q=sigma_q,
        y=y,
        logp=False,
        chunk_size=chunk_size,
    )

    return {
        "at": np.array(at),  # (B, n_steps, n_states)
        "Pt": np.array(Pt),  # (B, n_steps, n_states) filtered diagonal covariances
        "sigma_h": np.array(sigma_h),  # (B,)
        "sigma_q": np.array(sigma_q),  # (B, n_states)
        **transform_params,  # (B,) each
        "transform": transform,
        "Z": Z,
        "a0": a0,
        "P0": P0,
        "final_loss": np.array(best_loss),  # (B,)
    }


def predict_map_batch(
    fit_result: dict,
    Z_future: np.ndarray | None = None,
) -> np.ndarray:
    """Generate point forecasts for B series from a batch-fitted model.

    The forecast means and variances on the transformed scale are mapped back to
    units by ``transform.inverse`` of the fit's transform. The default log transform
    applies the Jensen / lognormal correction (+0.5·Var) so that the back-
    transformed forecast is an unbiased estimate of E[Y] rather than exp(E[log Y]).

    .. deprecated::
        Prefer :func:`bunobee.models.ssp.forecast.forecast_ssp` for new code. Same
        caveat as :func:`predict_map`: these are MAP point estimates with
        an analytic Jensen correction built on the terminal filtered covariance
        ``Pt[:, -1, :]``, not posterior draws, so ``forecast_ssp`` cannot reproduce
        them numerically. It is the sampling replacement — build the ``(horizon,
        n_series, n_states)`` design with ``build_forecast_design`` and take the mean
        over ``sample`` for the same E[Y] target.

    Parameters
    ----------
    fit_result : dict
        Output of ``fit_map_batch()``. Without a ``transform`` key the
        forecasts are inverted as :class:`LogTransform` inverts them.
    Z_future : np.ndarray | None
        (horizon, n_states) future design matrix, shared across all series.

    Returns
    -------
    np.ndarray
        Point forecasts of shape (B, horizon).
    """
    mu_future, var_future = forecast_gaussian_map_batch(fit_result, Z_future)
    return fit_result.get("transform", LogTransform()).inverse(mu_future, var_future, fit_result)


def forecast_gaussian_map_batch(fit_result: dict, Z_future: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forecast mean and variance on the transformed scale for B series from a batch-fitted model.

    The Gaussian that :func:`predict_map_batch` maps to point forecasts in units, and that
    ``fit_result["transform"].quantile`` maps to forecast quantiles in units.

    Parameters
    ----------
    fit_result : dict
        Output of ``fit_map_batch()``.
    Z_future : np.ndarray
        (horizon, n_states) future design matrix, shared across all series.

    Returns
    -------
    mu_future, var_future : np.ndarray
        Mean and variance of each series' forecast on the transformed scale, each of shape (B, horizon).
    """
    at = fit_result["at"]  # (B, n_steps, n_states)
    Pt = fit_result["Pt"]  # (B, n_steps, n_states) diagonal covariances
    sigma_h = np.asarray(fit_result["sigma_h"])  # (B,)

    Z_future = np.asarray(Z_future)  # (horizon, n_states)
    a_last = at[:, -1, :]  # (B, n_states)
    P_last = Pt[:, -1, :]  # (B, n_states) diagonal
    mu_future = a_last @ Z_future.T  # (B, horizon)

    # Use P_last (terminal filtered variance) + σ_h² for Jensen correction.
    # The daily random-walk drift h·σ_q² is excluded: the state transitions run
    # every day but the seasonal states are only active once per week, so the
    # daily accumulation inflates week-4 forecasts relative to week-1 with no
    # basis in the observations. P_last already encodes σ_q implicitly via the
    # converged Kalman filter, so the correction remains meaningful but horizon-flat.
    Z_sq = Z_future**2  # (horizon, n_states)
    var_future = P_last @ Z_sq.T + (sigma_h**2)[:, None]  # (B, horizon)
    return mu_future, var_future
