from __future__ import annotations

import logging

import optax
from jax import jit, value_and_grad
import numpy as np
import jax.numpy as jnp
from numpyro import distributions as dist
from tqdm.auto import tqdm

from ..ssp.univariate import kalman_filter_1d, kalman_filter_1d_batch

logger = logging.getLogger(__name__)


def _softplus(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log1p(jnp.exp(x))


def _softplus_inv(v: float) -> float:
    """Softplus inverse: maps a positive value to the unconstrained initialisation point."""
    return float(np.log(np.expm1(float(v)) + 1e-12))


def fit_one_series_opt(
    sales: np.ndarray,
    n_iter: int = 500,
    lr: float = 3e-2,
    Z: jnp.ndarray | None = None,
    patience: int | None = 50,
    tol: float = 1e-5,
    log_every: int | None = None,
) -> dict:
    """Fit a local-level + weekly-seasonality SSP model via MAP (Adam optimiser).

    Uses the same Kalman filter and prior specification as ``fit_one_series`` but
    replaces NUTS with gradient-based MAP estimation — much faster, no posterior
    uncertainty.

    Parameters
    ----------
    sales : np.ndarray
        1-D array of daily unit sales (n_steps,).
    n_iter : int
        Maximum number of Adam optimisation steps, by default 500.
    lr : float
        Adam learning rate, by default 3e-2.
    Z : jnp.ndarray | None
        (n_steps, n_states) pre-built design matrix. Pass ``Z_shared`` to avoid
        redundant dummy builds when fitting many series of the same length.
    patience : int | None
        Stop early if the loss does not improve by more than ``tol`` for this
        many consecutive steps. Set to ``None`` to disable early stopping.
    tol : float
        Minimum loss improvement to reset the patience counter, by default 1e-5.
    log_every : int | None
        Log loss at this step interval via DEBUG. ``None`` disables logging.

    Returns
    -------
    dict
        Keys: ``at`` (n_steps, n_states), ``sigma_h`` (float), ``sigma_q``
        (n_states,), ``response_norm`` (float), ``Z``, ``a0``, ``P0``,
        ``losses`` (list[float]), ``n_iter_run`` (int).
    """
    sales_clipped = np.clip(sales, 1e-1, None).astype(np.float32)
    response_norm = float(sales_clipped.mean())
    y = jnp.array(np.log(sales_clipped / response_norm))

    n_states = Z.shape[1]

    a0 = jnp.zeros(n_states)
    P0 = jnp.ones(n_states)

    def neg_log_posterior(params: jnp.ndarray) -> jnp.ndarray:
        sigma_h = _softplus(params[0])
        sigma_q_level = _softplus(params[1])
        sigma_q_seas = _softplus(params[2])
        sigma_q = jnp.concatenate(
            [
                sigma_q_level[None],
                jnp.repeat(sigma_q_seas[None], n_states - 1),
            ]
        )

        lp, _, _, _, _, _ = kalman_filter_1d(
            a0=a0,
            P0=P0,
            sigma_h=sigma_h,
            sigma_q=sigma_q,
            y=y,
            Z=Z,
            logp=True,
        )

        log_prior = (
            dist.Uniform(0.1, 0.5).log_prob(sigma_h)
            + dist.Uniform(0.01, 0.1).log_prob(sigma_q_level)
            + dist.Uniform(0.01, 0.1).log_prob(sigma_q_seas)
        )
        return -(lp + log_prior)

    params = jnp.array(
        [
            _softplus_inv(0.30),  # sigma_h midpoint of U(0.1, 0.5)
            _softplus_inv(0.055),  # sigma_q_level midpoint of U(0.01, 0.1)
            _softplus_inv(0.055),  # sigma_q_seas midpoint of U(0.01, 0.1)
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
    best_loss = float("inf")
    no_improve = 0

    for i in range(n_iter):
        params, opt_state, loss = _step(params, opt_state)
        loss_val = float(loss)
        losses.append(loss_val)

        if log_every is not None and i % log_every == 0:
            logger.debug("step %d  loss=%.6f", i, loss_val)

        if patience is not None:
            if best_loss - loss_val > tol:
                best_loss = loss_val
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                logger.debug("early stop at step %d  loss=%.6f", i, loss_val)
                break

    sigma_h = float(_softplus(params[0]))
    sigma_q_level = float(_softplus(params[1]))
    sigma_q_seas = float(_softplus(params[2]))
    sigma_q = jnp.concatenate(
        [
            jnp.array([sigma_q_level]),
            jnp.repeat(jnp.array([sigma_q_seas]), n_states - 1),
        ]
    )

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
        "response_norm": response_norm,
        "Z": Z,
        "a0": a0,
        "P0": P0,
        "losses": losses,
        "n_iter_run": len(losses),
    }


def predict_one_series_opt(
    fit_result: dict,
    Z_future: np.ndarray | None = None,
) -> np.ndarray:
    """Generate lognormal-mean forecasts from a MAP-fitted single-series model.

    Applies the Jensen / lognormal correction (+0.5·Var) so that the back-
    transformed forecast is an unbiased estimate of E[Y] rather than exp(E[log Y]).

    Parameters
    ----------
    fit_result : dict
        Output of ``fit_one_series_opt()``.
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
    response_norm = fit_result["response_norm"]

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

    return np.exp(mu_future + 0.5 * var_future) * response_norm


def fit_batch_series_opt(
    sales_matrix: np.ndarray,
    n_iter: int = 500,
    lr: float = 3e-2,
    Z: jnp.ndarray | None = None,
    chunk_size: int | None = 4096,
    log_every: int | None = None,
    show_progress: bool = False,
) -> dict:
    """Fit a local-level + weekly-seasonality SSP model for B series in parallel.

    Uses ``jax.vmap`` to batch the Kalman filter across series and runs Adam
    optimisation for a fixed number of steps, tracking the best parameters
    per series.

    Parameters
    ----------
    sales_matrix : np.ndarray
        2-D array of daily unit sales, shape (B, n_steps).
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

    Returns
    -------
    dict
        Keys: ``at`` (B, n_steps, n_states), ``sigma_h`` (B,),
        ``sigma_q`` (B, n_states), ``response_norm`` (B,), ``Z``,
        ``a0``, ``P0``, ``final_loss`` (B,).
    """
    B = sales_matrix.shape[0]

    sales_clipped = np.clip(sales_matrix, 1e-1, None).astype(np.float32)
    response_norm = sales_clipped.mean(axis=1)  # (B,)
    y = jnp.array(np.log(sales_clipped / response_norm[:, None]))  # (B, n_steps)

    n_states = Z.shape[1]
    a0 = jnp.zeros(n_states)
    P0 = jnp.ones(n_states)

    init_params = jnp.tile(
        jnp.array(
            [
                _softplus_inv(0.30),  # sigma_h midpoint of U(0.1, 0.5)
                _softplus_inv(0.055),  # sigma_q_level midpoint of U(0.01, 0.1)
                _softplus_inv(0.055),  # sigma_q_seas midpoint of U(0.01, 0.1)
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
        sigma_q = jnp.concatenate(
            [
                sigma_q_level[:, None],
                jnp.repeat(sigma_q_seas[:, None], n_states - 1, axis=1),
            ],
            axis=1,
        )  # (B, n_states)

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

        log_prior = (
            dist.Uniform(0.1, 0.5).log_prob(sigma_h)
            + dist.Uniform(0.01, 0.1).log_prob(sigma_q_level)
            + dist.Uniform(0.01, 0.1).log_prob(sigma_q_seas)
        )  # (B,)

        per_series = -(log_p + log_prior)  # (B,)
        return jnp.sum(per_series), per_series

    @jit
    def _step(params, opt_state, best_params, best_loss):
        (_, per_series_loss), grads = value_and_grad(_loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        improved = per_series_loss < best_loss
        new_best_loss = jnp.where(improved, per_series_loss, best_loss)
        new_best_params = jnp.where(improved[:, None], new_params, best_params)
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
    sigma_q_level = _softplus(best_params[:, 1])
    sigma_q_seas = _softplus(best_params[:, 2])
    sigma_q = jnp.concatenate(
        [
            sigma_q_level[:, None],
            jnp.repeat(sigma_q_seas[:, None], n_states - 1, axis=1),
        ],
        axis=1,
    )

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
        "response_norm": response_norm,  # (B,)
        "Z": Z,
        "a0": a0,
        "P0": P0,
        "final_loss": np.array(best_loss),  # (B,)
    }


def predict_batch_series_opt(
    fit_result: dict,
    Z_future: np.ndarray | None = None,
) -> np.ndarray:
    """Generate lognormal-mean forecasts for B series from a batch-fitted model.

    Applies the Jensen / lognormal correction (+0.5·Var) so that the back-
    transformed forecast is an unbiased estimate of E[Y] rather than exp(E[log Y]).

    Parameters
    ----------
    fit_result : dict
        Output of ``fit_batch_series_opt()``.
    Z_future : np.ndarray | None
        (horizon, n_states) future design matrix, shared across all series.

    Returns
    -------
    np.ndarray
        Point forecasts of shape (B, horizon).
    """
    at = fit_result["at"]  # (B, n_steps, n_states)
    Pt = fit_result["Pt"]  # (B, n_steps, n_states) diagonal covariances
    sigma_h = np.asarray(fit_result["sigma_h"])  # (B,)
    response_norm = fit_result["response_norm"]  # (B,)

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

    return np.exp(mu_future + 0.5 * var_future) * response_norm[:, None]
