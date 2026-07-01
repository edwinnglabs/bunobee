from __future__ import annotations

import logging

import jax.numpy as jnp
from jax import lax

logger = logging.getLogger(__name__)


def kalman_filter_1d_st(
    a0: jnp.ndarray,
    P0: jnp.ndarray,
    Z: jnp.ndarray,
    sigma_h: jnp.ndarray | float,
    sigma_q: jnp.ndarray | float,
    y: jnp.ndarray,
    logp: bool = False,
    a_obs: jnp.ndarray | None = None,
    P_obs: jnp.ndarray | None = None,
    positivity_idx: jnp.ndarray | None = None,
) -> tuple[float, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Multi-series Kalman filter with shared latent state and full state covariance.

    Overview
    --------
    Implements a full-covariance Kalman filter for n_series observed series that
    share a common latent state with independent observation noise::

        State evolution:    a_t = a_{t-1} + η_t,                η_t ~ N(0, diag(σ_q²))
        Observation model:  y_{t,j} = Z_{t,j} · a_t + ε_{t,j},  ε_{t,j} ~ N(0, σ_{h,j}²)

    Because all series share a single P_t, the joint innovation covariance
    F_t = Z_t P_t Z_tᵀ + diag(σ_h²) is a full (n_series × n_series) matrix —
    off-diagonal entries are non-zero whenever two series share state variance.
    The earlier diagonal approximation was incorrect for this model structure.

    Timestep update:

    1. **Predict**::

           ŷ_t = Z_t · a_t                           (n_series,)

    2. **Innovations**::

           v_t = y_t − ŷ_t                           (n_series,)
           F_t = Z_t P_t Z_tᵀ + diag(σ_h²)          (n_series, n_series)

    3. **Gain**::

           K_t = P_t Z_tᵀ F_t⁻¹                     (n_states, n_series)

    4. **Update**::

           a_t ← a_t + K_t v_t
           P_t ← P_t − K_t Z_t P_t + diag(σ_q²)    (symmetrized)

    Parameters
    ----------
    a0 : jnp.ndarray, shape (n_states,)
        Initial state mean.
    P0 : jnp.ndarray, shape (n_states, n_states)
        Initial state covariance matrix.
    Z : jnp.ndarray, shape (n_steps, n_series, n_states)
        Per-series design matrix. Z[t, j] is the loading vector for series j at time t.
    sigma_h : float or jnp.ndarray, shape (n_series,)
        Observation noise standard deviation, one per series.
    sigma_q : float or jnp.ndarray, shape () or (n_states,)
        Process noise standard deviation. Broadcast to (n_states,) if scalar.
    y : jnp.ndarray, shape (n_steps, n_series)
        Observed time series, one column per series.
    logp : bool, optional
        Accumulate the Gaussian log-likelihood. Default False.
    a_obs : jnp.ndarray | None, optional, shape (n_steps, n_states)
        Externally disclosed latent state means. Set rows to zero and pair
        with ``P_obs=inf`` at timesteps with no disclosure.
    P_obs : jnp.ndarray | None, optional, shape (n_steps, n_states)
        Externally disclosed latent state variances. Use ``jnp.inf`` for
        timesteps / states with no external information. When both
        ``a_obs`` and ``P_obs`` are None the filter runs without
        any state fusion.
    positivity_idx : jnp.ndarray | None, optional, shape (n_states,)
        Boolean mask — True selects states that must remain non-negative.
        ``None`` (default) disables positivity correction for all states.

    Returns
    -------
    log_p : float
        Accumulated Gaussian log-likelihood. Returns 0.0 when ``logp=False``.
    at : jnp.ndarray, shape (n_steps, n_states)
        Filtered state estimates (posterior mean after obs update + process noise).
    Pt : jnp.ndarray, shape (n_steps, n_states, n_states)
        Filtered state covariance matrices.
    vt : jnp.ndarray, shape (n_steps, n_series)
        Innovation (observation residual) per series at each timestep.
    Ft : jnp.ndarray, shape (n_steps, n_series, n_series)
        Full innovation covariance matrix at each timestep.
    Kt : jnp.ndarray, shape (n_steps, n_states, n_series)
        Kalman gain matrix at each timestep.
    """
    logger.debug(
        "kalman_filter_1d_st inputs — a0: %s, P0: %s, Z: %s, y: %s, sigma_h: %s, sigma_q: %s",
        a0.shape,
        P0.shape,
        Z.shape,
        y.shape,
        getattr(sigma_h, "shape", sigma_h),
        getattr(sigma_q, "shape", sigma_q),
    )

    n_states = a0.shape[0]
    n_series = y.shape[1]
    sigma_h_sq = jnp.broadcast_to(jnp.square(sigma_h), (n_series,))
    sigma_q_sq = jnp.broadcast_to(jnp.square(sigma_q), (n_states,))

    # default: loc=0, var=inf → zero precision → fusion is a no-op at undisclosed steps
    _has_obs_fusion = a_obs is not None or P_obs is not None
    _at_obs = a_obs if a_obs is not None else jnp.zeros((y.shape[0], n_states))
    _Pt_obs = P_obs if P_obs is not None else jnp.full((y.shape[0], n_states), jnp.inf)

    _has_positivity = positivity_idx is not None
    _positivity_idx = positivity_idx if positivity_idx is not None else jnp.zeros(n_states, dtype=bool)

    def _transition_fn(
        carry: tuple[jnp.ndarray, jnp.ndarray, float],
        xs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple, tuple]:
        """Single Kalman filter step: fuse latent obs, predict, update, enforce positivity."""
        at, Pt, log_p = carry
        # (n_series,), (n_series, n_states), (n_states,), (n_states,)
        yt, Zt, at_obs, Pt_obs = xs

        # ------ Latent obs fusion (skipped entirely when no obs info provided) ------
        if _has_obs_fusion:
            # Precision-weighted Bayesian fusion: full-covariance prior + diagonal obs.
            # When Pt_obs = inf → prec_obs = 0 → no-op (pure filter carry-through).
            # (n_states,)
            prec_obs_diag = 1.0 / Pt_obs
            # (n_states, n_states)
            Pt_inv = jnp.linalg.solve(Pt, jnp.eye(n_states))
            P_fused_inv = Pt_inv + jnp.diag(prec_obs_diag)
            Pt = jnp.linalg.solve(P_fused_inv, jnp.eye(n_states))
            Pt = 0.5 * (Pt + Pt.T)
            at = Pt @ (Pt_inv @ at + prec_obs_diag * at_obs)

        # ------ Prediction step ------
        # Predicted observations: (n_series,)
        yhat = Zt @ at
        vt = yt - yhat

        # Full innovation covariance: (n_series, n_series)
        Ft = Zt @ Pt @ Zt.T + jnp.diag(sigma_h_sq)

        # Kalman gain via solve to avoid explicit inversion: (n_states, n_series)
        # K = Pt Zt.T Ft⁻¹  →  Kᵀ = Ft⁻ᵀ Zt Pt  →  solve(Ft, Zt @ Pt).T
        Kt = jnp.linalg.solve(Ft, Zt @ Pt).T

        if logp:
            _, log_det_F = jnp.linalg.slogdet(Ft)
            mahal = vt @ jnp.linalg.solve(Ft, vt)
            log_p = log_p - 0.5 * (n_series * jnp.log(2.0 * jnp.pi) + log_det_F + mahal)

        # ------ Update step ------
        at = at + Kt @ vt

        # Covariance update + process noise; symmetrize to prevent numerical drift
        Pt = Pt - Kt @ Zt @ Pt + jnp.diag(sigma_q_sq)
        Pt = 0.5 * (Pt + Pt.T)

        # ------ Positivity correction ------
        if _has_positivity:
            enforce = _positivity_idx & (at < 0)
            # Treat as pseudo-observation: observe a_i=1e-3 with var=1e-3 for violated states
            pos_loc = jnp.where(enforce, 1e-3, 0.0)
            pos_var = jnp.where(enforce, 1e-3, jnp.inf)
            # (n_states,)
            prec_obs_diag = 1.0 / pos_var
            Pt_inv = jnp.linalg.solve(Pt, jnp.eye(n_states))
            P_fused_inv = Pt_inv + jnp.diag(prec_obs_diag)
            Pt = jnp.linalg.solve(P_fused_inv, jnp.eye(n_states))
            Pt = 0.5 * (Pt + Pt.T)
            at = Pt @ (Pt_inv @ at + prec_obs_diag * pos_loc)
            # Hard floor for numerical stability
            at = jnp.where(_positivity_idx, jnp.maximum(at, 1e-6), at)

        return (at, Pt, log_p), (at, Pt, vt, Ft, Kt)

    (_, _, log_p), (at, Pt, vt, Ft, Kt) = lax.scan(
        _transition_fn,
        (a0, P0, 0.0),
        (y, Z, _at_obs, _Pt_obs),
        length=y.shape[0],
    )
    return log_p, at, Pt, vt, Ft, Kt
