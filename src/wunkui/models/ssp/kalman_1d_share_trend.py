from __future__ import annotations

import logging

import jax.numpy as jnp
from jax import lax

logger = logging.getLogger("wunkui")


def kalman_filter_1d_st(
    a0: jnp.ndarray,
    P0: jnp.ndarray,
    Z: jnp.ndarray,
    sigma_h: jnp.ndarray | float,
    sigma_q: jnp.ndarray | float,
    y: jnp.ndarray,
    logp: bool = False,
    a_obs_loc: jnp.ndarray | None = None,
    a_obs_var: jnp.ndarray | None = None,
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
EW
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
    a_obs_loc : jnp.ndarray | None, optional, shape (n_steps, n_states)
        Externally disclosed latent state means. Set rows to zero and pair
        with ``a_obs_var=inf`` at timesteps with no disclosure.
    a_obs_var : jnp.ndarray | None, optional, shape (n_steps, n_states)
        Externally disclosed latent state variances. Use ``jnp.inf`` for
        timesteps / states with no external information. When both
        ``a_obs_loc`` and ``a_obs_var`` are None the filter runs without
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
        a0.shape, P0.shape, Z.shape, y.shape,
        getattr(sigma_h, "shape", sigma_h),
        getattr(sigma_q, "shape", sigma_q),
    )

    n_states = a0.shape[0]
    n_series = y.shape[1]
    sigma_h_sq = jnp.broadcast_to(jnp.square(sigma_h), (n_series,))
    sigma_q_sq = jnp.broadcast_to(jnp.square(sigma_q), (n_states,))

    # default: loc=0, var=inf → zero precision → fusion is a no-op at undisclosed steps
    _has_obs_fusion = a_obs_loc is not None or a_obs_var is not None
    _a_obs_loc = a_obs_loc if a_obs_loc is not None else jnp.zeros((y.shape[0], n_states))
    _a_obs_var = a_obs_var if a_obs_var is not None else jnp.full((y.shape[0], n_states), jnp.inf)

    _has_positivity = positivity_idx is not None
    _positivity_idx = positivity_idx if positivity_idx is not None else jnp.zeros(n_states, dtype=bool)

    def _transition_fn(
        carry: tuple[jnp.ndarray, jnp.ndarray, float],
        xs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple, tuple]:
        at, Pt, log_p = carry
        # (n_series,), (n_series, n_states), (n_states,), (n_states,)
        yt, Zt, at_obs_loc_t, at_obs_var_t = xs

        # ------ Latent obs fusion (skipped entirely when no obs info provided) ------
        if _has_obs_fusion:
            # Precision-weighted Bayesian fusion: full-covariance prior + diagonal obs.
            # When at_obs_var_t = inf → prec_obs = 0 → no-op (pure filter carry-through).
            prec_obs_diag = 1.0 / at_obs_var_t                         # (n_states,)
            Pt_inv = jnp.linalg.solve(Pt, jnp.eye(n_states))          # (n_states, n_states)
            P_fused_inv = Pt_inv + jnp.diag(prec_obs_diag)
            Pt = jnp.linalg.solve(P_fused_inv, jnp.eye(n_states))
            Pt = 0.5 * (Pt + Pt.T)
            at = Pt @ (Pt_inv @ at + prec_obs_diag * at_obs_loc_t)

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
            prec_obs_diag = 1.0 / pos_var                              # (n_states,)
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
        (y, Z, _a_obs_loc, _a_obs_var),
        length=y.shape[0],
    )
    return log_p, at, Pt, vt, Ft, Kt


def kalman_filter_1d_ekf_st(
    a0: jnp.ndarray,
    P0: jnp.ndarray,
    Z: jnp.ndarray,
    sigma_h: jnp.ndarray | float,
    sigma_q: jnp.ndarray | float,
    y: jnp.ndarray,
    logp: bool = False,
    exponent: float = 0.5,
    positivity_idx: jnp.ndarray | None = None,
    a_obs_loc: jnp.ndarray | None = None,
    a_obs_var: jnp.ndarray | None = None,
) -> tuple[float, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Multi-series EKF with shared latent state, full covariance, and log-state reparameterization.

    Overview
    --------
    Combines the nonlinear EKF mapping from ``kalman_filter_1d_ekf`` with the multi-series
    shared-state full-covariance structure of ``kalman_filter_1d_st``::

        State evolution:    a_t = a_{t-1} + η_t,                              η_t ~ N(0, diag(σ_q²))
        Observation model:  y_{t,j} = Σ_i Z_{t,j,i} · h(a_{t,i}) + ε_{t,j},  ε_{t,j} ~ N(0, σ_{h,j}²)

    where ``h(a_i) = exp(exponent · a_i)`` for nonlinear states and ``h(a_i) = a_i`` for linear.

    Each timestep runs four stages:

    1. **Predict** — propagate state forward, absorbing process noise::

           a_pred = a_{t-1}
           P_pred = P_{t-1} + diag(σ_q²)

    2. **State fusion** (optional) — precision-weighted merge of the predicted state with
       externally disclosed a-space observations (full-covariance form)::

           P_pred ← (P_pred⁻¹ + diag(1/a_obs_var_t))⁻¹
           a_pred ← P_pred · (P_pred_old⁻¹ · a_pred + a_obs_loc_t / a_obs_var_t)

       Steps where ``a_obs_var = inf`` are no-ops.

    3. **Linearise** — compute per-state effective values and Jacobians around a_pred::

           exp_a    = exp(exponent · a_pred)
           a_eff    = exp_a          for nonlinear states,  a_pred    for linear
           jac_diag = exponent·exp_a for nonlinear states,  1         for linear
           H_t      = Z_t * jac_diag                        (n_series, n_states)
           ŷ_t      = Z_t @ a_eff                           (n_series,)

    4. **Update** — Kalman correction using the linearised Jacobian H_t::

           v_t = y_t − ŷ_t
           F_t = H_t P_pred H_tᵀ + diag(σ_h²)
           K_t = P_pred H_tᵀ F_t⁻¹
           a_t = a_pred + K_t v_t
           P_t = P_pred − K_t H_t P_pred    (symmetrized)

    Parameters
    ----------
    a0 : jnp.ndarray, shape (n_states,)
        Initial state mean. For nonlinear states this is in log-intensity space;
        set ``a0[i] = log(v) / exponent`` to start state i at intensity v.
    P0 : jnp.ndarray, shape (n_states, n_states)
        Initial state covariance matrix (full, not diagonal).
    Z : jnp.ndarray, shape (n_steps, n_series, n_states)
        Per-series design matrix. ``Z[t, j]`` is the loading vector for series j at time t.
    sigma_h : float or jnp.ndarray, shape (n_series,)
        Observation noise standard deviation, one per series.
    sigma_q : float or jnp.ndarray, shape () or (n_states,)
        Process noise standard deviation. Broadcast to (n_states,) if scalar.
    y : jnp.ndarray, shape (n_steps, n_series)
        Observed time series, one column per series.
    logp : bool, optional
        Accumulate the approximate multivariate Gaussian log-likelihood. Default False.
    exponent : float, optional
        Exponent in the nonlinear mapping ``exp(exponent · a_t)``. Default 0.5.
    positivity_idx : jnp.ndarray | None, optional, shape (n_states,)
        Boolean mask — True selects states that use the nonlinear exp mapping.
        ``None`` (default) applies the nonlinear mapping to all states.
        Pass ``jnp.zeros(n_states, dtype=bool)`` to recover the linear ``kalman_filter_1d_st``.
    a_obs_loc : jnp.ndarray | None, optional, shape (n_steps, n_states)
        Externally disclosed latent state means in a-space (before the exp transformation).
        Set rows to zero and pair with ``a_obs_var=inf`` at timesteps with no disclosure.
    a_obs_var : jnp.ndarray | None, optional, shape (n_steps, n_states)
        Externally disclosed latent state variances in a-space. Use ``jnp.inf`` for
        timesteps / states with no external information.

    Returns
    -------
    log_p : float
        Accumulated approximate Gaussian log-likelihood. Returns 0.0 when ``logp=False``.
    at : jnp.ndarray, shape (n_steps, n_states)
        Filtered state estimates in a-space.
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
        "kalman_filter_1d_ekf_st inputs — a0: %s, P0: %s, Z: %s, y: %s, sigma_h: %s, sigma_q: %s",
        a0.shape, P0.shape, Z.shape, y.shape,
        getattr(sigma_h, "shape", sigma_h),
        getattr(sigma_q, "shape", sigma_q),
    )

    n_states = a0.shape[0]
    n_series = y.shape[1]
    sigma_h_sq = jnp.broadcast_to(jnp.square(sigma_h), (n_series,))
    sigma_q_sq = jnp.broadcast_to(jnp.square(sigma_q), (n_states,))

    # None → all states use the nonlinear exp mapping
    _positivity = positivity_idx if positivity_idx is not None else jnp.ones(n_states, dtype=bool)

    # default: loc=0, var=inf → zero precision → fusion is a no-op at undisclosed steps
    _has_obs_fusion = a_obs_loc is not None or a_obs_var is not None
    _a_obs_loc = a_obs_loc if a_obs_loc is not None else jnp.zeros((y.shape[0], n_states))
    _a_obs_var = a_obs_var if a_obs_var is not None else jnp.full((y.shape[0], n_states), jnp.inf)

    def _ekf_st_step(
        carry: tuple[jnp.ndarray, jnp.ndarray, float],
        xs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple, tuple]:
        at, Pt, log_p = carry
        yt, Zt, at_obs_loc_t, at_obs_var_t = xs

        # ------ Predict: absorb process noise before fusion and linearisation ------
        a_pred = at
        P_pred = Pt + jnp.diag(sigma_q_sq)
        P_pred = 0.5 * (P_pred + P_pred.T)

        # ------ State fusion in a-space (full-covariance form) ------
        # Bayesian precision-weighted fusion: when at_obs_var_t = inf → prec_obs = 0 → no-op.
        if _has_obs_fusion:
            prec_obs_diag = 1.0 / at_obs_var_t                              # (n_states,)
            Pt_inv = jnp.linalg.solve(P_pred, jnp.eye(n_states))
            P_fused_inv = Pt_inv + jnp.diag(prec_obs_diag)
            P_pred = jnp.linalg.solve(P_fused_inv, jnp.eye(n_states))
            P_pred = 0.5 * (P_pred + P_pred.T)
            a_pred = P_pred @ (Pt_inv @ a_pred + prec_obs_diag * at_obs_loc_t)

        # ------ Linearise around a_pred ------
        # exp(exponent·a_pred); clip to avoid overflow
        exp_a = jnp.exp(jnp.clip(exponent * a_pred, -10.0, 10.0))           # (n_states,)

        # Per-state effective observation value: exp mapping or identity
        a_eff = jnp.where(_positivity, exp_a, a_pred)                        # (n_states,)

        # Per-state Jacobian scalar: d h(a_i)/d a_i
        jac_diag = jnp.where(_positivity, exponent * exp_a, 1.0)             # (n_states,)

        # Linearised measurement matrix: H[j, i] = Z[j, i] * jac_diag[i]
        Ht = Zt * jac_diag                                                    # (n_series, n_states)

        # Predicted observations and innovations
        yhat = Zt @ a_eff                                                     # (n_series,)
        vt = yt - yhat                                                        # (n_series,)

        # ------ Update ------
        # Innovation covariance using linearised H (full matrix)
        Ft = Ht @ P_pred @ Ht.T + jnp.diag(sigma_h_sq)                      # (n_series, n_series)

        # Kalman gain via solve to avoid explicit inversion: K = P H.T F⁻¹
        Kt = jnp.linalg.solve(Ft, Ht @ P_pred).T                             # (n_states, n_series)

        # Approximate multivariate Gaussian log-likelihood (Laplace approximation)
        if logp:
            _, log_det_F = jnp.linalg.slogdet(Ft)
            mahal = vt @ jnp.linalg.solve(Ft, vt)
            log_p = log_p - 0.5 * (n_series * jnp.log(2.0 * jnp.pi) + log_det_F + mahal)

        at_new = a_pred + Kt @ vt
        Pt_new = P_pred - Kt @ Ht @ P_pred
        Pt_new = 0.5 * (Pt_new + Pt_new.T)

        return (at_new, Pt_new, log_p), (at_new, Pt_new, vt, Ft, Kt)

    (_, _, log_p), (at, Pt, vt, Ft, Kt) = lax.scan(
        _ekf_st_step,
        (a0, P0, 0.0),
        (y, Z, _a_obs_loc, _a_obs_var),
        length=y.shape[0],
    )
    return log_p, at, Pt, vt, Ft, Kt
