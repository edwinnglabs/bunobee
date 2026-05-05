from __future__ import annotations

import logging

import jax.numpy as jnp
from jax import lax

logger = logging.getLogger(__name__)


def kalman_filter_1d_ekf(
    a0: jnp.ndarray,
    P0: jnp.ndarray,
    Z: jnp.ndarray,
    sigma_h: jnp.ndarray | float,
    sigma_q: jnp.ndarray | float,
    y: jnp.ndarray,
    logp: bool = False,
    exponent: float = 0.5,
    positivity_idx: jnp.ndarray | None = None,
    # (n_steps, n_states) — observed latent state means in a-space; ignored where P_obs is inf
    a_obs: jnp.ndarray | None = None,
    # (n_steps, n_states) — observed latent state variances in a-space; inf = no information
    P_obs: jnp.ndarray | None = None,
) -> tuple[float, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """1D Extended Kalman Filter with optional log-state reparameterization.

    Overview
    --------
    Implements a diagonal-covariance EKF for the state-space model::

        State evolution:   a_t = a_{t-1} + η_t,   η_t ~ N(0, σ_q²I)
        Observation model: y_t = Σ_i Z_{t,i} · h(a_{t,i}) + ε_t,   ε_t ~ N(0, σ_h²)

    Each timestep runs four stages:

    1. **Predict** — propagate state forward:
       a_pred = a_{t-1},  P_pred = P_{t-1} + σ_q²

    2. **State fusion** (optional) — precision-weighted merge of the predicted
       state with any externally disclosed latent values in a-space::

           P_pred ← 1 / (1/P_pred + 1/Pt_obs)
           a_pred ← P_pred · (a_pred/P_pred + at_obs/Pt_obs)

       Steps where ``P_obs = inf`` are no-ops (pure filter carry-through).
       Fusion is applied after prediction so process noise is already absorbed
       before the linearisation point is updated.

    3. **Linearise** — compute per-state effective values and Jacobians:

       - Nonlinear states (``positivity_idx=True``):
         h(a_i) = exp(exponent · a_i),  H_i = exponent · Z_i · exp(exponent · a_i)
       - Linear states (``positivity_idx=False``):
         h(a_i) = a_i,  H_i = Z_i

       ``positivity_idx=None`` treats all states as nonlinear (default).
       Pass ``jnp.zeros(n_states, dtype=bool)`` to recover a fully linear KF.

    4. **Update** — standard Kalman correction::

           ŷ_t = Σ_i Z_{t,i} · h(a_{pred,i})
           v_t = y_t − ŷ_t
           F_t = Σ_i P_pred,i · H_{t,i}² + σ_h²
           K_t = P_pred · H_t / F_t
           a_t = a_pred + K_t · v_t
           P_t = P_pred · (1 − K_t · H_t)

    The Gaussian log-likelihood is a Laplace approximation — exact only for
    fully linear models. ``exponent=0.5`` (default) gives a softer nonlinearity
    (exp(0.5) ≈ 1.65 per unit change); ``exponent=1.0`` is the standard log-state
    EKF (scale factor e ≈ 2.72).

    Parameters
    ----------
    a0 : jnp.ndarray, shape (n_states,)
        Initial state mean. For nonlinear states this is in log-intensity space;
        set ``a0[i] = log(v) / exponent`` to start state i at intensity v.
        For linear states it is the direct value.
    P0 : jnp.ndarray, shape (n_states,)
        Initial state variance (diagonal).
    Z : jnp.ndarray, shape (n_steps, n_states)
        Design / measurement matrix. Each row Z[t] is the loading vector at time t.
    sigma_h : float or jnp.ndarray, shape ()
        Observation noise standard deviation (scalar).
    sigma_q : float or jnp.ndarray, shape () or (n_states,)
        Process noise standard deviation. Broadcast to (n_states,) if scalar.
    y : jnp.ndarray, shape (n_steps,)
        Observed scalar time series.
    logp : bool, optional
        Accumulate the approximate Gaussian log-likelihood. Default False.
    exponent : float, optional
        Exponent in the nonlinear mapping ``exp(exponent · a_t)``. Default 0.5.
    positivity_idx : jnp.ndarray | None, optional, shape (n_states,)
        Boolean mask — True selects states that use the nonlinear exp mapping.
        ``None`` (default) applies the nonlinear mapping to all states.
        Pass ``jnp.zeros(n_states, dtype=bool)`` for a fully linear filter.
    a_obs : jnp.ndarray | None, optional, shape (n_steps, n_states)
        Externally disclosed latent state means in a-space (before the exp
        transformation). Set rows to zero and pair with ``P_obs=inf`` at
        timesteps with no disclosure. Ignored when ``P_obs`` is inf.
    P_obs : jnp.ndarray | None, optional, shape (n_steps, n_states)
        Externally disclosed latent state variances in a-space. Use ``jnp.inf``
        for timesteps / states with no external information. When both
        ``a_obs`` and ``P_obs`` are None the filter runs without any
        state disclosure.

    Returns
    -------
    log_p : float
        Accumulated approximate Gaussian log-likelihood. Returns 0.0 when
        ``logp=False``.
    at : jnp.ndarray, shape (n_steps, n_states)
        Filtered state estimates in a-space. Recover intensities for nonlinear
        states via ``jnp.exp(exponent * at)``.
    Pt : jnp.ndarray, shape (n_steps, n_states)
        Filtered state variances (diagonal) in a-space.
    vt_arr : jnp.ndarray, shape (n_steps,)
        Innovation (observation residual) at each timestep.
    Ft_arr : jnp.ndarray, shape (n_steps,)
        Innovation variance at each timestep.
    Kt_arr : jnp.ndarray, shape (n_steps, n_states)
        Kalman gain vector at each timestep.
    """
    logger.debug(
        "kalman_filter_1d_ekf inputs — a0: %s, P0: %s, Z: %s, y: %s, sigma_h: %s, sigma_q: %s",
        a0.shape, P0.shape, Z.shape, y.shape,
        getattr(sigma_h, "shape", sigma_h),
        getattr(sigma_q, "shape", sigma_q),
    )

    sigma_h_sq = jnp.square(sigma_h)
    sigma_q_sq = jnp.square(sigma_q)
    n_states = a0.shape[0]

    # None → all states use the nonlinear exp mapping
    _positivity = positivity_idx if positivity_idx is not None else jnp.ones(n_states, dtype=bool)

    # default: loc=0, var=inf → zero precision → fusion is a no-op at undisclosed steps
    _has_obs_fusion = a_obs is not None or P_obs is not None
    _at_obs = a_obs if a_obs is not None else jnp.zeros((y.shape[0], n_states))
    _Pt_obs = P_obs if P_obs is not None else jnp.full((y.shape[0], n_states), jnp.inf)

    def _ekf_step(carry, xs):
        """Single EKF prediction–update step."""
        at, Pt, log_p = carry
        t, at_obs, Pt_obs = xs

        Zt = Z[t]
        yt = y[t]

        # Prediction (linear state evolution for all states)
        a_pred = at
        P_pred = Pt + sigma_q_sq

        # ------ Latent state fusion in a-space (skipped when no obs info provided) ------
        # Bayesian precision-weighted fusion of predicted state N(a_pred, P_pred)
        # with disclosed a-space observation N(at_obs, Pt_obs).
        # Pt_obs = inf → prec_obs = 0 → no-op (pure filter carry-through).
        # Fusion happens after prediction so process noise is already absorbed
        # before updating the linearisation point.
        if _has_obs_fusion:
            prec_filter = 1.0 / P_pred
            prec_obs = 1.0 / Pt_obs
            P_pred = 1.0 / (prec_filter + prec_obs)
            a_pred = P_pred * (prec_filter * a_pred + prec_obs * at_obs)

        # exp(exponent·a_pred); clip to avoid overflow
        exp_a = jnp.exp(jnp.clip(exponent * a_pred, -10.0, 10.0))

        # Per-state effective observation value:
        #   nonlinear states → exp(exponent·a),  linear states → a directly
        a_eff = jnp.where(_positivity, exp_a, a_pred)

        # Per-state Jacobian:
        #   nonlinear states → exponent·Z·exp(exponent·a),  linear states → Z
        Ht = jnp.where(_positivity, exponent * Zt * exp_a, Zt)

        # Predicted observation
        yhat = jnp.sum(Zt * a_eff)

        # Innovation (scalar)
        vt = yt - yhat

        # Innovation variance: F_t = sum(P_pred · H_t²) + σ_h²
        Ft = jnp.sum(P_pred * jnp.square(Ht)) + sigma_h_sq

        # Kalman gain
        Kt = P_pred * Ht / Ft

        # Approximate Gaussian log-likelihood (Laplace approximation)
        log_p = jnp.where(
            logp,
            log_p + -0.5 * (jnp.log(2 * jnp.pi) + jnp.log(Ft) + jnp.square(vt) / Ft),
            log_p,
        )

        # Update (P(1 - KH) is the diagonal approximation of P - KHP)
        at_new = a_pred + Kt * vt
        Pt_new = P_pred * (1.0 - Kt * Ht)

        return (at_new, Pt_new, log_p), (at_new, Pt_new, vt, Ft, Kt)

    (_, _, log_p), (at, Pt, vt_arr, Ft_arr, Kt_arr) = lax.scan(
        _ekf_step,
        (a0, P0, 0.0),
        (jnp.arange(y.shape[0]), _at_obs, _Pt_obs),
        length=y.shape[0],
    )
    return log_p, at, Pt, vt_arr, Ft_arr, Kt_arr


def kalman_dk_smoother_1d_ekf(
    at: jnp.ndarray,
    Pt: jnp.ndarray,
    vt: jnp.ndarray,
    Ft: jnp.ndarray,
    Kt: jnp.ndarray,
    Z: jnp.ndarray,
    a0: jnp.ndarray,
    P0: jnp.ndarray,
    sigma_q: jnp.ndarray | float,
    exponent: float = 0.5,
    positivity_idx: jnp.ndarray | None = None,
    a_obs: jnp.ndarray | None = None,
    P_obs: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Durbin-Koopman disturbance smoother for the diagonal 1D EKF state-space model.

    Runs a single backward pass over the stored filter quantities to recover
    the full-data posterior mean ``E[α_t | Y_n]`` for all ``t`` in a-space
    (recover intensities for nonlinear states via ``jnp.exp(exponent *
    at_smooth)``).  ``at``, ``Pt``, ``vt``, ``Ft``, ``Kt`` come directly from
    :func:`kalman_filter_1d_ekf`; ``Z``, ``a0``, ``P0``, ``sigma_q``,
    ``exponent``, ``positivity_idx``, ``a_obs``, ``P_obs`` must match
    the values originally passed to the filter.

    Backward recursion mirrors the linear D&K smoother but substitutes the
    per-timestep linearisation Jacobian ``H_t`` for ``Z_t``::

        H_t,i = exponent · Z_t,i · exp(exponent · ã_pred_t,i)   for nonlinear states
        H_t,i = Z_t,i                                            for linear states

    where ``ã_pred_t`` is the *post-fusion* predicted state — the same
    linearisation point the filter uses.  The y-update reverse step is::

        L_y       = 1 − K_t ⊙ H_t
        r_after_y = H_t ⊙ v_t / F_t + L_y ⊙ r_{t+1}

    The optional fusion pseudo-observation (Z_fusion = I) is processed last
    in the backward pass.  At undisclosed elements (``P_obs = inf``) the
    fusion contribution is 0 and ``L_fusion = 1``::

        F_fusion  = P_pred_t + Pt_obs
        L_fusion  = Pt_obs / F_fusion
        r_t       = (at_obs − a_pred_t) / F_fusion + L_fusion ⊙ r_after_y
        α̂_t      = a_pred_t + P_pred_t ⊙ r_t

    Because :func:`kalman_filter_1d_ekf` stores ``Pt[t] = P_{t|t}`` (pure
    posterior — process noise is applied at the *next* step's predict, unlike
    the linear filter), the D&K predicted variance is recovered by shifting
    and adding σ_q² uniformly::

        a_pred = [a0, at[0], at[1], ..., at[T-2]]
        P_pred = [P0, Pt[0], Pt[1], ..., Pt[T-2]] + σ_q²

    The smoother is approximate — exact only when every state is linear —
    and reuses the filter's Laplace linearisation point.

    Parameters
    ----------
    at : jnp.ndarray, shape (T, n_states)
        Filtered state means in a-space from :func:`kalman_filter_1d_ekf`.
    Pt : jnp.ndarray, shape (T, n_states)
        Filtered state variances (pure posterior ``P_{t|t}``).
    vt : jnp.ndarray, shape (T,)
        Innovations from :func:`kalman_filter_1d_ekf`.
    Ft : jnp.ndarray, shape (T,)
        Innovation variances from :func:`kalman_filter_1d_ekf`.
    Kt : jnp.ndarray, shape (T, n_states)
        Kalman gains from :func:`kalman_filter_1d_ekf`.
    Z : jnp.ndarray, shape (T, n_states)
        Design / measurement matrix (same as passed to the filter).
    a0 : jnp.ndarray, shape (n_states,)
        Initial state mean (same as passed to the filter).
    P0 : jnp.ndarray, shape (n_states,)
        Initial state variance (same as passed to the filter).
    sigma_q : float or jnp.ndarray
        Process noise standard deviation (same as passed to the filter).
    exponent : float, optional
        Exponent in the nonlinear mapping ``exp(exponent · a_t)``; must match
        the value used by the filter. Default 0.5.
    positivity_idx : jnp.ndarray | None, optional, shape (n_states,)
        Boolean mask — True selects states using the nonlinear exp mapping
        (must match the filter). ``None`` (default) treats all states as
        nonlinear. Pass ``jnp.zeros(n_states, dtype=bool)`` to recover a
        fully linear smoother.
    a_obs : jnp.ndarray | None, optional, shape (T, n_states)
        Disclosed latent state means in a-space passed to the filter.
        Use ``None`` when no fusion was applied.
    P_obs : jnp.ndarray | None, optional, shape (T, n_states)
        Disclosed latent state variances in a-space passed to the filter.
        Use ``jnp.inf`` at undisclosed timesteps / states; ``None`` for no
        fusion at all.

    Returns
    -------
    at_smooth : jnp.ndarray, shape (T, n_states)
        Smoothed state means ``E[α_t | Y_n]`` in a-space.
    """
    T, n_states = at.shape
    sigma_q_sq = jnp.square(sigma_q)

    # None → all states nonlinear (matches filter default).
    _positivity = positivity_idx if positivity_idx is not None else jnp.ones(n_states, dtype=bool)

    # Pre-fusion predicted state.  EKF stores Pt[t] = P_{t|t} (without σ_q²),
    # so shifting requires adding σ_q² at every t — including t=0, since the
    # filter's first-step P_pred is P0 + σ_q².
    a_pred = jnp.concatenate([a0[None], at[:-1]], axis=0)
    P_pred = jnp.concatenate([P0[None], Pt[:-1]], axis=0) + sigma_q_sq

    if a_obs is None or P_obs is None:
        a_obs = jnp.zeros((T, n_states))
        P_obs = jnp.full((T, n_states), jnp.inf)

    # Post-fusion predicted state — the linearisation point used by the filter.
    # At elements with P_obs = inf, prec_obs = 0 → fusion is a no-op.
    finite_obs = jnp.isfinite(P_obs)
    safe_obs_var = jnp.where(finite_obs, P_obs, 1.0)
    prec_filter = 1.0 / P_pred
    prec_obs = jnp.where(finite_obs, 1.0 / safe_obs_var, 0.0)
    P_pred_pf = 1.0 / (prec_filter + prec_obs)
    a_pred_pf = P_pred_pf * (prec_filter * a_pred + prec_obs * a_obs)

    # Clip mirrors the filter's overflow guard so the linearisation matches exactly.
    exp_a_pf = jnp.exp(jnp.clip(exponent * a_pred_pf, -10.0, 10.0))
    Ht = jnp.where(_positivity, exponent * Z * exp_a_pf, Z)

    def _r_step(r_next: jnp.ndarray, t: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Single backward step: y-update first, then fusion pseudo-observation."""
        Lt_y = 1.0 - Kt[t] * Ht[t]
        r_after_y = Ht[t] * vt[t] / Ft[t] + Lt_y * r_next

        finite = jnp.isfinite(P_obs[t])
        safe_var = jnp.where(finite, P_obs[t], 1.0)
        F_fusion = P_pred[t] + safe_var
        fusion_v_term = jnp.where(
            finite, (a_obs[t] - a_pred[t]) / F_fusion, 0.0
        )
        Lt_fusion = jnp.where(finite, safe_var / F_fusion, 1.0)
        r_t = fusion_v_term + Lt_fusion * r_after_y

        return r_t, r_t

    _, r_all = lax.scan(
        _r_step,
        jnp.zeros(n_states),
        jnp.arange(T),
        reverse=True,
        length=T,
    )

    return a_pred + P_pred * r_all


def kalman_rts_smoother_1d_ekf(
    at: jnp.ndarray,
    Pt: jnp.ndarray,
    sigma_q: jnp.ndarray | float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Approximate RTS smoother for the diagonal 1D EKF state-space model.

    Runs an extended Rauch-Tung-Striebel backward pass using the EKF filtered
    posterior means and diagonal variances. The state transition is still the
    identity random walk,

        α_t = α_{t-1} + η_t,   η_t ~ N(0, σ_q² I),

    so the transition Jacobian is exactly the identity even though the
    observation model is nonlinear. Under the EKF approximation, the standard
    RTS recursion therefore specialises element-wise to

        P_{t+1|t} = P_{t|t} + σ_q²
        J_t       = P_{t|t} / P_{t+1|t}
        a_{t|T}   = a_{t|t} + J_t · (a_{t+1|T} - a_{t+1|t})
        P_{t|T}   = P_{t|t} + J_t² · (P_{t+1|T} - P_{t+1|t})

    with ``a_{t+1|t} = a_{t|t}`` because the transition is the identity.

    Unlike :func:`kalman_filter_1d`, :func:`kalman_filter_1d_ekf` stores the
    pure posterior variance ``Pt[t] = P_{t|t}``, so the predicted covariance
    needed by RTS is recovered as ``Pt[t] + σ_q²``. The terminal condition is
    therefore simply ``a_{T-1|T} = at[-1]`` and ``P_{T-1|T} = Pt[-1]``.

    The smoother is approximate for the same reason the EKF itself is
    approximate: the observation model is linearised around the filter's
    forward pass. Disclosure fusion (``a_obs``, ``P_obs``) is already absorbed
    into the filtered posterior ``at``, ``Pt``, so no extra smoother arguments
    are required.

    Parameters
    ----------
    at : jnp.ndarray, shape (T, n_states)
        Filtered state means in a-space from :func:`kalman_filter_1d_ekf`.
    Pt : jnp.ndarray, shape (T, n_states)
        Filtered state variances (pure posterior ``P_{t|t}``) from
        :func:`kalman_filter_1d_ekf`.
    sigma_q : float or jnp.ndarray, shape () or (n_states,)
        Process noise standard deviation, the same value passed to the filter.

    Returns
    -------
    at_smooth : jnp.ndarray, shape (T, n_states)
        Smoothed state means ``E[α_t | Y_n]`` in a-space.
    Pt_smooth : jnp.ndarray, shape (T, n_states)
        Approximate marginal smoothed state variances ``Var[α_t | Y_n]`` in
        a-space.
    """
    logger.debug(
        "kalman_rts_smoother_1d_ekf inputs — at: %s, Pt: %s, sigma_q: %s",
        at.shape, Pt.shape, getattr(sigma_q, "shape", sigma_q),
    )

    if at.shape[0] == 1:
        return at, Pt

    sigma_q_sq = jnp.square(sigma_q)
    P_pred_next = Pt[:-1] + sigma_q_sq
    smoother_gain = Pt[:-1] / P_pred_next

    def _step(carry: tuple[jnp.ndarray, jnp.ndarray], t: int) -> tuple[
        tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ]:
        """Single backward extended RTS step at time t."""
        a_smooth_next, P_smooth_next = carry
        a_smooth_t = at[t] + smoother_gain[t] * (a_smooth_next - at[t])
        P_smooth_t = Pt[t] + jnp.square(smoother_gain[t]) * (P_smooth_next - P_pred_next[t])
        return (a_smooth_t, P_smooth_t), (a_smooth_t, P_smooth_t)

    init = (at[-1], Pt[-1])
    _, (at_prefix, Pt_prefix) = lax.scan(
        _step,
        init,
        jnp.arange(at.shape[0] - 1),
        reverse=True,
        length=at.shape[0] - 1,
    )

    at_smooth = jnp.concatenate([at_prefix, at[-1][None]], axis=0)
    Pt_smooth = jnp.concatenate([Pt_prefix, Pt[-1][None]], axis=0)
    return at_smooth, Pt_smooth

