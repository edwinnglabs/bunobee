from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax

logger = logging.getLogger("wunkui")


def kalman_filter_1d(
    # (n_states, )
    a0: jnp.ndarray,
    # (n_states, )
    P0: jnp.ndarray,
    # (n_steps, n_states)
    Z: jnp.ndarray,
    # (1, )
    sigma_h: Union[jnp.ndarray, float],
    # (n_states, )
    sigma_q: Union[jnp.array, float],
    # (n_steps,)
    y: jnp.array,
    logp: bool = False,
    # (n_steps, n_states) — observed latent state means; ignored where a_obs_var is inf
    a_obs_loc: Optional[jnp.ndarray] = None,
    # (n_steps, n_states) — observed latent state variances; inf = no information (pure filter)
    a_obs_var: Optional[jnp.ndarray] = None,
    # (n_states, )
    positivity_idx: Optional[jnp.ndarray] = None,
) -> Tuple[float, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.array, jnp.ndarray]:
    """Linear 1D Kalman filter with optional state fusion and positivity constraint.

    Overview
    --------
    Implements a diagonal-covariance linear Kalman filter for the state-space model::

        State evolution:   a_t = a_{t-1} + η_t,   η_t ~ N(0, σ_q²I)
        Observation model: y_t = Σ_i Z_{t,i} · a_{t,i} + ε_t,   ε_t ~ N(0, σ_h²)

    Each timestep runs four stages:

    1. **State fusion** (optional) — precision-weighted merge of the carried state
       with externally disclosed latent values before measurement::

           P_t ← 1 / (1/P_t + 1/a_obs_var_t)
           a_t ← P_t · (a_t/P_t + a_obs_loc_t/a_obs_var_t)

       Steps where ``a_obs_var = inf`` are no-ops (pure filter carry-through).

    2. **Predict** — compute predicted observation::

           ŷ_t = Σ_i Z_{t,i} · a_{t,i}

    3. **Update** — standard Kalman correction::

           v_t = y_t − ŷ_t
           F_t = Σ_i P_{t,i} · Z_{t,i}² + σ_h²
           K_t = P_t · Z_t / F_t
           a_t ← a_t + K_t · v_t
           P_t ← P_t · (1 − K_t · Z_t) + σ_q²

    4. **Positivity correction** (optional) — for states flagged by
       ``positivity_idx``, any updated value below zero is soft-clipped back
       toward 1e-3 via a second precision-weighted fusion, then hard-floored
       at 1e-6 to maintain a numerically stable gap from zero.

    Parameters
    ----------
    a0 : jnp.ndarray, shape (n_states,)
        Initial state mean.
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
        Accumulate the exact Gaussian log-likelihood. Default False.
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
        Filtered state estimates.
    Pt : jnp.ndarray, shape (n_steps, n_states)
        Filtered state variances (diagonal).
    vt : jnp.ndarray, shape (n_steps, 1)
        Innovation (observation residual) at each timestep.
    Ft : jnp.ndarray, shape (n_steps, 1)
        Innovation variance at each timestep.
    Kt : jnp.ndarray, shape (n_steps, n_states)
        Kalman gain vector at each timestep.
    """

    logger.debug("kalman_filter_1d inputs — a0: %s, P0: %s, Z: %s, y: %s, sigma_h: %s, sigma_q: %s",
                 a0.shape, P0.shape, Z.shape, y.shape,
                 getattr(sigma_h, "shape", sigma_h),
                 getattr(sigma_q, "shape", sigma_q))

    sigma_h_sq = jnp.square(sigma_h)
    sigma_q_sq = jnp.square(sigma_q)
    n_states = a0.shape[0]

    # default: loc=0, var=inf → zero precision → fusion is a no-op at undisclosed steps
    _has_obs_fusion = a_obs_loc is not None or a_obs_var is not None
    _a_obs_loc = a_obs_loc if a_obs_loc is not None else jnp.zeros((y.shape[0], n_states))
    _a_obs_var = a_obs_var if a_obs_var is not None else jnp.full((y.shape[0], n_states), jnp.inf)

    _has_positivity = positivity_idx is not None
    _positivity_idx = positivity_idx if positivity_idx is not None else jnp.zeros(n_states, dtype=bool)
    p = len(y)

    def _transition_fn(carry, xs):
        """transition function for Kalman filter"""

        # ------ Unpack ------
        # (n_states,), (n_states,), scalar
        at, Pt, log_p = carry
        t, at_obs_loc_t, at_obs_var_t = xs
        # scalar
        yt = y[t]
        # (n_states,)
        Zt = Z[t]

        # ------ Latent obs fusion (skipped entirely when no obs info provided) ------
        if _has_obs_fusion:
            # Bayesian Gaussian fusion of filter prior N(at, Pt) with disclosed obs
            # N(at_obs_loc_t, at_obs_var_t).
            # at_obs_var_t = inf → prec_obs = 0 → no-op (pure filter carry-through).
            prec_filter = 1.0 / Pt
            prec_obs    = 1.0 / at_obs_var_t
            Pt          = 1.0 / (prec_filter + prec_obs)
            at          = Pt * (prec_filter * at + prec_obs * at_obs_loc_t)

            # TODO: add logp if we observe latent states?

        # ------ Prediction step ------
        # (n_states,) * (n_states,) -> sum -> (1,)
        yhat = jnp.sum(Zt * at, -1, keepdims=True)

        # ------ Measurement step ------
        # scalar - (1,) -> (1,)
        vt = yt - yhat
        # (n_states,) * (n_states,) -> sum -> (1,)  +  scalar -> (1,)
        Ft = jnp.sum(Pt * jnp.square(Zt), -1, keepdims=True) + sigma_h_sq
        # (n_states,) * (n_states,) / (1,) -> (n_states,)
        Kt = Pt * Zt / Ft

        # scalar + scalar -> scalar
        if logp:
            log_p += -0.5 * (p * jnp.log(2 * jnp.pi) + jnp.sum(jnp.log(Ft) + jnp.square(vt) / Ft))

        # ------ Update step ------
        # to enforce positivity after Kalman update, we can either
        # in next measurement we use precision fusion
        # however, we also need to ensure such condition
        # when we return final estimate (assume we also do positivity in next step)
        # (n_states,) + (n_states,) * (1,) -> (n_states,)
        at = at + Kt * vt
        # (n_states,) * (n_states,) - (n_states,) * (n_states,) * (1,) -> (n_states,)
        Pt = Pt * (1 - Kt * Zt) + sigma_q_sq

        if _has_positivity:
            enforce = _positivity_idx & (at < 0)
            # soft adjustment jitter with 1e-3 to create numerically stable gap from zero
            at_adj = jnp.where(enforce, 1e-3, at)
            Pt_adj = jnp.where(enforce, 1e-3, Pt)
            prec_filter = 1.0 / Pt
            prec_obs    = 1.0 / Pt_adj
            Pt          = 1.0 / (prec_filter + prec_obs)
            at          = Pt * (prec_filter * at + prec_obs * at_adj)
            # avoid exact boundary issues
            at = jnp.where(_positivity_idx, jnp.maximum(at, 1e-6), at)

        return ((at, Pt, log_p), (at, Pt, vt, Ft, Kt))

    (_, _, log_p), (at, Pt, vt, Ft, Kt) = lax.scan(
        _transition_fn,
        (a0, P0, 0.0),
        (jnp.arange(y.shape[0]), _a_obs_loc, _a_obs_var),
        length=y.shape[0],
    )
    return log_p, at, Pt, vt, Ft, Kt


def kalman_filter_1d_ekf(
    a0: jnp.ndarray,
    P0: jnp.ndarray,
    Z: jnp.ndarray,
    sigma_h: Union[jnp.ndarray, float],
    sigma_q: Union[jnp.ndarray, float],
    y: jnp.ndarray,
    logp: bool = False,
    exponent: float = 0.5,
    positivity_idx: Optional[jnp.ndarray] = None,
    # (n_steps, n_states) — observed latent state means in a-space; ignored where a_obs_var is inf
    a_obs_loc: Optional[jnp.ndarray] = None,
    # (n_steps, n_states) — observed latent state variances in a-space; inf = no information
    a_obs_var: Optional[jnp.ndarray] = None,
) -> Tuple[float, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

           P_pred ← 1 / (1/P_pred + 1/a_obs_var_t)
           a_pred ← P_pred · (a_pred/P_pred + a_obs_loc_t/a_obs_var_t)

       Steps where ``a_obs_var = inf`` are no-ops (pure filter carry-through).
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
    a_obs_loc : jnp.ndarray | None, optional, shape (n_steps, n_states)
        Externally disclosed latent state means in a-space (before the exp
        transformation). Set rows to zero and pair with ``a_obs_var=inf`` at
        timesteps with no disclosure. Ignored when ``a_obs_var`` is inf.
    a_obs_var : jnp.ndarray | None, optional, shape (n_steps, n_states)
        Externally disclosed latent state variances in a-space. Use ``jnp.inf``
        for timesteps / states with no external information. When both
        ``a_obs_loc`` and ``a_obs_var`` are None the filter runs without any
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
    _has_obs_fusion = a_obs_loc is not None or a_obs_var is not None
    _a_obs_loc = a_obs_loc if a_obs_loc is not None else jnp.zeros((y.shape[0], n_states))
    _a_obs_var = a_obs_var if a_obs_var is not None else jnp.full((y.shape[0], n_states), jnp.inf)

    def _ekf_step(carry, xs):
        """Single EKF prediction–update step."""
        at, Pt, log_p = carry
        t, at_obs_loc_t, at_obs_var_t = xs

        Zt = Z[t]
        yt = y[t]

        # Prediction (linear state evolution for all states)
        a_pred = at
        P_pred = Pt + sigma_q_sq

        # ------ Latent state fusion in a-space (skipped when no obs info provided) ------
        # Bayesian precision-weighted fusion of predicted state N(a_pred, P_pred)
        # with disclosed a-space observation N(at_obs_loc_t, at_obs_var_t).
        # at_obs_var_t = inf → prec_obs = 0 → no-op (pure filter carry-through).
        # Fusion happens after prediction so process noise is already absorbed
        # before updating the linearisation point.
        if _has_obs_fusion:
            prec_filter = 1.0 / P_pred
            prec_obs = 1.0 / at_obs_var_t
            P_pred = 1.0 / (prec_filter + prec_obs)
            a_pred = P_pred * (prec_filter * a_pred + prec_obs * at_obs_loc_t)

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
        (jnp.arange(y.shape[0]), _a_obs_loc, _a_obs_var),
        length=y.shape[0],
    )
    return log_p, at, Pt, vt_arr, Ft_arr, Kt_arr


# this is simulation smoothing
# def kalman_smoother(
#     a0: jnp.ndarray,
#     P0: jnp.ndarray,
#     sigma_h: Union[jnp.ndarray, float],
#     sigma_q: Union[jnp.ndarray, float],
#     obs: jnp.ndarray,
#     rng_key: jnp.ndarray,
# ) -> jnp.ndarray:

#     # step 1. simulate obs based on non-optimized alpha
#     # step 2. run kalman smoother backward
#     # step 3. run kalman filter forward
#     # step 4. derive smoothed alpha for full dist. of alpha

#     n_steps = obs.shape[0]
#     sigma_q_sq = jnp.square(sigma_q)

#     rng_sub_keys = random.split(rng_key, 3)
#     # note that this alpha is ~ N(0, P0) not N(a0, P0)
#     init_sim_alpha = dist.Normal(0.0, 1.0).sample(
#         rng_sub_keys[0],
#     ) * jnp.sqrt(P0)

#     obs_noise = dist.Normal(0.0, 1.0).sample(
#         rng_sub_keys[1],
#         sample_shape=(n_steps, ),
#     ) * sigma_h

#     innov = dist.Normal(0.0, 1.0).sample(
#         rng_sub_keys[2],
#         sample_shape=(n_steps, ),
#     ) * sigma_q

#     def sim_obs_fn(carry, t):
#         alpha_t = carry

#         # observations simulate step
#         y_t = alpha_t + obs_noise[t]
#         alpha_t = alpha_t + innov[t]
#         return alpha_t, (alpha_t, y_t)

#     _, (alpha_plus, obs_plus) = lax.scan(
#         sim_obs_fn,
#         init=init_sim_alpha,
#         xs=jnp.arange(n_steps),
#         length=n_steps,
#     )

#     # output from scan has an additional shape in last dim
#     obs_diff = obs - jnp.squeeze(obs_plus, -1)

#     _, _, _, v, F, K = kalman_filter(
#         a0=a0,
#         P0=P0,
#         sigma_h=sigma_h,
#         sigma_q=sigma_q,
#         y=obs_diff,
#     )

#     def kalman_smoother_backward_fn(carry, t):
#         rt = carry
#         Lt = 1 - K[t]
#         rt = 1 / F[t] * v[t] + Lt * rt

#         return rt, rt

#     rT = jnp.zeros((1))
#     _, r = lax.scan(
#         kalman_smoother_backward_fn,
#         init=rT,
#         xs=jnp.arange(n_steps - 1, -1, -1),
#         length=n_steps,
#     )

#     r = jnp.squeeze(r, -1)
#     # flip on steps
#     r = jnp.flip(r, -1)
#     # include first step
#     r = jnp.concatenate([r, rT], -1)

#     def kalman_smoother_forward_fn(carry, t):
#         alpha_t = carry
#         # correction of mean vector given lookahead data points
#         # note that r vector is appended with initial value;
#         # so t + 1 means t in the actual maths
#         alpha_t = alpha_t + sigma_q_sq * r[t + 1]

#         return alpha_t, alpha_t

#     alpha_0 = a0 + P0 * r[0]

#     _, smoothed_obs_diff_alpha = lax.scan(
#         kalman_smoother_forward_fn,
#         init=alpha_0,
#         xs=jnp.arange(n_steps),
#         length=n_steps,
#     )   

#     smoothed_alpha = alpha_plus + smoothed_obs_diff_alpha
#     # (n_steps, )
#     return jnp.squeeze(smoothed_alpha, -1)


# def simulate_forecast(
#     # (n_sample, )
#     a0: jnp.ndarray,
#     # (n_sample, )
#     sigma_h: jnp.ndarray,
#     # (n_sample, )
#     sigma_q: jnp.ndarray,
#     # forecast horizon
#     n_steps: int,
#     rng_key: jnp.ndarray,
# ):
#     rng_sub_keys = random.split(rng_key, 2)
#     n_samples = sigma_h.shape[0]

#     # move steps to first dim to facilitate broadcast
#     # (n_steps, n_samples)
#     obs_noise = (
#         dist.Normal(loc=0.0, scale=1.0).sample(
#             rng_sub_keys[0], sample_shape=(n_steps, n_samples)
#         ) * sigma_h
#     )
#     innov = (
#         dist.Normal(loc=0.0, scale=1.0).sample(
#             rng_sub_keys[1], sample_shape=(n_steps, n_samples)
#         ) * sigma_q
#     )

#     def sim_transition_fn(carry, t):
#         at = carry
#         at = at + innov[t]

#         return at, at

#     _, states = lax.scan(
#         sim_transition_fn,
#         a0,
#         jnp.arange(n_steps),
#         length=n_steps,
#     )
#     # (n_steps, n_samples) -> (n_samples, n_steps)
#     res = states + obs_noise
#     res = jnp.swapaxes(res, -1, -2)
#     return res


# def simulate_forecast(
#     # (n_samples,) or (n_samples, n_state) when X_future is provided
#     a0: jnp.ndarray,
#     # (n_samples,)
#     sigma_h: jnp.ndarray,
#     # (n_samples,) or (n_samples, n_state) when X_future is provided
#     sigma_q: jnp.ndarray,
#     # forecast horizon
#     n_steps: int,
#     rng_key: jnp.ndarray,
#     # (n_steps, n_regressors) — future regressor values for time-varying coefficient forecast
#     X_future: Optional[jnp.ndarray] = None,
# ):
#     rng_sub_keys = random.split(rng_key, 2)
#     n_samples = sigma_h.shape[0]
#     has_X = X_future is not None

#     if has_X:
#         n_state = a0.shape[-1]

#         # (n_steps, n_samples, n_state)
#         innov = (
#             dist.Normal(loc=0.0, scale=1.0).sample(
#                 rng_sub_keys[1], sample_shape=(n_steps, n_samples, n_state)
#             ) * sigma_q
#         )
#         # (n_steps, n_samples)
#         obs_noise = (
#             dist.Normal(loc=0.0, scale=1.0).sample(
#                 rng_sub_keys[0], sample_shape=(n_steps, n_samples)
#             ) * sigma_h
#         )

#         def sim_transition_fn(carry, t):
#             at = carry                                                       # (n_samples, n_state)
#             at = at + innov[t]
#             return at, at

#         _, states = lax.scan(
#             sim_transition_fn, a0, jnp.arange(n_steps), length=n_steps,
#         )
#         # states: (n_steps, n_samples, n_state)
#         # Z_future: (n_steps, n_state) = [1, X_future]
#         Z_future = jnp.concatenate(
#             [jnp.ones((n_steps, 1)), X_future], axis=1
#         )
#         # obs: (n_steps, n_samples) via einsum, then swap to (n_samples, n_steps)
#         res = jnp.einsum("tsk,tk->ts", states, Z_future) + obs_noise
#         return jnp.swapaxes(res, -1, -2)

#     else:
#         # move steps to first dim to facilitate broadcast
#         # (n_steps, n_samples)
#         obs_noise = (
#             dist.Normal(loc=0.0, scale=1.0).sample(
#                 rng_sub_keys[0], sample_shape=(n_steps, n_samples)
#             ) * sigma_h
#         )
#         innov = (
#             dist.Normal(loc=0.0, scale=1.0).sample(
#                 rng_sub_keys[1], sample_shape=(n_steps, n_samples)
#             ) * sigma_q
#         )

#         def sim_transition_fn(carry, t):
#             at = carry
#             at = at + innov[t]

#             return at, at

#         _, states = lax.scan(
#             sim_transition_fn,
#             a0,
#             jnp.arange(n_steps),
#             length=n_steps,
#         )
#         # (n_steps, n_samples) -> (n_samples, n_steps)
#         res = states + obs_noise
#         res = jnp.swapaxes(res, -1, -2)
#         return res
