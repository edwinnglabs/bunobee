from __future__ import annotations

import logging

import jax.numpy as jnp
from jax import lax

logger = logging.getLogger(__name__)


def kalman_filter_1d(
    # (n_states, )
    a0: jnp.ndarray,
    # (n_states, )
    P0: jnp.ndarray,
    # (n_steps, n_states)
    Z: jnp.ndarray,
    # (1, )
    sigma_h: jnp.ndarray | float,
    # (n_states, )
    sigma_q: jnp.ndarray | float,
    # (n_steps,)
    y: jnp.ndarray,
    logp: bool = False,
    # (n_steps, n_states) — observed latent state means; ignored where P_obs is inf
    a_obs: jnp.ndarray | None = None,
    # (n_steps, n_states) — observed latent state variances; inf = no information (pure filter)
    P_obs: jnp.ndarray | None = None,
    # (n_states, )
    positivity_idx: jnp.ndarray | None = None,
) -> tuple[float, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Linear 1D Kalman filter with optional state fusion and positivity constraint.

    Overview
    --------
    Implements a diagonal-covariance linear Kalman filter for the state-space model::

        State evolution:   a_t = a_{t-1} + η_t,   η_t ~ N(0, σ_q²I)
        Observation model: y_t = Σ_i Z_{t,i} · a_{t,i} + ε_t,   ε_t ~ N(0, σ_h²)

    Each timestep runs four stages:

    1. **State fusion** (optional) — precision-weighted merge of the carried state
       with externally disclosed latent values before measurement::

           P_t ← 1 / (1/P_t + 1/Pt_obs)
           a_t ← P_t · (a_t/P_t + at_obs/Pt_obs)

       Steps where ``P_obs = inf`` are no-ops (pure filter carry-through).

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
    _has_obs_fusion = a_obs is not None or P_obs is not None
    _at_obs = a_obs if a_obs is not None else jnp.zeros((y.shape[0], n_states))
    _Pt_obs = P_obs if P_obs is not None else jnp.full((y.shape[0], n_states), jnp.inf)

    _has_positivity = positivity_idx is not None
    _positivity_idx = positivity_idx if positivity_idx is not None else jnp.zeros(n_states, dtype=bool)
    p = len(y)

    def _transition_fn(carry, xs):
        """transition function for Kalman filter"""

        # ------ Unpack ------
        # (n_states,), (n_states,), scalar
        at, Pt, log_p = carry
        t, at_obs, Pt_obs = xs
        # scalar
        yt = y[t]
        # (n_states,)
        Zt = Z[t]

        # ------ Latent obs fusion (skipped entirely when no obs info provided) ------
        if _has_obs_fusion:
            # Bayesian Gaussian fusion of filter prior N(at, Pt) with disclosed obs
            # N(at_obs, Pt_obs).
            # Pt_obs = inf → prec_obs = 0 → no-op (pure filter carry-through).
            prec_filter = 1.0 / Pt
            prec_obs    = 1.0 / Pt_obs
            Pt          = 1.0 / (prec_filter + prec_obs)
            at          = Pt * (prec_filter * at + prec_obs * at_obs)

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
            # Soft adjustment jitter with 1e-3 to create numerically stable gap from zero.
            # Apply fusion only where enforce=True; otherwise leave (at, Pt) untouched —
            # the previous unconditional fusion silently halved Pt every step for non-enforced
            # states, collapsing the filter variance and corrupting the smoother.
            prec_filter = 1.0 / Pt
            prec_obs    = 1.0 / 1e-3
            Pt_fused    = 1.0 / (prec_filter + prec_obs)
            at_fused    = Pt_fused * (prec_filter * at + prec_obs * 1e-3)
            Pt          = jnp.where(enforce, Pt_fused, Pt)
            at          = jnp.where(enforce, at_fused, at)
            # avoid exact boundary issues
            at = jnp.where(_positivity_idx, jnp.maximum(at, 1e-6), at)

        return ((at, Pt, log_p), (at, Pt, vt, Ft, Kt))

    (_, _, log_p), (at, Pt, vt, Ft, Kt) = lax.scan(
        _transition_fn,
        (a0, P0, 0.0),
        (jnp.arange(y.shape[0]), _at_obs, _Pt_obs),
        length=y.shape[0],
    )
    return log_p, at, Pt, vt, Ft, Kt


def kalman_dk_smoother_1d(
    at: jnp.ndarray,
    Pt: jnp.ndarray,
    vt: jnp.ndarray,
    Ft: jnp.ndarray,
    Kt: jnp.ndarray,
    Z: jnp.ndarray,
    a0: jnp.ndarray,
    P0: jnp.ndarray,
    a_obs: jnp.ndarray | None = None,
    P_obs: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Durbin-Koopman disturbance smoother for the diagonal 1D state-space model.

    Runs a single backward pass over the stored filter quantities to recover
    the full-data posterior mean ``E[α_t | Y_n]`` for all ``t``.  All inputs
    except ``Z``, ``a0``, and ``P0`` are taken directly from the output of
    :func:`kalman_filter_1d`, so no quantities need to be recomputed.

    Backward recursion (Durbin & Koopman 2012 §4.4.4) extended to handle the
    optional latent-state fusion that :func:`kalman_filter_1d` applies before
    each y-update.  At every step we backward-process the y observation first,
    then the fusion pseudo-observation (Z=I, mean=a_obs, var=P_obs).
    For the fusion obs, F_fusion = Z·P·Z' + H = P_pred + P_obs (per-state
    diagonal), so::

        L_y       = 1 − K_t ⊙ Z_t
        r_after_y = Z_t ⊙ v_t / F_t + L_y ⊙ r_{t+1}
        F_fusion  = P_pred_t + Pt_obs
        L_fusion  = Pt_obs / F_fusion
        r_t       = (at_obs − a_pred_t) / F_fusion + L_fusion ⊙ r_after_y
        α̂_t      = a_pred_t + P_pred_t ⊙ r_t

    At undisclosed elements (``P_obs = inf``) the fusion contribution is
    0 and ``L_fusion = 1``, recovering the standard D&K formula.

    Because :func:`kalman_filter_1d` stores ``at[t]`` as the filtered posterior
    and ``Pt[t]`` as ``P_{t|t} + σ_q²`` (the predicted variance for the *next*
    step), the D&K predicted quantities are recovered by shifting::

        a_pred = [a0, at[0], at[1], ..., at[T-2]]
        P_pred = [P0, Pt[0], Pt[1], ..., Pt[T-2]]

    Parameters
    ----------
    at : jnp.ndarray, shape (T, n_states)
        Filtered state means from :func:`kalman_filter_1d`.
    Pt : jnp.ndarray, shape (T, n_states)
        Filtered state variances from :func:`kalman_filter_1d`.
        Each ``Pt[t]`` stores ``P_{t|t} + σ_q²``, not the pure posterior.
    vt : jnp.ndarray, shape (T, 1)
        Innovations from :func:`kalman_filter_1d`.
    Ft : jnp.ndarray, shape (T, 1)
        Innovation variances from :func:`kalman_filter_1d`.
    Kt : jnp.ndarray, shape (T, n_states)
        Kalman gains from :func:`kalman_filter_1d`.
    Z : jnp.ndarray, shape (T, n_states)
        Design / measurement matrix (same as passed to the filter).
    a0 : jnp.ndarray, shape (n_states,)
        Initial state mean (same as passed to the filter).
    P0 : jnp.ndarray, shape (n_states,)
        Initial state variance (same as passed to the filter).
    a_obs : jnp.ndarray | None, optional, shape (T, n_states)
        Disclosed latent state means passed to the filter.  Use ``None`` when
        no fusion was applied.
    P_obs : jnp.ndarray | None, optional, shape (T, n_states)
        Disclosed latent state variances passed to the filter.  Use ``jnp.inf``
        at undisclosed timesteps / states; ``None`` for no fusion at all.

    Returns
    -------
    at_smooth : jnp.ndarray, shape (T, n_states)
        Smoothed state means ``E[α_t | Y_n]``.
    """
    T, n_states = at.shape

    # D&K predicted a_t, P_t: shift filter output by one and prepend initial conditions.
    # These are also the *pre-fusion* values, since fusion happens at the start of
    # each filter step (before any update).
    a_pred = jnp.concatenate([a0[None], at[:-1]], axis=0)
    P_pred = jnp.concatenate([P0[None], Pt[:-1]], axis=0)

    # When no fusion was applied, default to inf variance (zero precision)
    # so the fusion branch is a no-op.
    if a_obs is None or P_obs is None:
        a_obs = jnp.zeros((T, n_states))
        P_obs = jnp.full((T, n_states), jnp.inf)

    def _r_step(r_next: jnp.ndarray, t: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Single backward step: y-update first, then fusion pseudo-observation."""
        # y observation contribution
        Lt_y = 1.0 - Kt[t] * Z[t]
        r_after_y = Z[t] * vt[t] / Ft[t] + Lt_y * r_next

        # Fusion pseudo-observation: Z_fusion = I (per-state diagonal), so
        # F_fusion = P_pred + P_obs and the contribution is
        # (a_obs - a_pred) / F_fusion.  Mask out elements with P_obs = inf
        # (no disclosure) — there the contribution is 0 and L_fusion = 1.
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
