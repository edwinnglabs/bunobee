from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax, random

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
) -> Tuple[float, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.array, jnp.ndarray]:
    logger.debug("kalman_filter_1d inputs — a0: %s, P0: %s, Z: %s, y: %s, sigma_h: %s, sigma_q: %s",
                 a0.shape, P0.shape, Z.shape, y.shape,
                 getattr(sigma_h, "shape", sigma_h),
                 getattr(sigma_q, "shape", sigma_q))
    sigma_h_sq = jnp.square(sigma_h)
    sigma_q_sq = jnp.square(sigma_q)
    p = len(y)
    n_states = a0.shape[0]

    # default: loc=0, var=inf → zero precision → fusion is a no-op at undisclosed steps
    _a_obs_loc = a_obs_loc if a_obs_loc is not None else jnp.zeros((y.shape[0], n_states))
    _a_obs_var = a_obs_var if a_obs_var is not None else jnp.full((y.shape[0], n_states), jnp.inf)

    def _transition_fn(carry, xs):
        """transition function for Kalman filter"""
        # (n_states,), (n_states,), scalar
        at, Pt, log_p = carry
        t, at_obs_loc_t, at_obs_var_t = xs

        # Bayesian Gaussian fusion of filter prior N(at, Pt) with disclosed obs N(at_obs_loc_t, at_obs_var_t).
        # at_obs_var_t = inf → prec_obs = 0 → no-op (pure filter carry-through).
        prec_filter = 1.0 / Pt
        prec_obs    = 1.0 / at_obs_var_t
        Pt          = 1.0 / (prec_filter + prec_obs)
        at          = Pt * (prec_filter * at + prec_obs * at_obs_loc_t)

        # scalar
        yt = y[t]
        # (n_states,)
        Zt = Z[t]

        # (n_states,) * (n_states,) -> sum -> (1,)
        yhat = jnp.sum(Zt * at, -1, keepdims=True)

        # scalar - (1,) -> (1,)
        vt = yt - yhat

        # (n_states,) * (n_states,) -> sum -> (1,)  +  scalar -> (1,)
        Ft = jnp.sum(Pt * jnp.square(Zt), -1, keepdims=True) + sigma_h_sq

        # (n_states,) * (n_states,) / (1,) -> (n_states,)
        Kt = Pt * Zt / Ft

        # scalar + scalar -> scalar
        if logp:
            log_p += -0.5 * (p * jnp.log(2 * jnp.pi) + jnp.sum(jnp.log(Ft) + jnp.square(vt) / Ft))

        # (n_states,) + (n_states,) * (1,) -> (n_states,)
        at = at + Kt * vt

        # (n_states,) * (scalar - (n_states,)) + (n_states,) -> (n_states,)
        Pt = Pt * (1 -  Kt) + sigma_q_sq

        return ((at, Pt, log_p), (at, Pt, vt, Ft, Kt))

    (_, _, log_p), (at, Pt, vt, Ft, Kt) = lax.scan(
        _transition_fn,
        (a0, P0, 0.0),
        (jnp.arange(y.shape[0]), _a_obs_loc, _a_obs_var),
        length=y.shape[0],
    )
    return log_p, at, Pt, vt, Ft, Kt


def kalman_smoother(
    a0: jnp.ndarray,
    P0: jnp.ndarray,
    sigma_h: Union[jnp.ndarray, float],
    sigma_q: Union[jnp.ndarray, float],
    obs: jnp.ndarray,
    rng_key: jnp.ndarray,
) -> jnp.ndarray:

    # step 1. simulate obs based on non-optimized alpha
    # step 2. run kalman smoother backward
    # step 3. run kalman filter forward
    # step 4. derive smoothed alpha for full dist. of alpha

    n_steps = obs.shape[0]
    sigma_q_sq = jnp.square(sigma_q)

    rng_sub_keys = random.split(rng_key, 3)
    # note that this alpha is ~ N(0, P0) not N(a0, P0)
    init_sim_alpha = dist.Normal(0.0, 1.0).sample(
        rng_sub_keys[0],
    ) * jnp.sqrt(P0)

    obs_noise = dist.Normal(0.0, 1.0).sample(
        rng_sub_keys[1],
        sample_shape=(n_steps, ),
    ) * sigma_h

    innov = dist.Normal(0.0, 1.0).sample(
        rng_sub_keys[2],
        sample_shape=(n_steps, ),
    ) * sigma_q

    def sim_obs_fn(carry, t):
        alpha_t = carry

        # observations simulate step
        y_t = alpha_t + obs_noise[t]
        alpha_t = alpha_t + innov[t]
        return alpha_t, (alpha_t, y_t)

    _, (alpha_plus, obs_plus) = lax.scan(
        sim_obs_fn,
        init=init_sim_alpha,
        xs=jnp.arange(n_steps),
        length=n_steps,
    )

    # output from scan has an additional shape in last dim
    obs_diff = obs - jnp.squeeze(obs_plus, -1)

    _, _, _, v, F, K = kalman_filter(
        a0=a0,
        P0=P0,
        sigma_h=sigma_h,
        sigma_q=sigma_q,
        y=obs_diff,
    )

    def kalman_smoother_backward_fn(carry, t):
        rt = carry
        Lt = 1 - K[t]
        rt = 1 / F[t] * v[t] + Lt * rt

        return rt, rt

    rT = jnp.zeros((1))
    _, r = lax.scan(
        kalman_smoother_backward_fn,
        init=rT,
        xs=jnp.arange(n_steps - 1, -1, -1),
        length=n_steps,
    )

    r = jnp.squeeze(r, -1)
    # flip on steps
    r = jnp.flip(r, -1)
    # include first step
    r = jnp.concatenate([r, rT], -1)

    def kalman_smoother_forward_fn(carry, t):
        alpha_t = carry
        # correction of mean vector given lookahead data points
        # note that r vector is appended with initial value;
        # so t + 1 means t in the actual maths
        alpha_t = alpha_t + sigma_q_sq * r[t + 1]

        return alpha_t, alpha_t

    alpha_0 = a0 + P0 * r[0]

    _, smoothed_obs_diff_alpha = lax.scan(
        kalman_smoother_forward_fn,
        init=alpha_0,
        xs=jnp.arange(n_steps),
        length=n_steps,
    )   

    smoothed_alpha = alpha_plus + smoothed_obs_diff_alpha
    # (n_steps, )
    return jnp.squeeze(smoothed_alpha, -1)


def simulate_forecast(
    # (n_sample, )
    a0: jnp.ndarray,
    # (n_sample, )
    sigma_h: jnp.ndarray,
    # (n_sample, )
    sigma_q: jnp.ndarray,
    # forecast horizon
    n_steps: int,
    rng_key: jnp.ndarray,
):
    rng_sub_keys = random.split(rng_key, 2)
    n_samples = sigma_h.shape[0]

    # move steps to first dim to facilitate broadcast
    # (n_steps, n_samples)
    obs_noise = (
        dist.Normal(loc=0.0, scale=1.0).sample(
            rng_sub_keys[0], sample_shape=(n_steps, n_samples)
        ) * sigma_h
    )
    innov = (
        dist.Normal(loc=0.0, scale=1.0).sample(
            rng_sub_keys[1], sample_shape=(n_steps, n_samples)
        ) * sigma_q
    )

    def sim_transition_fn(carry, t):
        at = carry
        at = at + innov[t]

        return at, at

    _, states = lax.scan(
        sim_transition_fn,
        a0,
        jnp.arange(n_steps),
        length=n_steps,
    )
    # (n_steps, n_samples) -> (n_samples, n_steps)
    res = states + obs_noise
    res = jnp.swapaxes(res, -1, -2)
    return res


def simulate_forecast(
    # (n_samples,) or (n_samples, n_state) when X_future is provided
    a0: jnp.ndarray,
    # (n_samples,)
    sigma_h: jnp.ndarray,
    # (n_samples,) or (n_samples, n_state) when X_future is provided
    sigma_q: jnp.ndarray,
    # forecast horizon
    n_steps: int,
    rng_key: jnp.ndarray,
    # (n_steps, n_regressors) — future regressor values for time-varying coefficient forecast
    X_future: Optional[jnp.ndarray] = None,
):
    rng_sub_keys = random.split(rng_key, 2)
    n_samples = sigma_h.shape[0]
    has_X = X_future is not None

    if has_X:
        n_state = a0.shape[-1]

        # (n_steps, n_samples, n_state)
        innov = (
            dist.Normal(loc=0.0, scale=1.0).sample(
                rng_sub_keys[1], sample_shape=(n_steps, n_samples, n_state)
            ) * sigma_q
        )
        # (n_steps, n_samples)
        obs_noise = (
            dist.Normal(loc=0.0, scale=1.0).sample(
                rng_sub_keys[0], sample_shape=(n_steps, n_samples)
            ) * sigma_h
        )

        def sim_transition_fn(carry, t):
            at = carry                                                       # (n_samples, n_state)
            at = at + innov[t]
            return at, at

        _, states = lax.scan(
            sim_transition_fn, a0, jnp.arange(n_steps), length=n_steps,
        )
        # states: (n_steps, n_samples, n_state)
        # Z_future: (n_steps, n_state) = [1, X_future]
        Z_future = jnp.concatenate(
            [jnp.ones((n_steps, 1)), X_future], axis=1
        )
        # obs: (n_steps, n_samples) via einsum, then swap to (n_samples, n_steps)
        res = jnp.einsum("tsk,tk->ts", states, Z_future) + obs_noise
        return jnp.swapaxes(res, -1, -2)

    else:
        # move steps to first dim to facilitate broadcast
        # (n_steps, n_samples)
        obs_noise = (
            dist.Normal(loc=0.0, scale=1.0).sample(
                rng_sub_keys[0], sample_shape=(n_steps, n_samples)
            ) * sigma_h
        )
        innov = (
            dist.Normal(loc=0.0, scale=1.0).sample(
                rng_sub_keys[1], sample_shape=(n_steps, n_samples)
            ) * sigma_q
        )

        def sim_transition_fn(carry, t):
            at = carry
            at = at + innov[t]

            return at, at

        _, states = lax.scan(
            sim_transition_fn,
            a0,
            jnp.arange(n_steps),
            length=n_steps,
        )
        # (n_steps, n_samples) -> (n_samples, n_steps)
        res = states + obs_noise
        res = jnp.swapaxes(res, -1, -2)
        return res
