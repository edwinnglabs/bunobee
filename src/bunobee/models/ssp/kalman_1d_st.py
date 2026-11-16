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
    positivity: jnp.ndarray | None = None,
    mask: jnp.ndarray | None = None,
    T: jnp.ndarray | None = None,
) -> tuple[float, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Multi-series Kalman filter with shared latent state and full state covariance.

    Overview
    --------
    Implements a full-covariance Kalman filter for n_series observed series that
    share a common latent state with independent observation noise::

        State evolution:    a_t = T a_{t-1} + η_t,              η_t ~ N(0, diag(σ_q²))
        Observation model:  y_{t,j} = Z_{t,j} · a_t + ε_{t,j},  ε_{t,j} ~ N(0, σ_{h,j}²)

    The transition matrix T defaults to the identity, which makes every state a random
    walk. A non-identity T lets a state decay, e.g. an AR(1) deviation
    ``u_t = φ·u_{t-1} + ζ_t`` with ``T[i, i] = φ``.

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

    4. **Update and predict**::

           a_t ← T (a_t + K_t v_t)
           P_t ← T (P_t − K_t Z_t P_t) Tᵀ + diag(σ_q²)    (symmetrized)

       With ``T=None`` the products with T are skipped, not multiplied by an identity.

    Missing observations
    --------------------
    Passing ``mask`` turns steps 2–4 into a **predict-only** step wherever an entry is
    missing. Row j of Z_t is zeroed for a missing series, so ``v_{t,j} = 0``, column j
    of K_t is zero, and the missing series contributes nothing to either ``a_t`` or
    ``P_t``. When a whole row is missing the step reduces exactly to the pure
    prediction ``a_t = T a_{t-1}``, ``P_t = T P_{t-1} Tᵀ + diag(σ_q²)``, which is what lets a
    NaN padded ``y`` carry the state through interior gaps or past the end of the
    sample. The log-likelihood counts only the observed entries. Latent-state fusion
    and the positivity correction are state-space constraints rather than observations,
    so they still apply on a fully masked step.

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
    positivity : jnp.ndarray | None, optional, shape (n_states,)
        Boolean mask — True selects states that must remain non-negative.
        ``None`` (default) disables positivity correction for all states.
    mask : jnp.ndarray | None, optional, shape (n_steps, n_series)
        Boolean observation mask — ``True`` marks an observed entry of ``y``, ``False``
        a missing one. ``None`` (default) disables masking entirely and leaves every
        computation, ``log_p`` included, bitwise identical to the unmasked filter; a
        NaN in ``y`` then propagates into the state as before. Pass ``~jnp.isnan(y)``
        to make NaN entries predict-only. Masked entries of ``y`` are never read, so
        they may hold NaN, and rows that are entirely masked out need no meaningful
        ``Z`` either.
    T : jnp.ndarray | None, optional, shape (n_states, n_states)
        State transition matrix. ``None`` (default) is the identity, and leaves every
        output bitwise identical to the random-walk filter. The filter is traceable and
        differentiable in ``T``, so a NumPyro model can fit its entries (e.g. an AR(1)
        coefficient) through the log-likelihood.

    Returns
    -------
    log_p : float
        Accumulated Gaussian log-likelihood over the observed entries only.
        Returns 0.0 when ``logp=False``.
    at : jnp.ndarray, shape (n_steps, n_states)
        State mean after the obs update and the transition, i.e. the one-step-ahead
        prediction ``a_{t+1|t}``. With ``T=None`` it equals the filtered ``a_{t|t}``.
    Pt : jnp.ndarray, shape (n_steps, n_states, n_states)
        State covariance after the obs update, the transition and the process noise,
        i.e. ``P_{t+1|t}``. Use :func:`kalman_rts_smoother_1d_st` for the moments of the
        states over the fitted steps.
    vt : jnp.ndarray, shape (n_steps, n_series)
        Innovation (observation residual) per series at each timestep. Zero where
        ``mask`` marks an entry missing.
    Ft : jnp.ndarray, shape (n_steps, n_series, n_series)
        Full innovation covariance matrix at each timestep. A masked-out series
        keeps only its own ``sigma_h**2`` on the diagonal.
    Kt : jnp.ndarray, shape (n_steps, n_states, n_series)
        Kalman gain matrix at each timestep. Zero in the columns of masked-out series.
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
    _check_transition(T, a0.shape[0])

    log_p, (at, Pt, vt, Ft, Kt), _ = _forward_scan(
        a0=a0,
        P0=P0,
        Z=Z,
        sigma_h=sigma_h,
        sigma_q=sigma_q,
        y=y,
        logp=logp,
        a_obs=a_obs,
        P_obs=P_obs,
        positivity=positivity,
        mask=mask,
        T=T,
        keep_moments=False,
    )
    return log_p, at, Pt, vt, Ft, Kt


def kalman_rts_smoother_1d_st(
    a0: jnp.ndarray,
    P0: jnp.ndarray,
    Z: jnp.ndarray,
    sigma_h: jnp.ndarray | float,
    sigma_q: jnp.ndarray | float,
    y: jnp.ndarray,
    a_obs: jnp.ndarray | None = None,
    P_obs: jnp.ndarray | None = None,
    positivity: jnp.ndarray | None = None,
    mask: jnp.ndarray | None = None,
    T: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Rauch-Tung-Striebel smoother for the model of :func:`kalman_filter_1d_st`.

    Takes the filter's own arguments, runs its forward pass keeping the filtered moments
    ``a_{t|t}``, ``P_{t|t}`` and the one-step-ahead predictions
    ``a_{t+1|t} = T a_{t|t}``, ``P_{t+1|t} = T P_{t|t} Tᵀ + diag(σ_q²)``, then walks back::

        J_t     = P_{t|t} Tᵀ P_{t+1|t}⁻¹
        a_{t|n} = a_{t|t} + J_t (a_{t+1|n} − a_{t+1|t})
        P_{t|n} = P_{t|t} + J_t (P_{t+1|n} − P_{t+1|t}) J_tᵀ    (symmetrized)

    from ``a_{n-1|n} = a_{n-1|n-1}``. The filter's ``at`` / ``Pt`` are one-step-ahead
    predictions, from which ``a_{t|t}`` cannot be recovered for a general ``T``, which is
    why the smoother reruns the forward pass instead of taking the filter's outputs.

    Latent-state fusion (``a_obs``, ``P_obs``) and the observation ``mask`` are handled
    exactly: a fused disclosure is a Gaussian observation of the state at its step. The
    positivity pseudo-observation is applied after the pure prediction, so it is
    conditioned on like any other observation at the next step, but which states it
    fires on depends on the data and the hard floor is non-Gaussian, so smoothed moments
    through corrected steps are an approximation, as for :func:`kalman_rts_smoother_1d`.

    The smoother is traceable and differentiable in every array argument, ``T`` included.

    Parameters
    ----------
    a0, P0, Z, sigma_h, sigma_q, y, a_obs, P_obs, positivity, mask, T
        As in :func:`kalman_filter_1d_st`.

    Returns
    -------
    at_smooth : jnp.ndarray, shape (n_steps, n_states)
        Smoothed state means ``E[a_t | y_{0:n-1}]``.
    Pt_smooth : jnp.ndarray, shape (n_steps, n_states, n_states)
        Smoothed state covariances ``Cov[a_t | y_{0:n-1}]``.
    """
    logger.debug(
        "kalman_rts_smoother_1d_st inputs — a0: %s, P0: %s, Z: %s, y: %s, T: %s",
        a0.shape,
        P0.shape,
        Z.shape,
        y.shape,
        getattr(T, "shape", T),
    )
    _check_transition(T, a0.shape[0])
    _, _, (a_filt, P_filt, a_pred, P_pred) = _forward_scan(
        a0=a0,
        P0=P0,
        Z=Z,
        sigma_h=sigma_h,
        sigma_q=sigma_q,
        y=y,
        logp=False,
        a_obs=a_obs,
        P_obs=P_obs,
        positivity=positivity,
        mask=mask,
        T=T,
        keep_moments=True,
    )
    T_mat = T if T is not None else jnp.eye(a0.shape[0])

    def _step(
        carry: tuple[jnp.ndarray, jnp.ndarray],
        xs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
        """Single backward RTS step from t+1 to t."""
        a_smooth_next, P_smooth_next = carry
        # (n_states,), (n_states, n_states), (n_states,), (n_states, n_states)
        a_f, P_f, a_p, P_p = xs
        # J = P_f Tᵀ P_p⁻¹ → Jᵀ = P_p⁻¹ T P_f (both covariances symmetric)
        J = jnp.linalg.solve(P_p, T_mat @ P_f).T
        a_smooth = a_f + J @ (a_smooth_next - a_p)
        P_smooth = P_f + J @ (P_smooth_next - P_p) @ J.T
        P_smooth = 0.5 * (P_smooth + P_smooth.T)
        return (a_smooth, P_smooth), (a_smooth, P_smooth)

    # The prediction made at step t is for step t+1, so the last step's is unused.
    _, (a_head, P_head) = lax.scan(
        _step,
        (a_filt[-1], P_filt[-1]),
        (a_filt[:-1], P_filt[:-1], a_pred[:-1], P_pred[:-1]),
        reverse=True,
    )
    at_smooth = jnp.concatenate([a_head, a_filt[-1:]], axis=0)
    Pt_smooth = jnp.concatenate([P_head, P_filt[-1:]], axis=0)
    return at_smooth, Pt_smooth


def _check_transition(T: jnp.ndarray | None, n_states: int) -> None:
    """Raise if a given transition matrix is not ``(n_states, n_states)``.

    Parameters
    ----------
    T : jnp.ndarray or None
        Transition matrix, or ``None`` for the identity.
    n_states : int
        Number of states, ``a0.shape[0]``.

    Raises
    ------
    ValueError
        If ``T`` is given with a shape other than ``(n_states, n_states)``.
    """
    if T is not None and tuple(T.shape) != (n_states, n_states):
        raise ValueError(f"T must have shape ({n_states}, {n_states}), got {tuple(T.shape)}")


def _forward_scan(
    a0: jnp.ndarray,
    P0: jnp.ndarray,
    Z: jnp.ndarray,
    sigma_h: jnp.ndarray | float,
    sigma_q: jnp.ndarray | float,
    y: jnp.ndarray,
    logp: bool,
    a_obs: jnp.ndarray | None,
    P_obs: jnp.ndarray | None,
    positivity: jnp.ndarray | None,
    mask: jnp.ndarray | None,
    T: jnp.ndarray | None,
    keep_moments: bool,
) -> tuple[float, tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...] | None]:
    """Forward pass shared by :func:`kalman_filter_1d_st` and :func:`kalman_rts_smoother_1d_st`.

    Parameters
    ----------
    a0, P0, Z, sigma_h, sigma_q, y, logp, a_obs, P_obs, positivity, mask, T
        As in :func:`kalman_filter_1d_st`.
    keep_moments : bool
        Also return, per step, the filtered moments ``a_{t|t}``, ``P_{t|t}`` and the pure
        one-step-ahead prediction ``T a_{t|t}``, ``T P_{t|t} Tᵀ + diag(σ_q²)`` taken before
        the positivity correction. ``False`` leaves the traced filter untouched.

    Returns
    -------
    log_p : float
        As in :func:`kalman_filter_1d_st`.
    outputs : tuple of jnp.ndarray
        ``(at, Pt, vt, Ft, Kt)`` as returned by :func:`kalman_filter_1d_st`.
    moments : tuple of jnp.ndarray or None
        ``(a_filt, P_filt, a_pred, P_pred)`` with shapes ``(n_steps, n_states)`` and
        ``(n_steps, n_states, n_states)``, or ``None`` when ``keep_moments=False``.
    """
    n_states = a0.shape[0]
    n_series = y.shape[1]
    sigma_h_sq = jnp.broadcast_to(jnp.square(sigma_h), (n_series,))
    sigma_q_sq = jnp.broadcast_to(jnp.square(sigma_q), (n_states,))

    # default: loc=0, var=inf → zero precision → fusion is a no-op at undisclosed steps
    _has_obs_fusion = a_obs is not None or P_obs is not None
    _at_obs = a_obs if a_obs is not None else jnp.zeros((y.shape[0], n_states))
    _Pt_obs = P_obs if P_obs is not None else jnp.full((y.shape[0], n_states), jnp.inf)

    _has_positivity = positivity is not None
    _positivity = positivity if positivity is not None else jnp.zeros(n_states, dtype=bool)

    # None → no masking at all; the traced computation stays bitwise identical.
    _has_mask = mask is not None
    _mask = jnp.asarray(mask, dtype=bool) if mask is not None else jnp.ones(y.shape, dtype=bool)

    _has_T = T is not None

    def _transition_fn(
        carry: tuple[jnp.ndarray, jnp.ndarray, float],
        xs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple, tuple]:
        """Single Kalman filter step: fuse latent obs, update, transition, enforce positivity."""
        at, Pt, log_p = carry
        # (n_series,), (n_series, n_states), (n_states,), (n_states,), (n_series,)
        yt, Zt, at_obs, Pt_obs, obs_mask = xs

        # ------ Missing observations: zero out Z rows so masked series never update ------
        # A zeroed row j gives yhat_j = 0 and (with yt_j → 0) vt_j = 0, so Kt[:, j] = 0
        # and series j contributes nothing to the state or covariance update.
        if _has_mask:
            yt = jnp.where(obs_mask, yt, 0.0)
            Zt = jnp.where(obs_mask[:, None], Zt, 0.0)

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
            if _has_mask:
                # Masked series sit alone on the diagonal of Ft with variance sigma_h**2
                # and contribute vt = 0; drop their log-determinant and dimension count
                # so log_p scores the observed entries only.
                n_obs = jnp.sum(obs_mask)
                log_det_F = log_det_F - jnp.sum(jnp.where(obs_mask, 0.0, jnp.log(sigma_h_sq)))
                log_p = log_p - 0.5 * (n_obs * jnp.log(2.0 * jnp.pi) + log_det_F + mahal)
            else:
                log_p = log_p - 0.5 * (n_series * jnp.log(2.0 * jnp.pi) + log_det_F + mahal)

        # ------ Update step ------
        at = at + Kt @ vt

        if keep_moments:
            # Filtered moments a_{t|t}, P_{t|t}, kept for the RTS smoother only.
            P_filt = Pt - Kt @ Zt @ Pt
            moments = (at, 0.5 * (P_filt + P_filt.T))

        # Covariance update + transition + process noise; symmetrize to prevent numerical drift
        if _has_T:
            at = T @ at
            Pt = T @ (Pt - Kt @ Zt @ Pt) @ T.T + jnp.diag(sigma_q_sq)
        else:
            Pt = Pt - Kt @ Zt @ Pt + jnp.diag(sigma_q_sq)
        Pt = 0.5 * (Pt + Pt.T)

        if keep_moments:
            # Pure one-step-ahead prediction, before the positivity pseudo-observation.
            moments = moments + (at, Pt)

        # ------ Positivity correction ------
        if _has_positivity:
            enforce = _positivity & (at < 0)
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
            at = jnp.where(_positivity, jnp.maximum(at, 1e-6), at)

        outputs = (at, Pt, vt, Ft, Kt)
        return (at, Pt, log_p), (outputs, moments) if keep_moments else outputs

    (_, _, log_p), scanned = lax.scan(
        _transition_fn,
        (a0, P0, 0.0),
        (y, Z, _at_obs, _Pt_obs, _mask),
        length=y.shape[0],
    )
    if keep_moments:
        return log_p, scanned[0], scanned[1]
    return log_p, scanned, None
