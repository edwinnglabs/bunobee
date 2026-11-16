"""Transition matrix and RTS smoother of the full-covariance filter, against a dense reference."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bunobee.models.ssp.kalman_1d_st import kalman_filter_1d_st, kalman_rts_smoother_1d_st

jax.config.update("jax_enable_x64", True)

N_STEPS = 9
N_SERIES = 2


def _level_plus_ar1(phi: float = 0.7) -> dict[str, jnp.ndarray]:
    """A slow random-walk level plus a fast AR(1) deviation, seen by two series."""
    rng = np.random.default_rng(169)
    Z = np.stack([np.ones((N_STEPS, N_SERIES)), rng.uniform(0.3, 1.2, (N_STEPS, N_SERIES))], axis=-1)
    return dict(
        a0=jnp.array([0.4, 0.0]),
        P0=jnp.array([[0.5, 0.05], [0.05, 0.3]]),
        Z=jnp.asarray(Z),
        sigma_h=jnp.array([0.2, 0.3]),
        sigma_q=jnp.array([0.05, 0.25]),
        y=jnp.asarray(rng.normal(0.4, 0.4, (N_STEPS, N_SERIES))),
        T=jnp.array([[1.0, 0.0], [0.0, phi]]),
    )


def _dense_reference(
    a0: jnp.ndarray,
    P0: jnp.ndarray,
    Z: jnp.ndarray,
    sigma_h: jnp.ndarray,
    sigma_q: jnp.ndarray,
    y: jnp.ndarray,
    T: jnp.ndarray,
    mask: jnp.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact Gaussian log-likelihood and per-step conditional state moments of the stacked model.

    The states stack as ``x = m + A ξ`` with ``a_t = T a_{t-1} + η_t``, ``ξ = (a_0 − a0, η_1, ...)``
    and ``Cov(ξ) = blockdiag(P0, Q, ..., Q)``; the observed entries of ``y`` are ``H x + ε``.

    Returns
    -------
    log_p : float
        ``log N(y_obs; H m, H Σ Hᵀ + R)``.
    mean : np.ndarray, shape (n_steps, n_states)
        ``E[a_t | y_obs]``.
    cov : np.ndarray, shape (n_steps, n_states, n_states)
        ``Cov[a_t | y_obs]``.
    """
    a0, P0, Z, y, T = (np.asarray(v, dtype=float) for v in (a0, P0, Z, y, T))
    n_steps, n_series, n_states = Z.shape
    Q = np.diag(np.square(np.broadcast_to(np.asarray(sigma_q), (n_states,))))
    R_diag = np.square(np.broadcast_to(np.asarray(sigma_h), (n_series,)))

    powers = [np.eye(n_states)]
    for _ in range(n_steps):
        powers.append(T @ powers[-1])
    m = np.concatenate([powers[t] @ a0 for t in range(n_steps)])
    A = np.zeros((n_steps * n_states, n_steps * n_states))
    for t in range(n_steps):
        for s in range(t + 1):
            A[t * n_states : (t + 1) * n_states, s * n_states : (s + 1) * n_states] = powers[t - s]
    cov_xi = np.kron(np.eye(n_steps), Q)
    cov_xi[:n_states, :n_states] = P0
    S = A @ cov_xi @ A.T

    H = np.zeros((n_steps * n_series, n_steps * n_states))
    for t in range(n_steps):
        H[t * n_series : (t + 1) * n_series, t * n_states : (t + 1) * n_states] = Z[t]
    observed = np.ones(n_steps * n_series, dtype=bool) if mask is None else np.asarray(mask).reshape(-1)
    H = H[observed]
    y_obs = y.reshape(-1)[observed]
    R = np.diag(np.tile(R_diag, n_steps)[observed])

    F = H @ S @ H.T + R
    resid = y_obs - H @ m
    _, log_det = np.linalg.slogdet(F)
    log_p = -0.5 * (y_obs.size * np.log(2.0 * np.pi) + log_det + resid @ np.linalg.solve(F, resid))

    gain = S @ H.T @ np.linalg.inv(F)
    mean = (m + gain @ resid).reshape(n_steps, n_states)
    cov_full = S - gain @ H @ S
    blocks = [slice(t * n_states, (t + 1) * n_states) for t in range(n_steps)]
    cov = np.stack([cov_full[b, b] for b in blocks])
    return float(log_p), mean, cov


def test_identity_transition_matches_default() -> None:
    """``T = I`` reproduces the random-walk filter, fusion and positivity included."""
    case = _level_plus_ar1()
    del case["T"]
    a_obs = jnp.zeros((N_STEPS, 2)).at[3, 0].set(0.5)
    P_obs = jnp.full((N_STEPS, 2), jnp.inf).at[3, 0].set(0.02)
    kwargs = dict(case, logp=True, a_obs=a_obs, P_obs=P_obs, positivity=jnp.array([True, False]))

    default = kalman_filter_1d_st(**kwargs)
    identity = kalman_filter_1d_st(**kwargs, T=jnp.eye(2))

    for got, want in zip(identity, default):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_ar1_loglik_matches_dense_reference() -> None:
    """With an AR(1) state the filter's log-likelihood is the exact stacked Gaussian one."""
    case = _level_plus_ar1()
    mask = jnp.ones((N_STEPS, N_SERIES), dtype=bool).at[2, 1].set(False).at[5].set(False)

    log_p, *_ = kalman_filter_1d_st(**case, logp=True)
    log_p_masked, *_ = kalman_filter_1d_st(**case, logp=True, mask=mask)

    np.testing.assert_allclose(log_p, _dense_reference(**case)[0], rtol=1e-10)
    np.testing.assert_allclose(log_p_masked, _dense_reference(**case, mask=mask)[0], rtol=1e-10)


@pytest.mark.parametrize("masked", [False, True])
def test_smoother_matches_dense_conditional_moments(masked: bool) -> None:
    """The RTS means and full covariances are the stacked model's conditional moments."""
    case = _level_plus_ar1()
    mask = jnp.ones((N_STEPS, N_SERIES), dtype=bool).at[2, 1].set(False).at[5].set(False) if masked else None

    at_smooth, Pt_smooth = kalman_rts_smoother_1d_st(**case, mask=mask)
    _, mean, cov = _dense_reference(**case, mask=mask)

    assert at_smooth.shape == (N_STEPS, 2)
    assert Pt_smooth.shape == (N_STEPS, 2, 2)
    np.testing.assert_allclose(at_smooth, mean, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(Pt_smooth, cov, rtol=1e-9, atol=1e-10)


def test_smoother_identity_transition_matches_default() -> None:
    """``T = None`` in the smoother is the identity transition."""
    case = _level_plus_ar1()
    del case["T"]

    default = kalman_rts_smoother_1d_st(**case)
    identity = kalman_rts_smoother_1d_st(**case, T=jnp.eye(2))
    _, mean, cov = _dense_reference(**case, T=jnp.eye(2))

    for got, want in zip(identity, default):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(default[0], mean, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(default[1], cov, rtol=1e-9, atol=1e-10)


def test_loglik_gradient_in_transition_matches_finite_differences() -> None:
    """``d log_p / d T`` is finite, and matches central finite differences of the filter under jit."""
    case = _level_plus_ar1()
    T0 = case.pop("T")
    # Perturb the off-diagonal entries so every gradient entry is exercised away from zero.
    T0 = T0.at[0, 1].set(0.1).at[1, 0].set(-0.05)

    @jax.jit
    def loglik(T: jnp.ndarray) -> jnp.ndarray:
        return kalman_filter_1d_st(**case, logp=True, T=T)[0]

    grad = jax.grad(loglik)(T0)
    assert bool(jnp.all(jnp.isfinite(grad)))

    eps = 1e-6
    fd = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            step = jnp.zeros((2, 2)).at[i, j].set(eps)
            fd[i, j] = (loglik(T0 + step) - loglik(T0 - step)) / (2.0 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-6)


def test_wrong_transition_shape_raises() -> None:
    """A ``T`` that is not ``(n_states, n_states)`` is rejected up front."""
    case = _level_plus_ar1()
    case["T"] = jnp.eye(3)
    with pytest.raises(ValueError, match="T must have shape"):
        kalman_filter_1d_st(**case)
    with pytest.raises(ValueError, match="T must have shape"):
        kalman_rts_smoother_1d_st(**case)
