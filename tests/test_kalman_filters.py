from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from bunobee.models.ssp.kalman_1d import kalman_filter_1d
from bunobee.models.ssp.kalman_1d_ekf import kalman_filter_1d_ekf
from bunobee.models.ssp.kalman_1d_st import kalman_filter_1d_st
from bunobee.models.ssp.kalman_1d_st_ekf import kalman_filter_1d_ekf_st

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

T = 8
N_STATES = 2

_A0 = jnp.array([0.5, 0.3])
_P0 = jnp.array([1.0, 0.5])
_SIGMA_H = 0.1
_SIGMA_Q = jnp.array([0.05, 0.03])
_Z = jnp.array(
    [
        [1.0, 0.4],
        [0.9, 0.5],
        [1.1, 0.3],
        [1.0, 0.45],
        [0.95, 0.35],
        [1.05, 0.4],
        [0.85, 0.5],
        [1.0, 0.3],
    ]
)
_Y = jnp.array([1.2, 0.9, 1.1, 1.0, 1.3, 0.8, 1.15, 0.95])


def _reference_loglik(a0, P0, Z, sigma_h, sigma_q, y):
    """Prediction-error-decomposition log-likelihood computed in numpy."""
    T, n_states = Z.shape
    sigma_h_sq = float(sigma_h) ** 2
    sigma_q_sq = np.asarray(sigma_q, dtype=float) ** 2
    at = np.asarray(a0, dtype=float).copy()
    Pt = np.asarray(P0, dtype=float).copy()
    log_p = 0.0
    for t in range(T):
        Zt = np.asarray(Z[t], dtype=float)
        yt = float(y[t])
        yhat = float(np.dot(Zt, at))
        vt = yt - yhat
        Ft = float(np.dot(Zt * Pt, Zt) + sigma_h_sq)
        log_p += -0.5 * (np.log(2.0 * np.pi) + np.log(Ft) + vt**2 / Ft)
        Kt = Pt * Zt / Ft
        at = at + Kt * vt
        Pt = Pt * (1.0 - Kt * Zt) + sigma_q_sq
    return log_p


# ---------------------------------------------------------------------------
# Test 1 — log-likelihood matches an independent reference
# ---------------------------------------------------------------------------


def test_filter_1d_loglik_matches_reference() -> None:
    """kalman_filter_1d log_p matches an independent prediction-error-decomposition loop.

    Validates v_t, F_t, and the accumulated Gaussian log-likelihood without
    relying on any other filter output.
    """
    log_p_filter, *_ = kalman_filter_1d(
        a0=_A0,
        P0=_P0,
        Z=_Z,
        sigma_h=_SIGMA_H,
        sigma_q=_SIGMA_Q,
        y=_Y,
        logp=True,
    )
    log_p_ref = _reference_loglik(_A0, _P0, _Z, _SIGMA_H, _SIGMA_Q, _Y)

    assert jnp.allclose(log_p_filter, jnp.array(log_p_ref), atol=1e-10)


# ---------------------------------------------------------------------------
# Test 2 — multi-series filter reduces to scalar filter for n_series=1
# ---------------------------------------------------------------------------


def test_st_filter_reduces_to_1d_for_single_series() -> None:
    """kalman_filter_1d_st with n_series=1 and n_states=1 reproduces kalman_filter_1d.

    Cross-validates the full-covariance multi-series filter against the scalar
    filter on a single-state, single-series problem.  With n_states=1 the full
    covariance matrix collapses to a scalar and the two filters are algebraically
    identical — off-diagonal cross-terms (which cause divergence for n_states>1)
    cannot arise.
    """
    # Single-state, single-series problem
    a0_1 = jnp.array([0.5])
    P0_1 = jnp.array([1.0])
    sigma_q_1 = jnp.array([0.05])
    Z_1d = _Z[:, :1]  # (T, 1)
    Z_st = Z_1d[:, None, :]  # (T, 1, 1)
    y_st = _Y[:, None]  # (T, 1)
    P0_st = jnp.diag(P0_1)  # (1, 1)
    sigma_h_st = jnp.array([_SIGMA_H])

    log_p_1d, at_1d, Pt_1d, vt_1d, Ft_1d, _ = kalman_filter_1d(
        a0=a0_1,
        P0=P0_1,
        Z=Z_1d,
        sigma_h=_SIGMA_H,
        sigma_q=sigma_q_1,
        y=_Y,
        logp=True,
    )
    log_p_st, at_st, Pt_st, vt_st, Ft_st, _ = kalman_filter_1d_st(
        a0=a0_1,
        P0=P0_st,
        Z=Z_st,
        sigma_h=sigma_h_st,
        sigma_q=sigma_q_1,
        y=y_st,
        logp=True,
    )

    assert jnp.allclose(at_st, at_1d, atol=1e-10), "filtered means differ"
    assert jnp.allclose(log_p_st, log_p_1d, atol=1e-10), "log-likelihoods differ"
    assert jnp.allclose(Pt_st[:, 0, 0], Pt_1d[:, 0], atol=1e-10), "filtered variances differ"
    assert jnp.allclose(vt_st[:, 0], vt_1d[:, 0], atol=1e-10), "innovations differ"
    assert jnp.allclose(Ft_st[:, 0, 0], Ft_1d[:, 0], atol=1e-10), "innovation variances differ"


# ---------------------------------------------------------------------------
# Test 3 — EKF collapses to linear filter when positivity_idx is all-False
# ---------------------------------------------------------------------------


def test_ekf_reduces_to_linear_filter() -> None:
    """kalman_filter_1d_ekf with positivity_idx=all-False matches kalman_filter_1d.

    Because the EKF carries P_{t|t} (pure posterior) while the linear filter
    carries P_{t+1|t} = P_{t|t} + sigma_q^2, exact agreement requires the
    linear filter's P0 to equal P0_ekf + sigma_q^2.
    """
    positivity_idx = jnp.zeros(N_STATES, dtype=bool)
    sigma_q_sq = jnp.square(_SIGMA_Q)
    P0_ekf = _P0
    P0_lin = _P0 + sigma_q_sq  # align initial predicted covariance

    log_p_lin, at_lin, Pt_lin, vt_lin, Ft_lin, Kt_lin = kalman_filter_1d(
        a0=_A0,
        P0=P0_lin,
        Z=_Z,
        sigma_h=_SIGMA_H,
        sigma_q=_SIGMA_Q,
        y=_Y,
        logp=True,
        positivity_idx=positivity_idx,
    )
    log_p_ekf, at_ekf, Pt_ekf, vt_ekf, Ft_ekf, Kt_ekf = kalman_filter_1d_ekf(
        a0=_A0,
        P0=P0_ekf,
        Z=_Z,
        sigma_h=_SIGMA_H,
        sigma_q=_SIGMA_Q,
        y=_Y,
        logp=True,
        positivity_idx=positivity_idx,
    )

    assert jnp.allclose(at_ekf, at_lin, atol=1e-10), "filtered means differ"
    assert jnp.allclose(Pt_ekf + sigma_q_sq, Pt_lin, atol=1e-10), "filtered variances differ"
    assert jnp.allclose(log_p_ekf, log_p_lin, atol=1e-10), "log-likelihoods differ"
    assert jnp.allclose(vt_ekf, vt_lin[:, 0], atol=1e-10), "innovations differ"
    assert jnp.allclose(Ft_ekf, Ft_lin[:, 0], atol=1e-10), "innovation variances differ"


# ---------------------------------------------------------------------------
# Test 4 — multi-series EKF collapses to multi-series linear filter
# ---------------------------------------------------------------------------


def test_st_ekf_reduces_to_st_filter() -> None:
    """kalman_filter_1d_ekf_st with positivity_idx=all-False matches kalman_filter_1d_st.

    Cross-validates the multi-series EKF against the multi-series linear filter
    using the same P0-alignment trick as in test_ekf_reduces_to_linear_filter.
    """
    n_series = 2
    positivity_idx = jnp.zeros(N_STATES, dtype=bool)
    sigma_q_sq = jnp.square(_SIGMA_Q)

    # Build a 2-series problem by stacking two copies of _Z and _Y
    Z_st = jnp.stack([_Z, 0.8 * _Z], axis=1)  # (T, 2, n_states)
    y_st = jnp.stack([_Y, 0.9 * _Y], axis=1)  # (T, 2)
    sigma_h_st = jnp.array([_SIGMA_H, 0.12])

    P0_ekf = jnp.diag(_P0)
    P0_lin = jnp.diag(_P0 + sigma_q_sq)  # align initial predicted covariance

    log_p_st, at_st, Pt_st, vt_st, Ft_st, _ = kalman_filter_1d_st(
        a0=_A0,
        P0=P0_lin,
        Z=Z_st,
        sigma_h=sigma_h_st,
        sigma_q=_SIGMA_Q,
        y=y_st,
        logp=True,
    )
    log_p_ekf_st, at_ekf_st, Pt_ekf_st, vt_ekf_st, Ft_ekf_st, _ = kalman_filter_1d_ekf_st(
        a0=_A0,
        P0=P0_ekf,
        Z=Z_st,
        sigma_h=sigma_h_st,
        sigma_q=_SIGMA_Q,
        y=y_st,
        logp=True,
        positivity_idx=positivity_idx,
    )

    assert jnp.allclose(at_ekf_st, at_st, atol=1e-10), "filtered means differ"
    assert jnp.allclose(Pt_ekf_st + jnp.diag(sigma_q_sq), Pt_st, atol=1e-10), "filtered covariances differ"
    assert jnp.allclose(log_p_ekf_st, log_p_st, atol=1e-10), "log-likelihoods differ"
    assert jnp.allclose(vt_ekf_st, vt_st, atol=1e-10), "innovations differ"
    assert jnp.allclose(Ft_ekf_st, Ft_st, atol=1e-10), "innovation covariances differ"


# ---------------------------------------------------------------------------
# Test 5 — positivity correction keeps flagged states non-negative
# ---------------------------------------------------------------------------


def test_positivity_keeps_states_nonnegative() -> None:
    """Flagged filtered states remain >= 1e-6 every step for both 1d and st filters.

    Uses data that would drive the state negative without the positivity floor.
    """
    # Large negative observations drive the first state negative without positivity
    y_neg = jnp.full(T, -5.0)
    positivity_idx = jnp.array([True, False])

    _, at_1d, *_ = kalman_filter_1d(
        a0=_A0,
        P0=_P0,
        Z=_Z,
        sigma_h=_SIGMA_H,
        sigma_q=_SIGMA_Q,
        y=y_neg,
        positivity_idx=positivity_idx,
    )

    assert jnp.all(at_1d[:, 0] >= 1e-6), "1d filter violated positivity floor"

    # Confirm the second (unflagged) state does go negative — sanity check
    _, at_no_pos, *_ = kalman_filter_1d(
        a0=_A0,
        P0=_P0,
        Z=_Z,
        sigma_h=_SIGMA_H,
        sigma_q=_SIGMA_Q,
        y=y_neg,
    )
    assert jnp.any(at_no_pos < 0), "positivity was not needed — test fixture invalid"

    # Multi-series filter
    Z_st = _Z[:, None, :]
    y_st = y_neg[:, None]
    P0_st = jnp.diag(_P0)

    _, at_st, *_ = kalman_filter_1d_st(
        a0=_A0,
        P0=P0_st,
        Z=Z_st,
        sigma_h=jnp.array([_SIGMA_H]),
        sigma_q=_SIGMA_Q,
        y=y_st,
        positivity_idx=positivity_idx,
    )

    assert jnp.all(at_st[:, 0] >= 1e-6), "st filter violated positivity floor"


# ---------------------------------------------------------------------------
# Test 6 — state fusion pulls toward disclosure; P_obs=inf is a no-op
# ---------------------------------------------------------------------------


def test_state_fusion_pulls_toward_disclosure() -> None:
    """Disclosed latent state with tiny P_obs pulls the filtered mean to ~the disclosed value.

    Also verifies that P_obs=inf at all steps is equivalent to running without fusion.
    """
    # Disclose state 0 at step 3 with very tight variance
    a_obs = jnp.zeros((T, N_STATES))
    P_obs = jnp.full((T, N_STATES), jnp.inf)
    disclosed_value = 0.75
    a_obs = a_obs.at[3, 0].set(disclosed_value)
    P_obs = P_obs.at[3, 0].set(1e-8)

    _, at_fused, *_ = kalman_filter_1d(
        a0=_A0,
        P0=_P0,
        Z=_Z,
        sigma_h=_SIGMA_H,
        sigma_q=_SIGMA_Q,
        y=_Y,
        a_obs=a_obs,
        P_obs=P_obs,
    )

    # With such tight variance, filtered state should be very close to disclosed value
    assert (
        abs(float(at_fused[3, 0]) - disclosed_value) < 1e-4
    ), f"fusion did not pull state toward disclosure: got {at_fused[3, 0]:.6f}"

    # P_obs=inf everywhere is a no-op — result must match the unfused filter
    P_obs_inf = jnp.full((T, N_STATES), jnp.inf)
    _, at_noop, *_ = kalman_filter_1d(
        a0=_A0,
        P0=_P0,
        Z=_Z,
        sigma_h=_SIGMA_H,
        sigma_q=_SIGMA_Q,
        y=_Y,
        a_obs=jnp.zeros((T, N_STATES)),
        P_obs=P_obs_inf,
    )
    _, at_no_fusion, *_ = kalman_filter_1d(
        a0=_A0,
        P0=_P0,
        Z=_Z,
        sigma_h=_SIGMA_H,
        sigma_q=_SIGMA_Q,
        y=_Y,
    )

    assert jnp.allclose(at_noop, at_no_fusion, atol=1e-10), "P_obs=inf is not a no-op"


# ---------------------------------------------------------------------------
# Test 7 — output shapes and finiteness for all four filters
# ---------------------------------------------------------------------------


def test_filter_output_shapes_and_finiteness() -> None:
    """All four filters return documented shapes and produce finite at/Pt/vt/Ft/Kt."""
    n_series = 2
    Z_st = jnp.stack([_Z, 0.8 * _Z], axis=1)  # (T, 2, N_STATES)
    y_st = jnp.stack([_Y, 0.9 * _Y], axis=1)  # (T, 2)
    sigma_h_st = jnp.array([_SIGMA_H, 0.12])
    P0_full = jnp.diag(_P0)

    # --- kalman_filter_1d ---
    log_p, at, Pt, vt, Ft, Kt = kalman_filter_1d(
        a0=_A0, P0=_P0, Z=_Z, sigma_h=_SIGMA_H, sigma_q=_SIGMA_Q, y=_Y, logp=True
    )
    assert at.shape == (T, N_STATES)
    assert Pt.shape == (T, N_STATES)
    assert vt.shape == (T, 1)
    assert Ft.shape == (T, 1)
    assert Kt.shape == (T, N_STATES)
    assert jnp.isfinite(log_p)
    assert jnp.all(jnp.isfinite(at))
    assert jnp.all(jnp.isfinite(Pt))
    assert jnp.all(jnp.isfinite(vt))
    assert jnp.all(jnp.isfinite(Ft))
    assert jnp.all(jnp.isfinite(Kt))

    # --- kalman_filter_1d_st ---
    log_p, at, Pt, vt, Ft, Kt = kalman_filter_1d_st(
        a0=_A0, P0=P0_full, Z=Z_st, sigma_h=sigma_h_st, sigma_q=_SIGMA_Q, y=y_st, logp=True
    )
    assert at.shape == (T, N_STATES)
    assert Pt.shape == (T, N_STATES, N_STATES)
    assert vt.shape == (T, n_series)
    assert Ft.shape == (T, n_series, n_series)
    assert Kt.shape == (T, N_STATES, n_series)
    assert jnp.isfinite(log_p)
    assert jnp.all(jnp.isfinite(at))
    assert jnp.all(jnp.isfinite(Pt))
    assert jnp.all(jnp.isfinite(vt))
    assert jnp.all(jnp.isfinite(Ft))
    assert jnp.all(jnp.isfinite(Kt))

    # --- kalman_filter_1d_ekf ---
    log_p, at, Pt, vt, Ft, Kt = kalman_filter_1d_ekf(
        a0=_A0, P0=_P0, Z=_Z, sigma_h=_SIGMA_H, sigma_q=_SIGMA_Q, y=_Y, logp=True
    )
    assert at.shape == (T, N_STATES)
    assert Pt.shape == (T, N_STATES)
    assert vt.shape == (T,)
    assert Ft.shape == (T,)
    assert Kt.shape == (T, N_STATES)
    assert jnp.isfinite(log_p)
    assert jnp.all(jnp.isfinite(at))
    assert jnp.all(jnp.isfinite(Pt))
    assert jnp.all(jnp.isfinite(vt))
    assert jnp.all(jnp.isfinite(Ft))
    assert jnp.all(jnp.isfinite(Kt))

    # --- kalman_filter_1d_ekf_st ---
    log_p, at, Pt, vt, Ft, Kt = kalman_filter_1d_ekf_st(
        a0=_A0, P0=P0_full, Z=Z_st, sigma_h=sigma_h_st, sigma_q=_SIGMA_Q, y=y_st, logp=True
    )
    assert at.shape == (T, N_STATES)
    assert Pt.shape == (T, N_STATES, N_STATES)
    assert vt.shape == (T, n_series)
    assert Ft.shape == (T, n_series, n_series)
    assert Kt.shape == (T, N_STATES, n_series)
    assert jnp.isfinite(log_p)
    assert jnp.all(jnp.isfinite(at))
    assert jnp.all(jnp.isfinite(Pt))
    assert jnp.all(jnp.isfinite(vt))
    assert jnp.all(jnp.isfinite(Ft))
    assert jnp.all(jnp.isfinite(Kt))
