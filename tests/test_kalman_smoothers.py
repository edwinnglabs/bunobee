from __future__ import annotations

import jax
import jax.numpy as jnp

from bunobee.models.ssp.kalman_1d import (
    kalman_dk_smoother_1d,
    kalman_filter_1d,
    kalman_rts_smoother_1d,
)
from bunobee.models.ssp.kalman_1d_ekf import (
    kalman_dk_smoother_1d_ekf,
    kalman_filter_1d_ekf,
    kalman_rts_smoother_1d_ekf,
)

jax.config.update("jax_enable_x64", True)


def _run_filter_case() -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Build a deterministic filter run that exercises disclosure and positivity flags."""
    a0 = jnp.array([0.1, 0.2, 0.15])
    P0 = jnp.array([0.8, 0.3, 0.25])
    sigma_h = jnp.array(0.05)
    sigma_q = jnp.array([0.02, 0.01, 0.015])
    positivity_idx = jnp.array([False, True, True])

    Z = jnp.array(
        [
            [1.0, 0.4, 0.2],
            [1.0, 0.3, 0.25],
            [1.0, 0.5, 0.1],
            [1.0, 0.2, 0.35],
            [1.0, 0.45, 0.15],
            [1.0, 0.25, 0.3],
        ]
    )
    y = jnp.array([0.24, 0.23, 0.27, 0.21, 0.265, 0.225])

    a_obs = jnp.zeros_like(Z)
    P_obs = jnp.full_like(Z, jnp.inf)
    a_obs = a_obs.at[2, 1].set(0.22)
    P_obs = P_obs.at[2, 1].set(0.01)
    a_obs = a_obs.at[4, 2].set(0.18)
    P_obs = P_obs.at[4, 2].set(0.02)

    _, at, Pt, vt, Ft, Kt = kalman_filter_1d(
        a0=a0,
        P0=P0,
        Z=Z,
        sigma_h=sigma_h,
        sigma_q=sigma_q,
        y=y,
        logp=True,
        a_obs=a_obs,
        P_obs=P_obs,
        positivity_idx=positivity_idx,
    )
    return at, Pt, vt, Ft, Kt, Z, a0, P0, a_obs, P_obs, sigma_q


def test_rts_matches_dk_smoothed_mean() -> None:
    """RTS and D&K should agree on the smoothed state mean for the linear model."""
    at, Pt, vt, Ft, Kt, Z, a0, P0, a_obs, P_obs, sigma_q = _run_filter_case()

    at_dk = kalman_dk_smoother_1d(
        at=at,
        Pt=Pt,
        vt=vt,
        Ft=Ft,
        Kt=Kt,
        Z=Z,
        a0=a0,
        P0=P0,
        a_obs=a_obs,
        P_obs=P_obs,
    )
    at_rts, _ = kalman_rts_smoother_1d(
        at=at,
        Pt=Pt,
        sigma_q=sigma_q,
    )

    assert jnp.allclose(at_rts, at_dk, atol=1e-8)


def test_rts_smoothed_variance_respects_filter_posterior_bound() -> None:
    """RTS variance must end at the filter posterior and not exceed it earlier."""
    at, Pt, _, _, _, _, _, _, _, _, sigma_q = _run_filter_case()

    _, Pt_smooth = kalman_rts_smoother_1d(
        at=at,
        Pt=Pt,
        sigma_q=sigma_q,
    )
    P_filt = Pt - jnp.square(sigma_q)

    assert jnp.allclose(Pt_smooth[-1], P_filt[-1], atol=1e-10)
    assert jnp.all(Pt_smooth[:-1] <= P_filt[:-1] + 1e-10)


def test_ekf_rts_matches_dk_in_fully_linear_limit() -> None:
    """The EKF RTS smoother should collapse to D&K when all states are linear."""
    a0 = jnp.array([0.1, 0.2, 0.15])
    P0 = jnp.array([0.8, 0.3, 0.25])
    sigma_h = jnp.array(0.05)
    sigma_q = jnp.array([0.02, 0.01, 0.015])
    positivity_idx = jnp.zeros(3, dtype=bool)

    Z = jnp.array(
        [
            [1.0, 0.4, 0.2],
            [1.0, 0.3, 0.25],
            [1.0, 0.5, 0.1],
            [1.0, 0.2, 0.35],
            [1.0, 0.45, 0.15],
            [1.0, 0.25, 0.3],
        ]
    )
    y = jnp.array([0.24, 0.23, 0.27, 0.21, 0.265, 0.225])

    a_obs = jnp.zeros_like(Z)
    P_obs = jnp.full_like(Z, jnp.inf)
    a_obs = a_obs.at[2, 1].set(0.22)
    P_obs = P_obs.at[2, 1].set(0.01)
    a_obs = a_obs.at[4, 2].set(0.18)
    P_obs = P_obs.at[4, 2].set(0.02)

    _, at_ekf, Pt_ekf, vt_ekf, Ft_ekf, Kt_ekf = kalman_filter_1d_ekf(
        a0=a0,
        P0=P0,
        Z=Z,
        sigma_h=sigma_h,
        sigma_q=sigma_q,
        y=y,
        logp=True,
        exponent=0.5,
        positivity_idx=positivity_idx,
        a_obs=a_obs,
        P_obs=P_obs,
    )
    at_dk = kalman_dk_smoother_1d_ekf(
        at=at_ekf,
        Pt=Pt_ekf,
        vt=vt_ekf,
        Ft=Ft_ekf,
        Kt=Kt_ekf,
        Z=Z,
        a0=a0,
        P0=P0,
        sigma_q=sigma_q,
        exponent=0.5,
        positivity_idx=positivity_idx,
        a_obs=a_obs,
        P_obs=P_obs,
    )
    at_rts, Pt_rts = kalman_rts_smoother_1d_ekf(
        at=at_ekf,
        Pt=Pt_ekf,
        sigma_q=sigma_q,
    )

    assert jnp.allclose(at_rts, at_dk, atol=1e-8)
    assert jnp.allclose(Pt_rts[-1], Pt_ekf[-1], atol=1e-10)
    assert jnp.all(Pt_rts[:-1] <= Pt_ekf[:-1] + 1e-10)


def test_ekf_rts_matches_linear_rts_in_fully_linear_limit() -> None:
    """The EKF RTS smoother should match the linear RTS smoother when linearised away."""
    a0 = jnp.array([0.1, 0.2, 0.15])
    P0 = jnp.array([0.8, 0.3, 0.25])
    sigma_h = jnp.array(0.05)
    sigma_q = jnp.array([0.02, 0.01, 0.015])
    positivity_idx = jnp.zeros(3, dtype=bool)

    Z = jnp.array(
        [
            [1.0, 0.4, 0.2],
            [1.0, 0.3, 0.25],
            [1.0, 0.5, 0.1],
            [1.0, 0.2, 0.35],
            [1.0, 0.45, 0.15],
            [1.0, 0.25, 0.3],
        ]
    )
    y = jnp.array([0.24, 0.23, 0.27, 0.21, 0.265, 0.225])

    a_obs = jnp.zeros_like(Z)
    P_obs = jnp.full_like(Z, jnp.inf)
    a_obs = a_obs.at[2, 1].set(0.22)
    P_obs = P_obs.at[2, 1].set(0.01)
    a_obs = a_obs.at[4, 2].set(0.18)
    P_obs = P_obs.at[4, 2].set(0.02)

    _, at_lin, Pt_lin, _, _, _ = kalman_filter_1d(
        a0=a0,
        P0=P0,
        Z=Z,
        sigma_h=sigma_h,
        sigma_q=sigma_q,
        y=y,
        logp=True,
        a_obs=a_obs,
        P_obs=P_obs,
        positivity_idx=positivity_idx,
    )
    _, at_ekf, Pt_ekf, _, _, _ = kalman_filter_1d_ekf(
        a0=a0,
        P0=P0,
        Z=Z,
        sigma_h=sigma_h,
        sigma_q=sigma_q,
        y=y,
        logp=True,
        exponent=0.5,
        positivity_idx=positivity_idx,
        a_obs=a_obs,
        P_obs=P_obs,
    )
    at_rts_lin, Pt_rts_lin = kalman_rts_smoother_1d(
        at=at_lin,
        Pt=Pt_lin,
        sigma_q=sigma_q,
    )
    at_rts_ekf, Pt_rts_ekf = kalman_rts_smoother_1d_ekf(
        at=at_ekf,
        Pt=Pt_ekf,
        sigma_q=sigma_q,
    )

    assert jnp.allclose(at_lin, at_ekf, atol=2e-5)
    assert jnp.allclose(Pt_lin - jnp.square(sigma_q), Pt_ekf, atol=3e-4)
    assert jnp.allclose(at_rts_lin, at_rts_ekf, atol=1e-5)
    assert jnp.allclose(Pt_rts_lin, Pt_rts_ekf, atol=2e-6)
