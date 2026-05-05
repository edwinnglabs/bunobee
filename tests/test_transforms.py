from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from wunkui.models.ssp.transforms import transform_to_ekf


K = 0.5


def _closed_form(mu_x: float, var_x: float, k: float = K) -> tuple[float, float]:
    sigma_y_sq = jnp.log1p(var_x / mu_x**2)
    mu_y = jnp.log(mu_x) - 0.5 * sigma_y_sq
    return float(mu_y / k), float(sigma_y_sq / (k * k))


def _make_priors(
    a0_nat: jnp.ndarray,
    P0_nat: jnp.ndarray,
    sigma_loc: jnp.ndarray | float = 0.0,
    sigma_scale: jnp.ndarray | float | None = None,
    a_obs_nat: jnp.ndarray | None = None,
    P_obs_nat: jnp.ndarray | None = None,
) -> dict:
    n_states = a0_nat.shape[0]
    loc_arr = jnp.broadcast_to(jnp.asarray(sigma_loc, dtype=float), (n_states,))
    scale_src = sigma_loc if sigma_scale is None else sigma_scale
    scale_arr = jnp.broadcast_to(jnp.asarray(scale_src, dtype=float), (n_states,))
    return {
        "a0_nat": a0_nat,
        "P0_nat": P0_nat,
        "sigma_q_loc_prior_nat": loc_arr,
        "sigma_q_scale_prior_nat": scale_arr,
        "a_obs_nat": a_obs_nat,
        "P_obs_nat": P_obs_nat,
        "obs_idx": None,
    }


class TestTransformToEkf:
    def test_closed_form_match_positivity_state(self):
        a0_nat = jnp.array([2.0, 5.0])
        P0_nat = jnp.diag(jnp.array([0.25, 1.0]))
        positivity = jnp.array([True, True])

        out = transform_to_ekf(_make_priors(a0_nat, P0_nat), positivity, exponent=K)

        for i in range(2):
            mu_ref, var_ref = _closed_form(float(a0_nat[i]), float(P0_nat[i, i]))
            assert jnp.allclose(out["a0"][i], mu_ref, atol=1e-12)
            assert jnp.allclose(out["P0"][i, i], var_ref, atol=1e-12)

    def test_linear_state_passthrough(self):
        a0_nat = jnp.array([1.5, -3.0, 7.0])
        P0_nat = jnp.array([[0.2, 0.05, 0.0], [0.05, 0.4, 0.0], [0.0, 0.0, 0.9]])
        sigma_q = jnp.array([0.1, 0.2, 0.3])
        positivity = jnp.zeros(3, dtype=bool)

        out = transform_to_ekf(
            _make_priors(a0_nat, P0_nat, sigma_loc=sigma_q),
            positivity,
            exponent=K,
        )

        assert jnp.allclose(out["a0"], a0_nat)
        assert jnp.allclose(out["P0"], P0_nat)
        assert jnp.allclose(out["sigma_q_loc_prior"], sigma_q)
        assert jnp.allclose(out["sigma_q_scale_prior"], sigma_q)

    def test_inf_variance_preserved(self):
        n_steps, n_states = 3, 2
        a0_nat = jnp.array([1.0, 1.0])
        P0_nat = jnp.eye(2)
        positivity = jnp.array([True, True])

        a_obs_nat = jnp.zeros((n_steps, n_states))
        P_obs_nat = jnp.full((n_steps, n_states), jnp.inf)
        a_obs_nat = a_obs_nat.at[1, 0].set(2.0)
        P_obs_nat = P_obs_nat.at[1, 0].set(0.5)

        out = transform_to_ekf(
            _make_priors(a0_nat, P0_nat, sigma_loc=0.1, a_obs_nat=a_obs_nat, P_obs_nat=P_obs_nat),
            positivity,
            exponent=K,
        )

        inf_mask = jnp.isinf(P_obs_nat)
        assert jnp.all(jnp.isinf(out["P_obs"][inf_mask]))
        assert jnp.allclose(out["a_obs"][inf_mask], a_obs_nat[inf_mask])

        mu_ref, var_ref = _closed_form(2.0, 0.5)
        assert jnp.allclose(out["a_obs"][1, 0], mu_ref, atol=1e-12)
        assert jnp.allclose(out["P_obs"][1, 0], var_ref, atol=1e-12)

    def test_mixed_positivity_covariance(self):
        a0_nat = jnp.array([2.0, 3.0])
        P0_nat = jnp.array([[0.5, 0.1], [0.1, 0.4]])
        positivity = jnp.array([True, False])

        out = transform_to_ekf(_make_priors(a0_nat, P0_nat), positivity, exponent=K)

        _, var00_ref = _closed_form(2.0, 0.5)
        assert jnp.allclose(out["P0"][0, 0], var00_ref, atol=1e-12)

        assert jnp.allclose(out["P0"][1, 1], P0_nat[1, 1], atol=1e-12)

        mixed_ref = float(P0_nat[0, 1]) / (K * float(a0_nat[0]))
        assert jnp.allclose(out["P0"][0, 1], mixed_ref, atol=1e-12)
        assert jnp.allclose(out["P0"][1, 0], mixed_ref, atol=1e-12)

    def test_both_positivity_off_diagonal(self):
        a0_nat = jnp.array([2.0, 3.0])
        P0_nat = jnp.array([[0.5, 0.2], [0.2, 0.4]])
        positivity = jnp.array([True, True])

        out = transform_to_ekf(_make_priors(a0_nat, P0_nat), positivity, exponent=K)

        off_ref = float(jnp.log1p(jnp.array(0.2 / (2.0 * 3.0))) / (K * K))
        assert jnp.allclose(out["P0"][0, 1], off_ref, atol=1e-12)
        assert jnp.allclose(out["P0"][1, 0], off_ref, atol=1e-12)

    def test_sigma_q_lognormal_match(self):
        a0_nat = jnp.array([2.0, 4.0])
        P0_nat = jnp.eye(2)
        sigma_q = jnp.array([0.3, 0.5])
        positivity = jnp.array([True, False])

        out = transform_to_ekf(
            _make_priors(a0_nat, P0_nat, sigma_loc=sigma_q),
            positivity,
            exponent=K,
        )

        _, var_ref = _closed_form(2.0, 0.3**2)
        assert jnp.allclose(out["sigma_q_loc_prior"][0] ** 2, var_ref, atol=1e-12)
        assert jnp.allclose(out["sigma_q_loc_prior"][1], sigma_q[1], atol=1e-12)

    def test_scalar_sigma_broadcasts(self):
        a0_nat = jnp.array([1.0, 2.0])
        P0_nat = jnp.eye(2)
        positivity = jnp.array([True, True])

        out = transform_to_ekf(
            _make_priors(a0_nat, P0_nat, sigma_loc=0.1),
            positivity,
            exponent=K,
        )

        assert out["sigma_q_loc_prior"].shape == (2,)
        assert out["sigma_q_scale_prior"].shape == (2,)

    def test_hyperprior_loc_and_scale_use_same_formula(self):
        a0_nat = jnp.array([3.0, 6.0])
        P0_nat = jnp.eye(2)
        sigma_loc = jnp.array([0.4, 0.2])
        sigma_scale = jnp.array([0.05, 0.1])
        positivity = jnp.array([True, True])

        out = transform_to_ekf(
            _make_priors(a0_nat, P0_nat, sigma_loc=sigma_loc, sigma_scale=sigma_scale),
            positivity,
            exponent=K,
        )

        for i in range(2):
            _, loc_var_ref = _closed_form(float(a0_nat[i]), float(sigma_loc[i] ** 2))
            _, scale_var_ref = _closed_form(float(a0_nat[i]), float(sigma_scale[i] ** 2))
            assert jnp.allclose(out["sigma_q_loc_prior"][i] ** 2, loc_var_ref, atol=1e-12)
            assert jnp.allclose(out["sigma_q_scale_prior"][i] ** 2, scale_var_ref, atol=1e-12)

    def test_sigma_q_bounds_use_same_formula(self):
        a0_nat = jnp.array([2.0, 3.0])
        P0_nat = jnp.eye(2)
        positivity = jnp.array([True, False])

        priors = _make_priors(a0_nat, P0_nat, sigma_loc=jnp.array([0.4, 0.2]), sigma_scale=jnp.array([0.05, 0.1]))
        priors["sigma_q_low_prior_nat"] = jnp.array([0.1, 0.05])
        priors["sigma_q_high_prior_nat"] = jnp.array([0.8, 0.25])

        out = transform_to_ekf(priors, positivity, exponent=K)

        _, low_var_ref = _closed_form(2.0, 0.1**2)
        _, high_var_ref = _closed_form(2.0, 0.8**2)
        assert jnp.allclose(out["sigma_q_low_prior"][0] ** 2, low_var_ref, atol=1e-12)
        assert jnp.allclose(out["sigma_q_high_prior"][0] ** 2, high_var_ref, atol=1e-12)
        assert jnp.allclose(out["sigma_q_low_prior"][1], 0.05, atol=1e-12)
        assert jnp.allclose(out["sigma_q_high_prior"][1], 0.25, atol=1e-12)

    def test_raises_on_mismatched_obs(self):
        a0_nat = jnp.array([1.0])
        P0_nat = jnp.eye(1)
        positivity = jnp.array([True])
        obs = jnp.zeros((2, 1))

        with pytest.raises(ValueError, match="both be provided or both be None"):
            transform_to_ekf(
                _make_priors(a0_nat, P0_nat, a_obs_nat=obs, P_obs_nat=None),
                positivity,
            )

    def test_returns_none_when_obs_none(self):
        a0_nat = jnp.array([1.0])
        P0_nat = jnp.eye(1)
        positivity = jnp.array([True])

        out = transform_to_ekf(_make_priors(a0_nat, P0_nat), positivity)

        assert out["a_obs"] is None
        assert out["P_obs"] is None

    def test_obs_idx_passthrough(self):
        a0_nat = jnp.array([1.0])
        P0_nat = jnp.eye(1)
        positivity = jnp.array([True])
        idx = jnp.array([3, 7, 11])

        priors = _make_priors(a0_nat, P0_nat)
        priors["obs_idx"] = idx
        out = transform_to_ekf(priors, positivity)

        assert jnp.array_equal(out["obs_idx"], idx)
