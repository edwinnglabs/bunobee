from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from wunkui.models.ssp.transforms import to_a_space


K = 0.5


def _closed_form(mu_x: float, var_x: float, k: float = K) -> tuple[float, float]:
    sigma_y_sq = jnp.log1p(var_x / mu_x**2)
    mu_y = jnp.log(mu_x) - 0.5 * sigma_y_sq
    return float(mu_y / k), float(sigma_y_sq / (k * k))


class TestToASpace:
    def test_closed_form_match_positivity_state(self):
        a0 = jnp.array([2.0, 5.0])
        P0 = jnp.diag(jnp.array([0.25, 1.0]))
        positivity = jnp.array([True, True])

        a0_a, P0_a, _, _, _ = to_a_space(a0, P0, 0.0, None, None, positivity, exponent=K)

        for i in range(2):
            mu_ref, var_ref = _closed_form(float(a0[i]), float(P0[i, i]))
            assert jnp.allclose(a0_a[i], mu_ref, atol=1e-12)
            assert jnp.allclose(P0_a[i, i], var_ref, atol=1e-12)

    def test_linear_state_passthrough(self):
        a0 = jnp.array([1.5, -3.0, 7.0])
        P0 = jnp.array([[0.2, 0.05, 0.0], [0.05, 0.4, 0.0], [0.0, 0.0, 0.9]])
        sigma_q = jnp.array([0.1, 0.2, 0.3])
        positivity = jnp.zeros(3, dtype=bool)

        a0_a, P0_a, sigma_q_a, _, _ = to_a_space(a0, P0, sigma_q, None, None, positivity, exponent=K)

        assert jnp.allclose(a0_a, a0)
        assert jnp.allclose(P0_a, P0)
        assert jnp.allclose(sigma_q_a, sigma_q)

    def test_inf_variance_preserved(self):
        n_steps, n_states = 3, 2
        a0 = jnp.array([1.0, 1.0])
        P0 = jnp.eye(2)
        positivity = jnp.array([True, True])

        a_obs_loc = jnp.zeros((n_steps, n_states))
        a_obs_var = jnp.full((n_steps, n_states), jnp.inf)
        # one finite disclosure at (t=1, state=0)
        a_obs_loc = a_obs_loc.at[1, 0].set(2.0)
        a_obs_var = a_obs_var.at[1, 0].set(0.5)

        _, _, _, loc_a, var_a = to_a_space(a0, P0, 0.1, a_obs_loc, a_obs_var, positivity, exponent=K)

        # Undisclosed rows: var stays inf, loc stays at its original (zero) value
        inf_mask = jnp.isinf(a_obs_var)
        assert jnp.all(jnp.isinf(var_a[inf_mask]))
        assert jnp.allclose(loc_a[inf_mask], a_obs_loc[inf_mask])

        # Disclosed entry matches the closed-form lognormal match
        mu_ref, var_ref = _closed_form(2.0, 0.5)
        assert jnp.allclose(loc_a[1, 0], mu_ref, atol=1e-12)
        assert jnp.allclose(var_a[1, 0], var_ref, atol=1e-12)

    def test_mixed_positivity_covariance(self):
        a0 = jnp.array([2.0, 3.0])
        P0 = jnp.array([[0.5, 0.1], [0.1, 0.4]])
        positivity = jnp.array([True, False])

        _, P0_a, _, _, _ = to_a_space(a0, P0, 0.0, None, None, positivity, exponent=K)

        # (0,0): full lognormal match using mu=2, var=0.5
        _, var00_ref = _closed_form(2.0, 0.5)
        assert jnp.allclose(P0_a[0, 0], var00_ref, atol=1e-12)

        # (1,1): linear/linear passthrough
        assert jnp.allclose(P0_a[1, 1], P0[1, 1], atol=1e-12)

        # (0,1) mixed: delta-method — Cov / (k · mu_x_0)
        mixed_ref = float(P0[0, 1]) / (K * float(a0[0]))
        assert jnp.allclose(P0_a[0, 1], mixed_ref, atol=1e-12)
        assert jnp.allclose(P0_a[1, 0], mixed_ref, atol=1e-12)

    def test_both_positivity_off_diagonal(self):
        a0 = jnp.array([2.0, 3.0])
        P0 = jnp.array([[0.5, 0.2], [0.2, 0.4]])
        positivity = jnp.array([True, True])

        _, P0_a, _, _, _ = to_a_space(a0, P0, 0.0, None, None, positivity, exponent=K)

        off_ref = float(jnp.log1p(jnp.array(0.2 / (2.0 * 3.0))) / (K * K))
        assert jnp.allclose(P0_a[0, 1], off_ref, atol=1e-12)
        assert jnp.allclose(P0_a[1, 0], off_ref, atol=1e-12)

    def test_sigma_q_lognormal_match(self):
        a0 = jnp.array([2.0, 4.0])
        P0 = jnp.eye(2)
        sigma_q = jnp.array([0.3, 0.5])
        positivity = jnp.array([True, False])

        _, _, sq_a, _, _ = to_a_space(a0, P0, sigma_q, None, None, positivity, exponent=K)

        _, var_ref = _closed_form(2.0, 0.3**2)
        assert jnp.allclose(sq_a[0] ** 2, var_ref, atol=1e-12)
        # linear state: passthrough
        assert jnp.allclose(sq_a[1], sigma_q[1], atol=1e-12)

    def test_scalar_sigma_q_broadcasts(self):
        a0 = jnp.array([1.0, 2.0])
        P0 = jnp.eye(2)
        positivity = jnp.array([True, True])

        _, _, sq_a, _, _ = to_a_space(a0, P0, 0.1, None, None, positivity, exponent=K)

        assert sq_a.shape == (2,)

    def test_raises_on_mismatched_obs(self):
        a0 = jnp.array([1.0])
        P0 = jnp.eye(1)
        positivity = jnp.array([True])
        obs = jnp.zeros((2, 1))

        with pytest.raises(ValueError, match="both be provided or both be None"):
            to_a_space(a0, P0, 0.0, obs, None, positivity)

    def test_returns_none_when_obs_none(self):
        a0 = jnp.array([1.0])
        P0 = jnp.eye(1)
        positivity = jnp.array([True])

        _, _, _, loc_a, var_a = to_a_space(a0, P0, 0.0, None, None, positivity)

        assert loc_a is None
        assert var_a is None
