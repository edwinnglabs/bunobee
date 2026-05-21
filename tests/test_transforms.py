from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

jax.config.update("jax_enable_x64", True)

from bunobee.models.ssp.transforms import transform_to_ekf, transform_to_ekf_st


K = 0.5


def _closed_form(mu_x: float, var_x: float, k: float = K) -> tuple[float, float]:
    sigma_y_sq = jnp.log1p(var_x / mu_x**2)
    mu_y = jnp.log(mu_x) - 0.5 * sigma_y_sq
    return float(mu_y / k), float(sigma_y_sq / (k * k))


def _make_priors(
    a0_nat: jnp.ndarray,
    P0_nat: jnp.ndarray,
    positivity_idx: jnp.ndarray,
    sigma_loc: jnp.ndarray | float = 0.0,
    sigma_scale: jnp.ndarray | float | None = None,
    a_obs_nat: jnp.ndarray | None = None,
    P_obs_nat: jnp.ndarray | None = None,
    obs_idx: jnp.ndarray | None = None,
    sigma_low: jnp.ndarray | None = None,
    sigma_high: jnp.ndarray | None = None,
    family: str = "truncated_normal",
    alpha: jnp.ndarray | float | None = None,
    beta: jnp.ndarray | float | None = None,
) -> xr.Dataset:
    n_states = a0_nat.shape[0]
    p0_arr = np.asarray(P0_nat)
    p0_dims: tuple[str, ...] = ("state",) if p0_arr.ndim == 1 else ("state", "state_dual")

    data_vars: dict = {
        "a0": (("state",), np.asarray(a0_nat)),
        "P0": (p0_dims, p0_arr),
        "positivity_idx": (("state",), np.asarray(positivity_idx, dtype=bool)),
    }

    if family == "truncated_normal":
        loc_arr = np.broadcast_to(np.asarray(sigma_loc, dtype=float), (n_states,)).copy()
        scale_src = sigma_loc if sigma_scale is None else sigma_scale
        scale_arr = np.broadcast_to(np.asarray(scale_src, dtype=float), (n_states,)).copy()
        data_vars["sigma_q_loc_prior"] = (("state",), loc_arr)
        data_vars["sigma_q_scale_prior"] = (("state",), scale_arr)
        if sigma_low is not None:
            data_vars["sigma_q_low_prior"] = (("state",), np.asarray(sigma_low))
        if sigma_high is not None:
            data_vars["sigma_q_high_prior"] = (("state",), np.asarray(sigma_high))
    elif family == "beta":
        if alpha is None or beta is None or sigma_scale is None:
            raise ValueError("Beta family requires alpha, beta, and sigma_scale")
        alpha_arr = np.broadcast_to(np.asarray(alpha, dtype=float), (n_states,)).copy()
        beta_arr = np.broadcast_to(np.asarray(beta, dtype=float), (n_states,)).copy()
        scale_arr = np.broadcast_to(np.asarray(sigma_scale, dtype=float), (n_states,)).copy()
        data_vars["sigma_q_alpha_prior"] = (("state",), alpha_arr)
        data_vars["sigma_q_beta_prior"] = (("state",), beta_arr)
        data_vars["sigma_q_scale_prior"] = (("state",), scale_arr)

    if a_obs_nat is not None:
        data_vars["a_obs"] = (("time", "state"), np.asarray(a_obs_nat))
    if P_obs_nat is not None:
        data_vars["P_obs"] = (("time", "state"), np.asarray(P_obs_nat))
    if obs_idx is not None:
        data_vars["obs_idx"] = (("obs_point",), np.asarray(obs_idx))

    return xr.Dataset(data_vars, attrs={"sigma_q_family": family})


class TestTransformToEkf:
    """Diagonal-P0 variant (single-series, kalman_filter_1d_ekf target)."""

    def test_diagonal_input_gives_diagonal_output(self):
        a0_nat = jnp.array([2.0, 5.0, -1.0])
        P0_diag = jnp.array([0.25, 1.0, 0.4])
        positivity = jnp.array([True, True, False])

        out = transform_to_ekf(_make_priors(a0_nat, P0_diag, positivity), exponent=K)

        assert out["P0"].ndim == 1
        assert "state_dual" not in out.coords
        for i, pos in enumerate([True, True, False]):
            _, var_ref = _closed_form(float(a0_nat[i]), float(P0_diag[i]))
            if pos:
                assert jnp.allclose(out["P0"].values[i], var_ref, atol=1e-12)
            else:
                assert jnp.allclose(out["P0"].values[i], P0_diag[i], atol=1e-12)

    def test_closed_form_match_positivity_state(self):
        a0_nat = jnp.array([2.0, 5.0])
        P0_diag = jnp.array([0.25, 1.0])
        positivity = jnp.array([True, True])

        out = transform_to_ekf(_make_priors(a0_nat, P0_diag, positivity), exponent=K)

        for i in range(2):
            mu_ref, var_ref = _closed_form(float(a0_nat[i]), float(P0_diag[i]))
            assert jnp.allclose(out["a0"].values[i], mu_ref, atol=1e-12)
            assert jnp.allclose(out["P0"].values[i], var_ref, atol=1e-12)

    def test_sigma_q_lognormal_match(self):
        a0_nat = jnp.array([2.0, 4.0])
        P0_diag = jnp.ones(2)
        sigma_q = jnp.array([0.3, 0.5])
        positivity = jnp.array([True, False])

        out = transform_to_ekf(
            _make_priors(a0_nat, P0_diag, positivity, sigma_loc=sigma_q),
            exponent=K,
        )

        _, var_ref = _closed_form(2.0, 0.3**2)
        assert jnp.allclose(out["sigma_q_loc_prior"].values[0] ** 2, var_ref, atol=1e-12)
        assert jnp.allclose(out["sigma_q_loc_prior"].values[1], sigma_q[1], atol=1e-12)

    def test_inf_variance_preserved(self):
        n_steps, n_states = 3, 2
        a0_nat = jnp.array([1.0, 1.0])
        P0_diag = jnp.ones(2)
        positivity = jnp.array([True, True])

        a_obs_nat = jnp.zeros((n_steps, n_states))
        P_obs_nat = jnp.full((n_steps, n_states), jnp.inf)
        a_obs_nat = a_obs_nat.at[1, 0].set(2.0)
        P_obs_nat = P_obs_nat.at[1, 0].set(0.5)

        out = transform_to_ekf(
            _make_priors(
                a0_nat, P0_diag, positivity,
                sigma_loc=0.1, a_obs_nat=a_obs_nat, P_obs_nat=P_obs_nat,
            ),
            exponent=K,
        )

        inf_mask = jnp.isinf(P_obs_nat)
        assert jnp.all(jnp.isinf(out["P_obs"].values[inf_mask]))
        assert jnp.allclose(out["a_obs"].values[inf_mask], a_obs_nat[inf_mask])

        mu_ref, var_ref = _closed_form(2.0, 0.5)
        assert jnp.allclose(out["a_obs"].values[1, 0], mu_ref, atol=1e-12)
        assert jnp.allclose(out["P_obs"].values[1, 0], var_ref, atol=1e-12)

    def test_raises_on_mismatched_obs(self):
        a0_nat = jnp.array([1.0])
        P0_diag = jnp.ones(1)
        positivity = jnp.array([True])
        obs = jnp.zeros((2, 1))

        with pytest.raises(ValueError, match="both be present or both be absent"):
            transform_to_ekf(
                _make_priors(a0_nat, P0_diag, positivity, a_obs_nat=obs, P_obs_nat=None),
            )

    def test_raises_when_positivity_missing(self):
        ds = xr.Dataset(
            {
                "a0": (("state",), np.array([1.0])),
                "P0": (("state",), np.array([1.0])),
                "sigma_q_loc_prior": (("state",), np.array([0.1])),
                "sigma_q_scale_prior": (("state",), np.array([0.1])),
            }
        )

        with pytest.raises(ValueError, match="positivity_idx"):
            transform_to_ekf(ds)

    def test_omits_obs_when_absent(self):
        a0_nat = jnp.array([1.0])
        P0_diag = jnp.ones(1)
        positivity = jnp.array([True])

        out = transform_to_ekf(_make_priors(a0_nat, P0_diag, positivity))

        assert "a_obs" not in out
        assert "P_obs" not in out

    def test_obs_idx_passthrough(self):
        a0_nat = jnp.array([1.0])
        P0_diag = jnp.ones(1)
        positivity = jnp.array([True])
        idx = jnp.array([3, 7, 11])

        priors = _make_priors(a0_nat, P0_diag, positivity, obs_idx=idx)
        out = transform_to_ekf(priors)

        assert jnp.array_equal(out["obs_idx"].values, np.asarray(idx))

    def test_default_family_attr_propagates(self):
        a0_nat = jnp.array([1.0, 2.0])
        P0_diag = jnp.ones(2)
        positivity = jnp.array([True, True])

        out = transform_to_ekf(_make_priors(a0_nat, P0_diag, positivity))

        assert out.attrs["sigma_q_family"] == "truncated_normal"

    def test_raises_on_2d_P0(self):
        a0_nat = jnp.array([1.0, 2.0])
        P0_full = jnp.eye(2)
        positivity = jnp.array([True, True])

        with pytest.raises(ValueError, match="transform_to_ekf_st"):
            transform_to_ekf(_make_priors(a0_nat, P0_full, positivity))


class TestTransformToEkfSt:
    """Full-covariance P0 variant (multi-series, kalman_filter_1d_ekf_st target)."""

    def test_closed_form_match_positivity_state(self):
        a0_nat = jnp.array([2.0, 5.0])
        P0_nat = jnp.diag(jnp.array([0.25, 1.0]))
        positivity = jnp.array([True, True])

        out = transform_to_ekf_st(_make_priors(a0_nat, P0_nat, positivity), exponent=K)

        for i in range(2):
            mu_ref, var_ref = _closed_form(float(a0_nat[i]), float(P0_nat[i, i]))
            assert jnp.allclose(out["a0"].values[i], mu_ref, atol=1e-12)
            assert jnp.allclose(out["P0"].values[i, i], var_ref, atol=1e-12)

    def test_linear_state_passthrough(self):
        a0_nat = jnp.array([1.5, -3.0, 7.0])
        P0_nat = jnp.array([[0.2, 0.05, 0.0], [0.05, 0.4, 0.0], [0.0, 0.0, 0.9]])
        sigma_q = jnp.array([0.1, 0.2, 0.3])
        positivity = jnp.zeros(3, dtype=bool)

        out = transform_to_ekf_st(
            _make_priors(a0_nat, P0_nat, positivity, sigma_loc=sigma_q),
            exponent=K,
        )

        assert jnp.allclose(out["a0"].values, a0_nat)
        assert jnp.allclose(out["P0"].values, P0_nat)
        assert jnp.allclose(out["sigma_q_loc_prior"].values, sigma_q)
        assert jnp.allclose(out["sigma_q_scale_prior"].values, sigma_q)

    def test_inf_variance_preserved(self):
        n_steps, n_states = 3, 2
        a0_nat = jnp.array([1.0, 1.0])
        P0_nat = jnp.eye(2)
        positivity = jnp.array([True, True])

        a_obs_nat = jnp.zeros((n_steps, n_states))
        P_obs_nat = jnp.full((n_steps, n_states), jnp.inf)
        a_obs_nat = a_obs_nat.at[1, 0].set(2.0)
        P_obs_nat = P_obs_nat.at[1, 0].set(0.5)

        out = transform_to_ekf_st(
            _make_priors(
                a0_nat, P0_nat, positivity,
                sigma_loc=0.1, a_obs_nat=a_obs_nat, P_obs_nat=P_obs_nat,
            ),
            exponent=K,
        )

        inf_mask = jnp.isinf(P_obs_nat)
        assert jnp.all(jnp.isinf(out["P_obs"].values[inf_mask]))
        assert jnp.allclose(out["a_obs"].values[inf_mask], a_obs_nat[inf_mask])

        mu_ref, var_ref = _closed_form(2.0, 0.5)
        assert jnp.allclose(out["a_obs"].values[1, 0], mu_ref, atol=1e-12)
        assert jnp.allclose(out["P_obs"].values[1, 0], var_ref, atol=1e-12)

    def test_mixed_positivity_covariance(self):
        a0_nat = jnp.array([2.0, 3.0])
        P0_nat = jnp.array([[0.5, 0.1], [0.1, 0.4]])
        positivity = jnp.array([True, False])

        out = transform_to_ekf_st(_make_priors(a0_nat, P0_nat, positivity), exponent=K)

        _, var00_ref = _closed_form(2.0, 0.5)
        assert jnp.allclose(out["P0"].values[0, 0], var00_ref, atol=1e-12)

        assert jnp.allclose(out["P0"].values[1, 1], P0_nat[1, 1], atol=1e-12)

        mixed_ref = float(P0_nat[0, 1]) / (K * float(a0_nat[0]))
        assert jnp.allclose(out["P0"].values[0, 1], mixed_ref, atol=1e-12)
        assert jnp.allclose(out["P0"].values[1, 0], mixed_ref, atol=1e-12)

    def test_both_positivity_off_diagonal(self):
        a0_nat = jnp.array([2.0, 3.0])
        P0_nat = jnp.array([[0.5, 0.2], [0.2, 0.4]])
        positivity = jnp.array([True, True])

        out = transform_to_ekf_st(_make_priors(a0_nat, P0_nat, positivity), exponent=K)

        off_ref = float(jnp.log1p(jnp.array(0.2 / (2.0 * 3.0))) / (K * K))
        assert jnp.allclose(out["P0"].values[0, 1], off_ref, atol=1e-12)
        assert jnp.allclose(out["P0"].values[1, 0], off_ref, atol=1e-12)

    def test_sigma_q_lognormal_match(self):
        a0_nat = jnp.array([2.0, 4.0])
        P0_nat = jnp.eye(2)
        sigma_q = jnp.array([0.3, 0.5])
        positivity = jnp.array([True, False])

        out = transform_to_ekf_st(
            _make_priors(a0_nat, P0_nat, positivity, sigma_loc=sigma_q),
            exponent=K,
        )

        _, var_ref = _closed_form(2.0, 0.3**2)
        assert jnp.allclose(out["sigma_q_loc_prior"].values[0] ** 2, var_ref, atol=1e-12)
        assert jnp.allclose(out["sigma_q_loc_prior"].values[1], sigma_q[1], atol=1e-12)

    def test_scalar_sigma_broadcasts(self):
        a0_nat = jnp.array([1.0, 2.0])
        P0_nat = jnp.eye(2)
        positivity = jnp.array([True, True])

        out = transform_to_ekf_st(
            _make_priors(a0_nat, P0_nat, positivity, sigma_loc=0.1),
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

        out = transform_to_ekf_st(
            _make_priors(a0_nat, P0_nat, positivity, sigma_loc=sigma_loc, sigma_scale=sigma_scale),
            exponent=K,
        )

        for i in range(2):
            _, loc_var_ref = _closed_form(float(a0_nat[i]), float(sigma_loc[i] ** 2))
            _, scale_var_ref = _closed_form(float(a0_nat[i]), float(sigma_scale[i] ** 2))
            assert jnp.allclose(out["sigma_q_loc_prior"].values[i] ** 2, loc_var_ref, atol=1e-12)
            assert jnp.allclose(out["sigma_q_scale_prior"].values[i] ** 2, scale_var_ref, atol=1e-12)

    def test_sigma_q_bounds_use_same_formula(self):
        a0_nat = jnp.array([2.0, 3.0])
        P0_nat = jnp.eye(2)
        positivity = jnp.array([True, False])

        priors = _make_priors(
            a0_nat, P0_nat, positivity,
            sigma_loc=jnp.array([0.4, 0.2]),
            sigma_scale=jnp.array([0.05, 0.1]),
            sigma_low=jnp.array([0.1, 0.05]),
            sigma_high=jnp.array([0.8, 0.25]),
        )

        out = transform_to_ekf_st(priors, exponent=K)

        _, low_var_ref = _closed_form(2.0, 0.1**2)
        _, high_var_ref = _closed_form(2.0, 0.8**2)
        assert jnp.allclose(out["sigma_q_low_prior"].values[0] ** 2, low_var_ref, atol=1e-12)
        assert jnp.allclose(out["sigma_q_high_prior"].values[0] ** 2, high_var_ref, atol=1e-12)
        assert jnp.allclose(out["sigma_q_low_prior"].values[1], 0.05, atol=1e-12)
        assert jnp.allclose(out["sigma_q_high_prior"].values[1], 0.25, atol=1e-12)

    def test_preserves_input_coords(self):
        a0_nat = jnp.array([1.0, 2.0])
        P0_nat = jnp.eye(2)
        positivity = jnp.array([False, True])
        state_labels = ["intercept", "channel_a"]

        ds = _make_priors(a0_nat, P0_nat, positivity).assign_coords(
            state=state_labels, state_dual=state_labels,
        )

        out = transform_to_ekf_st(ds)

        assert list(out.coords["state"].values) == state_labels
        assert list(out.coords["state_dual"].values) == state_labels

    def test_default_family_attr_propagates(self):
        a0_nat = jnp.array([1.0, 2.0])
        P0_nat = jnp.eye(2)
        positivity = jnp.array([True, True])

        out = transform_to_ekf_st(_make_priors(a0_nat, P0_nat, positivity))

        assert out.attrs["sigma_q_family"] == "truncated_normal"

    def test_raises_on_1d_P0(self):
        a0_nat = jnp.array([1.0, 2.0])
        P0_diag = jnp.ones(2)
        positivity = jnp.array([True, True])

        with pytest.raises(ValueError, match="transform_to_ekf"):
            transform_to_ekf_st(_make_priors(a0_nat, P0_diag, positivity))

    def test_raises_on_3d_P0(self):
        a0_nat = jnp.array([1.0, 2.0])
        positivity = jnp.array([True, True])
        priors = _make_priors(a0_nat, jnp.eye(2), positivity)
        priors = priors.drop_vars("P0").assign(
            P0=(("state", "state_dual", "extra"), np.zeros((2, 2, 2)))
        )

        with pytest.raises(ValueError, match="2-D"):
            transform_to_ekf_st(priors)


def _beta_scale_a(mu_x: float, scale_nat: float, alpha: float, beta: float, k: float = K) -> float:
    mode_frac = (alpha - 1.0) / (alpha + beta - 2.0)
    mode_nat = scale_nat * mode_frac
    sigma_y_sq = jnp.log1p(mode_nat ** 2 / mu_x ** 2)
    mode_a = jnp.sqrt(sigma_y_sq) / k
    return float(mode_a / mode_frac)


class TestTransformToEkfBeta:
    """Diagonal-P0 variant, Beta sigma_q family."""

    def test_alpha_beta_passthrough_and_scale_mode_match(self):
        a0_nat = jnp.array([2.0, 5.0])
        P0_diag = jnp.array([0.25, 1.0])
        positivity = jnp.array([True, True])
        scale_nat = jnp.array([0.2, 0.5])

        out = transform_to_ekf(
            _make_priors(
                a0_nat, P0_diag, positivity,
                family="beta", alpha=2.0, beta=10.0, sigma_scale=scale_nat,
            ),
            exponent=K,
        )

        assert out.attrs["sigma_q_family"] == "beta"
        assert jnp.allclose(out["sigma_q_alpha_prior"].values, 2.0)
        assert jnp.allclose(out["sigma_q_beta_prior"].values, 10.0)

        for i in range(2):
            ref = _beta_scale_a(float(a0_nat[i]), float(scale_nat[i]), 2.0, 10.0)
            assert jnp.allclose(out["sigma_q_scale_prior"].values[i], ref, atol=1e-12)

    def test_raises_when_alpha_le_one(self):
        a0_nat = jnp.array([1.0, 2.0])
        P0_diag = jnp.ones(2)
        positivity = jnp.array([True, True])

        priors = _make_priors(
            a0_nat, P0_diag, positivity,
            family="beta", alpha=jnp.array([1.0, 2.0]), beta=jnp.array([5.0, 5.0]),
            sigma_scale=jnp.array([0.1, 0.2]),
        )
        with pytest.raises(ValueError, match="alpha > 1 and beta > 1"):
            transform_to_ekf(priors)


class TestMatchModes:
    """`match` argument: mean / median / linearize."""

    def test_median_drops_mean_correction(self):
        # Median match: exp(k * a0_a) = mu_x exactly; variance unchanged from mean-match.
        a0_nat = jnp.array([0.1, 0.1])
        P0_diag = jnp.array([0.1, 0.1])
        positivity = jnp.array([True, False])  # linear second state for sanity

        out = transform_to_ekf(
            _make_priors(a0_nat, P0_diag, positivity, sigma_loc=0.1),
            exponent=K,
            match="median",
        )

        # Positivity state: a0_a = log(mu_x)/k, lambda(a0_a) = mu_x.
        assert jnp.allclose(out["a0"].values[0], float(jnp.log(0.1)) / K, atol=1e-12)
        assert jnp.allclose(jnp.exp(K * out["a0"].values[0]), 0.1, atol=1e-12)
        # Variance is the same as mean-match: log1p(0.1 / 0.01) / k^2.
        _, var_ref = _closed_form(0.1, 0.1)
        assert jnp.allclose(out["P0"].values[0], var_ref, atol=1e-12)
        # Linear state passes through.
        assert jnp.allclose(out["a0"].values[1], 0.1, atol=1e-12)
        assert jnp.allclose(out["P0"].values[1], 0.1, atol=1e-12)
        assert out.attrs["match"] == "median"

    def test_linearize_uses_delta_method(self):
        # Linearize: sigma_a = sigma_x / (k * mu_x); a0 same as median.
        a0_nat = jnp.array([0.1, 0.1])
        P0_diag = jnp.array([0.1, 0.1])
        positivity = jnp.array([True, False])

        out = transform_to_ekf(
            _make_priors(a0_nat, P0_diag, positivity, sigma_loc=0.05),
            exponent=K,
            match="linearize",
        )

        assert jnp.allclose(out["a0"].values[0], float(jnp.log(0.1)) / K, atol=1e-12)
        # Delta-method variance: P0_a = P0_nat / (k * mu_x)^2 = 0.1 / 0.0025 = 40.
        assert jnp.allclose(out["P0"].values[0], 0.1 / (K * 0.1) ** 2, atol=1e-12)
        # sigma_q via delta: 0.05 / (0.5 * 0.1) = 1.0
        assert jnp.allclose(out["sigma_q_loc_prior"].values[0], 0.05 / (K * 0.1), atol=1e-12)
        # Linear state pass-through everywhere.
        assert jnp.allclose(out["a0"].values[1], 0.1, atol=1e-12)
        assert jnp.allclose(out["P0"].values[1], 0.1, atol=1e-12)
        assert jnp.allclose(out["sigma_q_loc_prior"].values[1], 0.05, atol=1e-12)
        assert out.attrs["match"] == "linearize"

    def test_mean_is_backward_compatible_default(self):
        # Calling without `match` should match the explicit mean-match formula.
        a0_nat = jnp.array([0.3, 1.5])
        P0_diag = jnp.array([0.1, 0.4])
        positivity = jnp.array([True, True])

        out_default = transform_to_ekf(
            _make_priors(a0_nat, P0_diag, positivity, sigma_loc=0.1), exponent=K,
        )
        out_explicit = transform_to_ekf(
            _make_priors(a0_nat, P0_diag, positivity, sigma_loc=0.1), exponent=K,
            match="mean",
        )

        assert jnp.allclose(out_default["a0"].values, out_explicit["a0"].values)
        assert jnp.allclose(out_default["P0"].values, out_explicit["P0"].values)
        assert jnp.allclose(
            out_default["sigma_q_loc_prior"].values,
            out_explicit["sigma_q_loc_prior"].values,
        )

    def test_a_obs_uses_match_mode(self):
        # a_obs/P_obs should follow the same match formula as a0/P0.
        n_steps = 2
        a0_nat = jnp.array([0.1])
        P0_diag = jnp.array([0.01])
        positivity = jnp.array([True])
        a_obs_nat = jnp.array([[0.2], [0.5]])
        P_obs_nat = jnp.array([[0.04], [jnp.inf]])

        out = transform_to_ekf(
            _make_priors(
                a0_nat, P0_diag, positivity,
                sigma_loc=0.05, a_obs_nat=a_obs_nat, P_obs_nat=P_obs_nat,
            ),
            exponent=K,
            match="linearize",
        )

        # Disclosed step: linearize formula.
        assert jnp.allclose(out["a_obs"].values[0, 0], float(jnp.log(0.2)) / K, atol=1e-12)
        assert jnp.allclose(out["P_obs"].values[0, 0], 0.04 / (K * 0.2) ** 2, atol=1e-12)
        # Undisclosed step: inf preserved, a_obs passed through.
        assert jnp.isinf(out["P_obs"].values[1, 0])
        assert jnp.allclose(out["a_obs"].values[1, 0], 0.5, atol=1e-12)

    def test_st_linearize_off_diagonal(self):
        # Both-positivity off-diag under linearize: C / (k^2 * mu_i * mu_j).
        a0_nat = jnp.array([0.1, 0.2])
        P0_nat = jnp.array([[0.01, 0.005], [0.005, 0.04]])
        positivity = jnp.array([True, True])

        out = transform_to_ekf_st(
            _make_priors(a0_nat, P0_nat, positivity), exponent=K, match="linearize",
        )

        off_ref = 0.005 / (K * K * 0.1 * 0.2)
        assert jnp.allclose(out["P0"].values[0, 1], off_ref, atol=1e-12)
        assert jnp.allclose(out["P0"].values[1, 0], off_ref, atol=1e-12)
        # Diagonal also delta-method.
        assert jnp.allclose(out["P0"].values[0, 0], 0.01 / (K * 0.1) ** 2, atol=1e-12)
        assert jnp.allclose(out["P0"].values[1, 1], 0.04 / (K * 0.2) ** 2, atol=1e-12)

    def test_raises_on_unknown_match(self):
        a0_nat = jnp.array([1.0])
        P0_diag = jnp.ones(1)
        positivity = jnp.array([True])

        with pytest.raises(ValueError, match="unknown match mode"):
            transform_to_ekf(_make_priors(a0_nat, P0_diag, positivity), match="mode")


class TestTransformToEkfStBeta:
    """Full-covariance P0 variant, Beta sigma_q family."""

    def test_alpha_beta_passthrough_and_scale_mode_match(self):
        a0_nat = jnp.array([2.0, 5.0])
        P0_nat = jnp.diag(jnp.array([0.25, 1.0]))
        positivity = jnp.array([True, True])
        scale_nat = jnp.array([0.2, 0.5])

        out = transform_to_ekf_st(
            _make_priors(
                a0_nat, P0_nat, positivity,
                family="beta", alpha=2.0, beta=10.0, sigma_scale=scale_nat,
            ),
            exponent=K,
        )

        assert out.attrs["sigma_q_family"] == "beta"
        assert jnp.allclose(out["sigma_q_alpha_prior"].values, 2.0)
        assert jnp.allclose(out["sigma_q_beta_prior"].values, 10.0)

        for i in range(2):
            ref = _beta_scale_a(float(a0_nat[i]), float(scale_nat[i]), 2.0, 10.0)
            assert jnp.allclose(out["sigma_q_scale_prior"].values[i], ref, atol=1e-12)

    def test_linear_state_scale_passthrough(self):
        a0_nat = jnp.array([2.0, -1.0])
        P0_nat = jnp.eye(2)
        positivity = jnp.array([True, False])
        scale_nat = jnp.array([0.3, 0.7])

        out = transform_to_ekf_st(
            _make_priors(
                a0_nat, P0_nat, positivity,
                family="beta", alpha=jnp.array([3.0, 4.0]), beta=jnp.array([8.0, 6.0]),
                sigma_scale=scale_nat,
            ),
            exponent=K,
        )

        # positivity entry: mode-matched
        ref = _beta_scale_a(2.0, 0.3, 3.0, 8.0)
        assert jnp.allclose(out["sigma_q_scale_prior"].values[0], ref, atol=1e-12)
        # linear entry: passthrough
        assert jnp.allclose(out["sigma_q_scale_prior"].values[1], 0.7, atol=1e-12)
        # alpha/beta pass through on both
        assert jnp.allclose(out["sigma_q_alpha_prior"].values, jnp.array([3.0, 4.0]))
        assert jnp.allclose(out["sigma_q_beta_prior"].values, jnp.array([8.0, 6.0]))

    def test_ssp_v2_style_scaled_beta(self):
        # Mirrors the v2 notebook: scale_nat = sdy * 0.1 for intercept,
        # sdy_over_sdx * 0.1 for regressors; Beta(2, 10).
        sdy = 0.078
        sdy_over_sdx = jnp.array([0.58, 0.29, 0.45])
        # state 0 = intercept (linear); states 1..3 = regressors (positivity)
        a0_nat = jnp.array([0.0, 1.2, 0.8, 1.5])
        P0_nat = jnp.eye(4)
        positivity = jnp.array([False, True, True, True])
        scale_nat = jnp.concatenate([jnp.array([sdy * 0.1]), sdy_over_sdx * 0.1])

        out = transform_to_ekf_st(
            _make_priors(
                a0_nat, P0_nat, positivity,
                family="beta", alpha=2.0, beta=10.0, sigma_scale=scale_nat,
            ),
            exponent=K,
        )

        # intercept passes through unchanged
        assert jnp.allclose(out["sigma_q_scale_prior"].values[0], sdy * 0.1, atol=1e-12)
        # regressors mode-matched against their reference levels
        for i in range(1, 4):
            ref = _beta_scale_a(float(a0_nat[i]), float(scale_nat[i]), 2.0, 10.0)
            assert jnp.allclose(out["sigma_q_scale_prior"].values[i], ref, atol=1e-12)

    def test_raises_when_alpha_le_one(self):
        a0_nat = jnp.array([1.0, 2.0])
        P0_nat = jnp.eye(2)
        positivity = jnp.array([True, True])

        priors = _make_priors(
            a0_nat, P0_nat, positivity,
            family="beta", alpha=jnp.array([1.0, 2.0]), beta=jnp.array([5.0, 5.0]),
            sigma_scale=jnp.array([0.1, 0.2]),
        )
        with pytest.raises(ValueError, match="alpha > 1 and beta > 1"):
            transform_to_ekf_st(priors)

    def test_raises_when_required_var_missing(self):
        a0_nat = jnp.array([1.0])
        P0_nat = jnp.eye(1)
        positivity = jnp.array([True])

        priors = _make_priors(
            a0_nat, P0_nat, positivity,
            family="beta", alpha=2.0, beta=10.0, sigma_scale=0.1,
        )
        priors = priors.drop_vars("sigma_q_alpha_prior")

        with pytest.raises(ValueError, match="sigma_q_alpha_prior"):
            transform_to_ekf_st(priors)

    def test_raises_on_unknown_family(self):
        a0_nat = jnp.array([1.0])
        P0_nat = jnp.eye(1)
        positivity = jnp.array([True])

        priors = _make_priors(a0_nat, P0_nat, positivity)
        priors.attrs["sigma_q_family"] = "lognormal"

        with pytest.raises(ValueError, match="unknown sigma_q_family"):
            transform_to_ekf_st(priors)
