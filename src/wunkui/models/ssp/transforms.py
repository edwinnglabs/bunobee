from __future__ import annotations

import logging

import jax.numpy as jnp

logger = logging.getLogger("wunkui")


def to_a_space(
    a0: jnp.ndarray,
    P0: jnp.ndarray,
    sigma_q: jnp.ndarray | float,
    a_obs_loc: jnp.ndarray | None,
    a_obs_var: jnp.ndarray | None,
    positivity_idx: jnp.ndarray,
    exponent: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]:
    """Convert natural-scale EKF priors to a-space via lognormal moment-matching.

    Converts user-supplied priors on the natural/intensity scale (where
    ``x = exp(exponent · a)``) into Gaussian parameters in a-space, so they
    can be passed directly to ``kalman_filter_1d_ekf_st``. Only columns where
    ``positivity_idx[i] = True`` are transformed; linear states pass through
    unchanged. Entries of ``a_obs_var`` equal to ``jnp.inf`` are preserved
    (undisclosed-timestep no-ops).

    For each positivity state ``i`` with reference level ``μ_x_i > 0`` and
    variance ``σ_x_i² ≥ 0`` (with ``k = exponent``)::

        σ_y_i² = log(1 + σ_x_i² / μ_x_i²)
        μ_y_i  = log(μ_x_i) − 0.5 · σ_y_i²
        μ_a_i  = μ_y_i / k
        σ_a_i² = σ_y_i² / k²

    Off-diagonals of ``P0``:

    * both states positivity: ``Cov(A_i, A_j) = log(1 + C_ij / (μ_x_i μ_x_j)) / k²``
    * mixed (one positivity, one linear): ``Cov(A_i, X_j) = C_ij / (k · μ_x_i)``
    * both linear: unchanged

    Parameters
    ----------
    a0 : jnp.ndarray, shape (n_states,)
        Initial state mean in natural scale. Entries selected by
        ``positivity_idx`` must be strictly positive.
    P0 : jnp.ndarray, shape (n_states, n_states)
        Initial state covariance in natural scale.
    sigma_q : float or jnp.ndarray, shape () or (n_states,)
        Process-noise standard deviation in natural scale. Treated as a
        per-state std with reference level ``a0``.
    a_obs_loc : jnp.ndarray | None, shape (n_steps, n_states)
        Externally disclosed means in natural scale, or ``None``.
    a_obs_var : jnp.ndarray | None, shape (n_steps, n_states)
        Externally disclosed variances in natural scale, or ``None``. Entries
        equal to ``jnp.inf`` mark undisclosed steps and are preserved.
    positivity_idx : jnp.ndarray, shape (n_states,)
        Boolean mask — True selects states that use the nonlinear ``exp``
        mapping in ``kalman_filter_1d_ekf_st``.
    exponent : float, optional
        Exponent ``k`` in the forward map ``x = exp(k·a)``. Default 0.5.

    Returns
    -------
    a0_a : jnp.ndarray, shape (n_states,)
        a-space initial mean.
    P0_a : jnp.ndarray, shape (n_states, n_states)
        a-space initial covariance (symmetrized).
    sigma_q_a : jnp.ndarray, shape (n_states,)
        a-space process-noise standard deviation.
    a_obs_loc_a : jnp.ndarray | None, shape (n_steps, n_states)
        a-space observed means; ``None`` if input was ``None``.
    a_obs_var_a : jnp.ndarray | None, shape (n_steps, n_states)
        a-space observed variances; ``None`` if input was ``None``.

    Raises
    ------
    ValueError
        If exactly one of ``a_obs_loc`` / ``a_obs_var`` is ``None``.
    """
    if (a_obs_loc is None) != (a_obs_var is None):
        raise ValueError("a_obs_loc and a_obs_var must both be provided or both be None")

    k = exponent
    n_states = a0.shape[0]
    positivity = jnp.asarray(positivity_idx, dtype=bool)

    # Dummy of 1.0 for non-positivity states so log/divide remain finite
    safe_a0 = jnp.where(positivity, a0, 1.0)

    # a0 and diag(P0): joint lognormal moment-match per positivity state
    var_diag = jnp.diag(P0)
    sigma_y_sq_diag = jnp.log1p(var_diag / jnp.square(safe_a0))
    mu_y = jnp.log(safe_a0) - 0.5 * sigma_y_sq_diag
    a0_a = jnp.where(positivity, mu_y / k, a0)

    # P0 full covariance: exact multivariate lognormal match for (pos, pos) pairs,
    # delta-method hybrid for mixed pairs, identity for (lin, lin)
    safe_a0_outer = safe_a0[:, None] * safe_a0[None, :]
    cov_both = jnp.log1p(P0 / safe_a0_outer) / (k * k)
    denom_i = jnp.where(positivity, k * safe_a0, 1.0)[:, None]
    denom_j = jnp.where(positivity, k * safe_a0, 1.0)[None, :]
    cov_mixed = P0 / (denom_i * denom_j)
    pos_i = positivity[:, None]
    pos_j = positivity[None, :]
    both_pos = pos_i & pos_j
    P0_a = jnp.where(both_pos, cov_both, cov_mixed)
    P0_a = 0.5 * (P0_a + P0_a.T)

    # sigma_q as std with reference level a0
    sigma_q_arr = jnp.broadcast_to(jnp.asarray(sigma_q), (n_states,))
    sq_sigma_y_sq = jnp.log1p(jnp.square(sigma_q_arr) / jnp.square(safe_a0))
    sigma_q_a = jnp.where(positivity, jnp.sqrt(sq_sigma_y_sq) / k, sigma_q_arr)

    if a_obs_loc is None:
        return a0_a, P0_a, sigma_q_a, None, None

    # Transform only positivity columns at timesteps with finite variance; leave
    # undisclosed rows (var = inf) untouched so EKF fusion remains a no-op.
    finite_var = jnp.isfinite(a_obs_var)
    safe_var_obs = jnp.where(finite_var, a_obs_var, 0.0)
    safe_loc_obs = jnp.where(a_obs_loc > 0, a_obs_loc, 1.0)

    sigma_y_sq_obs = jnp.log1p(safe_var_obs / jnp.square(safe_loc_obs))
    mu_y_obs = jnp.log(safe_loc_obs) - 0.5 * sigma_y_sq_obs

    transform_mask = positivity[None, :] & finite_var
    a_obs_loc_a = jnp.where(transform_mask, mu_y_obs / k, a_obs_loc)
    a_obs_var_a = jnp.where(transform_mask, sigma_y_sq_obs / (k * k), a_obs_var)

    return a0_a, P0_a, sigma_q_a, a_obs_loc_a, a_obs_var_a
