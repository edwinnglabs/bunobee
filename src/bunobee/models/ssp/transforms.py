from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np
import xarray as xr

logger = logging.getLogger("bunobee")


def transform_to_ekf(
    ssp_priors_nat: xr.Dataset,
    exponent: float = 0.5,
) -> xr.Dataset:
    """Transform a natural-scale prior dataset into the EKF a-space prior dataset.

    The input ``ssp_priors_nat`` is structured as if for a vanilla Kalman
    filter — every entry lives in the natural / intensity scale and carries
    the ``_nat`` suffix. The returned dataset drops the suffix and contains
    the same quantities expressed in a-space, ready for
    ``kalman_filter_1d_ekf`` (which uses the forward map
    ``x = exp(exponent · a)`` for positivity states).

    Only entries selected by ``positivity_idx`` (read from
    ``ssp_priors_nat["positivity_idx"]``) are transformed; linear states pass
    through unchanged. Entries of ``P_obs_nat`` equal to ``jnp.inf`` are
    preserved as undisclosed-timestep no-ops.

    For each positivity state ``i`` with reference level ``μ_x_i > 0`` and
    variance ``σ_x_i² ≥ 0`` (with ``k = exponent``)::

        σ_y_i² = log(1 + σ_x_i² / μ_x_i²)
        μ_y_i  = log(μ_x_i) − 0.5 · σ_y_i²
        μ_a_i  = μ_y_i / k
        σ_a_i² = σ_y_i² / k²

    Off-diagonals of ``P0_nat``:

    * both states positivity: ``Cov(A_i, A_j) = log(1 + C_ij / (μ_x_i μ_x_j)) / k²``
    * mixed (one positivity, one linear): ``Cov(A_i, X_j) = C_ij / (k · μ_x_i)``
    * both linear: unchanged

    Parameters
    ----------
    ssp_priors_nat : xr.Dataset
        Natural-scale prior dataset. Required variables:

        - ``a0_nat`` : dims ``(state,)`` — initial state mean. Entries
          selected by ``positivity_idx`` must be strictly positive.
        - ``P0_nat`` : dims ``(state, state_dual)`` — initial state
          covariance.
        - ``sigma_q_loc_prior_nat`` : dims ``(state,)`` or compressed ``(2,)``
          — hyperprior loc for ``sigma_q``. Per-state form is transformed
          element-wise against ``a0_nat``. Compressed form
          ``[first_state, shared_remaining]`` is transformed against
          ``a0_nat[:2]`` as representatives; callers using the compressed
          form must arrange states ``1:`` to share a common positivity flag
          and reference level.
        - ``sigma_q_scale_prior_nat`` : same dims as
          ``sigma_q_loc_prior_nat``; transformed via the same per-element
          formula as the loc.
        - ``positivity_idx`` : dims ``(state,)``, boolean — ``True`` selects
          states that use the nonlinear ``exp`` mapping in
          ``kalman_filter_1d_ekf``.

        Optional variables:

        - ``sigma_q_low_prior_nat`` / ``sigma_q_high_prior_nat`` : truncation
          bounds for the ``sigma_q`` TruncatedNormal in natural scale; same
          shape rules as ``sigma_q_loc_prior_nat``; transformed via the same
          per-element formula as the loc. Omitted variables are absent from
          the returned dataset.
        - ``a_obs_nat`` / ``P_obs_nat`` : dims ``(time, state)`` — externally
          disclosed means / variances; ``jnp.inf`` in ``P_obs_nat`` marks
          undisclosed steps. Must be both present or both absent.
        - ``obs_idx`` : disclosure index, passed through unchanged.

    exponent : float, optional
        Exponent ``k`` in the forward map ``x = exp(k·a)``. Default 0.5.

    Returns
    -------
    xr.Dataset
        a-space prior dataset with un-suffixed variable names: ``a0``,
        ``P0`` (full covariance, symmetrised), ``sigma_q_loc_prior``,
        ``sigma_q_scale_prior``, optionally ``sigma_q_low_prior``,
        ``sigma_q_high_prior``, ``a_obs``, ``P_obs``, ``obs_idx``, and the
        passthrough ``positivity_idx``. Dimensions and coordinates are
        preserved from the input.

    Raises
    ------
    ValueError
        If ``positivity_idx`` is missing, or if exactly one of ``a_obs_nat``
        / ``P_obs_nat`` is present.

    Notes
    -----
    The ``sigma_q`` hyperprior is not a latent-state moment; it parameterises
    the process-noise scale itself. In the EKF a-space there is no exact
    state-independent analogue of a natural-scale additive ``sigma_q`` for a
    positivity state, so the transform keeps the TruncatedNormal prior family
    and applies a local positive scale conversion against the reference level
    from ``a0_nat``. For positivity entries this uses::

        sigma_a = sqrt(log(1 + (sigma_nat / ref_level)^2)) / exponent

    which stays positive and matches the small-noise limit
    ``sigma_a ≈ sigma_nat / (exponent · ref_level)``.
    """
    if "positivity_idx" not in ssp_priors_nat:
        raise ValueError("ssp_priors_nat must contain a `positivity_idx` variable")

    has_a_obs = "a_obs_nat" in ssp_priors_nat
    has_P_obs = "P_obs_nat" in ssp_priors_nat
    if has_a_obs != has_P_obs:
        raise ValueError("a_obs_nat and P_obs_nat must both be present or both be absent")

    positivity = jnp.asarray(ssp_priors_nat["positivity_idx"].values, dtype=bool)
    a0_nat = jnp.asarray(ssp_priors_nat["a0_nat"].values)
    P0_nat = jnp.asarray(ssp_priors_nat["P0_nat"].values)
    sigma_q_loc_prior_nat = jnp.asarray(ssp_priors_nat["sigma_q_loc_prior_nat"].values)
    sigma_q_scale_prior_nat = jnp.asarray(ssp_priors_nat["sigma_q_scale_prior_nat"].values)

    if sigma_q_loc_prior_nat.shape != sigma_q_scale_prior_nat.shape:
        raise ValueError(
            "sigma_q_loc_prior_nat and sigma_q_scale_prior_nat must share the same shape; "
            f"got {sigma_q_loc_prior_nat.shape} and {sigma_q_scale_prior_nat.shape}"
        )

    k = exponent
    n_states = a0_nat.shape[0]

    safe_init = jnp.where(positivity, a0_nat, 1.0)

    var_diag = jnp.diag(P0_nat)
    sigma_y_sq_diag = jnp.log1p(var_diag / jnp.square(safe_init))
    mu_y = jnp.log(safe_init) - 0.5 * sigma_y_sq_diag
    a0 = jnp.where(positivity, mu_y / k, a0_nat)

    safe_init_outer = safe_init[:, None] * safe_init[None, :]
    cov_both = jnp.log1p(P0_nat / safe_init_outer) / (k * k)
    denom_i = jnp.where(positivity, k * safe_init, 1.0)[:, None]
    denom_j = jnp.where(positivity, k * safe_init, 1.0)[None, :]
    cov_mixed = P0_nat / (denom_i * denom_j)
    both_pos = positivity[:, None] & positivity[None, :]
    P0 = jnp.where(both_pos, cov_both, cov_mixed)
    P0 = 0.5 * (P0 + P0.T)

    sigma_ref_level, sigma_positivity = _resolve_sigma_alignment(
        sigma_q_loc_prior_nat.shape, n_states, safe_init, positivity
    )
    sigma_q_loc_prior = _moment_match_sigma(
        sigma_q_loc_prior_nat, sigma_ref_level, sigma_positivity, k
    )
    sigma_q_scale_prior = _moment_match_sigma(
        sigma_q_scale_prior_nat, sigma_ref_level, sigma_positivity, k
    )

    sigma_dims = ssp_priors_nat["sigma_q_loc_prior_nat"].dims
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "a0": (ssp_priors_nat["a0_nat"].dims, np.asarray(a0)),
        "P0": (ssp_priors_nat["P0_nat"].dims, np.asarray(P0)),
        "sigma_q_loc_prior": (sigma_dims, np.asarray(sigma_q_loc_prior)),
        "sigma_q_scale_prior": (sigma_dims, np.asarray(sigma_q_scale_prior)),
    }

    if "sigma_q_low_prior_nat" in ssp_priors_nat:
        sigma_q_low_prior_nat = jnp.asarray(ssp_priors_nat["sigma_q_low_prior_nat"].values)
        sigma_q_low_prior = _moment_match_sigma(
            sigma_q_low_prior_nat, sigma_ref_level, sigma_positivity, k
        )
        data_vars["sigma_q_low_prior"] = (sigma_dims, np.asarray(sigma_q_low_prior))

    if "sigma_q_high_prior_nat" in ssp_priors_nat:
        sigma_q_high_prior_nat = jnp.asarray(ssp_priors_nat["sigma_q_high_prior_nat"].values)
        sigma_q_high_prior = _moment_match_sigma(
            sigma_q_high_prior_nat, sigma_ref_level, sigma_positivity, k
        )
        data_vars["sigma_q_high_prior"] = (sigma_dims, np.asarray(sigma_q_high_prior))

    if has_a_obs:
        a_obs_nat = jnp.asarray(ssp_priors_nat["a_obs_nat"].values)
        P_obs_nat = jnp.asarray(ssp_priors_nat["P_obs_nat"].values)

        finite_var = jnp.isfinite(P_obs_nat)
        safe_var_obs = jnp.where(finite_var, P_obs_nat, 0.0)
        safe_loc_obs = jnp.where(a_obs_nat > 0, a_obs_nat, 1.0)

        sigma_y_sq_obs = jnp.log1p(safe_var_obs / jnp.square(safe_loc_obs))
        mu_y_obs = jnp.log(safe_loc_obs) - 0.5 * sigma_y_sq_obs

        transform_mask = positivity[None, :] & finite_var
        a_obs = jnp.where(transform_mask, mu_y_obs / k, a_obs_nat)
        P_obs = jnp.where(transform_mask, sigma_y_sq_obs / (k * k), P_obs_nat)

        obs_dims = ssp_priors_nat["a_obs_nat"].dims
        data_vars["a_obs"] = (obs_dims, np.asarray(a_obs))
        data_vars["P_obs"] = (obs_dims, np.asarray(P_obs))

    if "obs_idx" in ssp_priors_nat:
        obs_idx_var = ssp_priors_nat["obs_idx"]
        data_vars["obs_idx"] = (obs_idx_var.dims, np.asarray(obs_idx_var.values))

    pos_var = ssp_priors_nat["positivity_idx"]
    data_vars["positivity_idx"] = (pos_var.dims, np.asarray(pos_var.values))

    return xr.Dataset(data_vars=data_vars, coords=ssp_priors_nat.coords)


def _moment_match_sigma(
    sigma_nat: jnp.ndarray,
    ref_level: jnp.ndarray,
    positivity: jnp.ndarray,
    k: float,
) -> jnp.ndarray:
    """Convert natural-scale increment standard deviations into a-space scales.

    For positivity states the EKF uses the latent parameterisation
    ``lambda = exp(k * a)``, so an additive standard deviation specified on the
    natural scale must be converted into the corresponding standard deviation of
    the latent increment ``eta_a``. Using a local reference level
    ``lambda_ref = ref_level``, this helper applies the same variance map as the
    diagonal ``P0`` transform, but without the mean-shift term required for
    state-level moment matching:

    ``sigma_a^2 = log(1 + sigma_nat^2 / lambda_ref^2) / k^2``

    and therefore

    ``sigma_a = sqrt(log(1 + sigma_nat^2 / lambda_ref^2)) / k``.

    This is appropriate for ``sigma_q`` because it parameterises the scale of a
    zero-mean increment, not the mean/variance pair of a latent state level.
    Linear states pass through unchanged.

    Parameters
    ----------
    sigma_nat : jnp.ndarray
        Natural-scale standard deviation(s) to convert.
    ref_level : jnp.ndarray
        Reference level(s) in the natural scale used to localise the lognormal
        variance map. For ``sigma_q`` this is typically derived from
        ``a0_nat``.
    positivity : jnp.ndarray
        Boolean mask selecting the states that use the nonlinear
        ``lambda = exp(k * a)`` mapping.
    k : float
        Exponent in the forward map ``lambda = exp(k * a)``.

    Returns
    -------
    jnp.ndarray
        Converted a-space standard deviation(s), with linear-state entries left
        unchanged.
    """
    sigma_arr = jnp.broadcast_to(jnp.asarray(sigma_nat), ref_level.shape)
    sq_sigma_y_sq = jnp.log1p(jnp.square(sigma_arr) / jnp.square(ref_level))
    return jnp.where(positivity, jnp.sqrt(sq_sigma_y_sq) / k, sigma_arr)


def _resolve_sigma_alignment(
    sigma_shape: tuple[int, ...],
    n_states: int,
    safe_init: jnp.ndarray,
    positivity: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pick reference level and positivity mask aligned with the sigma prior shape.

    Per-state input ``(n_states,)`` keeps full ``safe_init`` and ``positivity``.
    Compressed input ``(2,)`` (with ``n_states != 2``) uses ``[state[0], state[1]]``
    as representatives — ``state[1]`` stands in for the shared block at
    ``states[1:]`` and the caller is responsible for that block being uniform.
    """
    if sigma_shape == (n_states,):
        return safe_init, positivity
    if sigma_shape == (2,) and n_states >= 2:
        ref = jnp.array([safe_init[0], safe_init[1]])
        pos = jnp.array([positivity[0], positivity[1]])
        return ref, pos
    raise ValueError(
        "sigma_q prior shape must be (n_states,) or compressed (2,); "
        f"got shape {sigma_shape} for n_states={n_states}"
    )
