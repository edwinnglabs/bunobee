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

    The ``sigma_q`` prior family is selected by
    ``ssp_priors_nat.attrs["sigma_q_family"]`` (default
    ``"truncated_normal"``); the chosen family is propagated to
    ``attrs["sigma_q_family"]`` on the returned dataset so downstream
    samplers can dispatch on the same key.

    Parameters
    ----------
    ssp_priors_nat : xr.Dataset
        Natural-scale prior dataset. Required variables:

        - ``a0_nat`` : dims ``(state,)`` — initial state mean. Entries
          selected by ``positivity_idx`` must be strictly positive.
        - ``P0_nat`` : dims ``(state, state_dual)`` — initial state
          covariance.
        - ``positivity_idx`` : dims ``(state,)``, boolean — ``True`` selects
          states that use the nonlinear ``exp`` mapping in
          ``kalman_filter_1d_ekf``.

        ``sigma_q`` block, family ``"truncated_normal"`` (default):

        - ``sigma_q_loc_prior_nat`` : dims ``(state,)`` or compressed ``(2,)``
          — TruncatedNormal loc for ``sigma_q``. Per-state form is
          transformed element-wise against ``a0_nat``. Compressed form
          ``[first_state, shared_remaining]`` is transformed against
          ``a0_nat[:2]`` as representatives; callers using the compressed
          form must arrange states ``1:`` to share a common positivity flag
          and reference level.
        - ``sigma_q_scale_prior_nat`` : same dims; transformed via the same
          per-element formula as the loc.
        - ``sigma_q_low_prior_nat`` / ``sigma_q_high_prior_nat`` (optional):
          TruncatedNormal truncation bounds in natural scale; transformed
          identically to the loc.

        ``sigma_q`` block, family ``"beta"``:

        - ``sigma_q_alpha_prior`` / ``sigma_q_beta_prior`` : same shape rules
          as ``sigma_q_scale_prior_nat`` — dimensionless Beta shape
          parameters; passed through unchanged. Both must be ``> 1``
          element-wise so the Beta has a defined mode.
        - ``sigma_q_scale_prior_nat`` : multiplicative scale of the natural-
          scale prior ``sigma_q ~ scale_nat · Beta(α, β)``. Rescaled to a
          new ``scale_a`` so the mode of ``scale_a · Beta(α, β)`` in a-space
          equals the pushforward of the natural-scale mode.

        Other optional variables:

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
        ``P0`` (full covariance, symmetrised), the family-specific sigma_q
        variables (see above; suffix ``_nat`` is dropped on transformed
        outputs), optionally ``a_obs``, ``P_obs``, ``obs_idx``, and the
        passthrough ``positivity_idx``. ``attrs["sigma_q_family"]`` records
        the selected family. Dimensions and coordinates are preserved from
        the input.

    Raises
    ------
    ValueError
        If ``positivity_idx`` is missing, if ``sigma_q_family`` is
        unrecognised, if the family's required sigma_q variables are
        missing or shape-mismatched, if Beta shape parameters are not all
        ``> 1``, or if exactly one of ``a_obs_nat`` / ``P_obs_nat`` is
        present.

    Notes
    -----
    The ``sigma_q`` hyperprior is not a latent-state moment; it parameterises
    the process-noise scale itself. In the EKF a-space there is no exact
    state-independent analogue of a natural-scale additive ``sigma_q`` for a
    positivity state, so the transform keeps the chosen prior family and
    applies a local positive scale conversion against the reference level
    from ``a0_nat``. For positivity entries this uses::

        sigma_a = sqrt(log(1 + (sigma_nat / ref_level)^2)) / exponent

    which stays positive and matches the small-noise limit
    ``sigma_a ≈ sigma_nat / (exponent · ref_level)``. For the ``"beta"``
    family the same map is applied to the natural-scale mode
    ``scale_nat · (α − 1) / (α + β − 2)`` and the result is divided back by
    the mode fraction to recover the a-space scale.
    """
    if "positivity_idx" not in ssp_priors_nat:
        raise ValueError("ssp_priors_nat must contain a `positivity_idx` variable")

    has_a_obs = "a_obs_nat" in ssp_priors_nat
    has_P_obs = "P_obs_nat" in ssp_priors_nat
    if has_a_obs != has_P_obs:
        raise ValueError("a_obs_nat and P_obs_nat must both be present or both be absent")

    family = ssp_priors_nat.attrs.get("sigma_q_family", "truncated_normal")

    positivity = jnp.asarray(ssp_priors_nat["positivity_idx"].values, dtype=bool)
    a0_nat = jnp.asarray(ssp_priors_nat["a0_nat"].values)
    P0_nat = jnp.asarray(ssp_priors_nat["P0_nat"].values)

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

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "a0": (ssp_priors_nat["a0_nat"].dims, np.asarray(a0)),
        "P0": (ssp_priors_nat["P0_nat"].dims, np.asarray(P0)),
    }
    data_vars.update(
        _transform_sigma_q_block(ssp_priors_nat, family, n_states, safe_init, positivity, k)
    )

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

    return xr.Dataset(
        data_vars=data_vars,
        coords=ssp_priors_nat.coords,
        attrs={"sigma_q_family": family},
    )


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


def _transform_sigma_q_block(
    ssp_priors_nat: xr.Dataset,
    family: str,
    n_states: int,
    safe_init: jnp.ndarray,
    positivity: jnp.ndarray,
    k: float,
) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    """Dispatch sigma_q natural→a-space conversion on the chosen prior family.

    See ``transform_to_ekf`` for the per-family schema. Both branches reuse
    ``_moment_match_sigma`` for sigma-like quantities and
    ``_resolve_sigma_alignment`` for the per-state vs compressed ``(2,)``
    shape rules.
    """
    if family == "truncated_normal":
        return _transform_sigma_q_truncated_normal(
            ssp_priors_nat, n_states, safe_init, positivity, k
        )
    if family == "beta":
        return _transform_sigma_q_beta(
            ssp_priors_nat, n_states, safe_init, positivity, k
        )
    raise ValueError(
        f"unknown sigma_q_family: {family!r}; expected 'truncated_normal' or 'beta'"
    )


def _transform_sigma_q_truncated_normal(
    ssp_priors_nat: xr.Dataset,
    n_states: int,
    safe_init: jnp.ndarray,
    positivity: jnp.ndarray,
    k: float,
) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    for name in ("sigma_q_loc_prior_nat", "sigma_q_scale_prior_nat"):
        if name not in ssp_priors_nat:
            raise ValueError(f"sigma_q_family='truncated_normal' requires {name!r}")

    loc_nat = jnp.asarray(ssp_priors_nat["sigma_q_loc_prior_nat"].values)
    scale_nat = jnp.asarray(ssp_priors_nat["sigma_q_scale_prior_nat"].values)
    if loc_nat.shape != scale_nat.shape:
        raise ValueError(
            "sigma_q_loc_prior_nat and sigma_q_scale_prior_nat must share the same shape; "
            f"got {loc_nat.shape} and {scale_nat.shape}"
        )

    ref_level, sigma_positivity = _resolve_sigma_alignment(
        loc_nat.shape, n_states, safe_init, positivity
    )
    sigma_dims = ssp_priors_nat["sigma_q_loc_prior_nat"].dims

    out: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "sigma_q_loc_prior": (
            sigma_dims,
            np.asarray(_moment_match_sigma(loc_nat, ref_level, sigma_positivity, k)),
        ),
        "sigma_q_scale_prior": (
            sigma_dims,
            np.asarray(_moment_match_sigma(scale_nat, ref_level, sigma_positivity, k)),
        ),
    }

    for src_name, out_name in (
        ("sigma_q_low_prior_nat", "sigma_q_low_prior"),
        ("sigma_q_high_prior_nat", "sigma_q_high_prior"),
    ):
        if src_name in ssp_priors_nat:
            arr = jnp.asarray(ssp_priors_nat[src_name].values)
            out[out_name] = (
                sigma_dims,
                np.asarray(_moment_match_sigma(arr, ref_level, sigma_positivity, k)),
            )

    return out


def _transform_sigma_q_beta(
    ssp_priors_nat: xr.Dataset,
    n_states: int,
    safe_init: jnp.ndarray,
    positivity: jnp.ndarray,
    k: float,
) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    required = ("sigma_q_alpha_prior", "sigma_q_beta_prior", "sigma_q_scale_prior_nat")
    for name in required:
        if name not in ssp_priors_nat:
            raise ValueError(f"sigma_q_family='beta' requires {name!r}")

    alpha = jnp.asarray(ssp_priors_nat["sigma_q_alpha_prior"].values)
    beta = jnp.asarray(ssp_priors_nat["sigma_q_beta_prior"].values)
    scale_nat = jnp.asarray(ssp_priors_nat["sigma_q_scale_prior_nat"].values)
    if alpha.shape != beta.shape or alpha.shape != scale_nat.shape:
        raise ValueError(
            "sigma_q_alpha_prior, sigma_q_beta_prior, and sigma_q_scale_prior_nat must share the same shape; "
            f"got {alpha.shape}, {beta.shape}, and {scale_nat.shape}"
        )

    # Mode-matching requires a defined mode for Beta(alpha, beta), i.e. both > 1.
    if bool(jnp.any(alpha <= 1.0)) or bool(jnp.any(beta <= 1.0)):
        raise ValueError(
            "Beta mode-matching requires alpha > 1 and beta > 1 element-wise; "
            f"got alpha={np.asarray(alpha)}, beta={np.asarray(beta)}"
        )

    ref_level, sigma_positivity = _resolve_sigma_alignment(
        scale_nat.shape, n_states, safe_init, positivity
    )
    sigma_dims = ssp_priors_nat["sigma_q_scale_prior_nat"].dims

    mode_frac = (alpha - 1.0) / (alpha + beta - 2.0)
    mode_nat = scale_nat * mode_frac
    mode_a = _moment_match_sigma(mode_nat, ref_level, sigma_positivity, k)
    scale_a = jnp.where(sigma_positivity, mode_a / mode_frac, scale_nat)

    return {
        "sigma_q_alpha_prior": (sigma_dims, np.asarray(alpha)),
        "sigma_q_beta_prior": (sigma_dims, np.asarray(beta)),
        "sigma_q_scale_prior": (sigma_dims, np.asarray(scale_a)),
    }
