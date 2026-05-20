"""Natural-scale to EKF a-space prior transforms for the SSP Kalman filters.

The SSP EKF filters use a latent reparameterisation ``lambda = exp(k * a)`` for
positivity states (``k`` is the ``exponent`` argument throughout this module).
This module converts a natural-scale prior dataset, written in the same
variable names used by the linear filters, into the a-space prior dataset
consumed by the EKF filters.

Moment matching (per positivity state ``i`` with reference level ``mu_x_i > 0``
and variance ``sigma_x_i^2 >= 0``)::

    sigma_y_i^2 = log(1 + sigma_x_i^2 / mu_x_i^2)
    mu_y_i      = log(mu_x_i) - 0.5 * sigma_y_i^2
    mu_a_i      = mu_y_i / k
    sigma_a_i^2 = sigma_y_i^2 / k^2

Linear states pass through unchanged. Entries of ``P_obs`` equal to ``jnp.inf``
are preserved as undisclosed-timestep no-ops.

The ``sigma_q`` hyperprior is not a state moment but a process-noise scale.
The transform applies a local positive-scale conversion against the reference
level from ``a0`` (see :func:`_moment_match_sigma`).

Two public entry points target the two EKF filters:

- :func:`transform_to_ekf`    -- 1-D diagonal ``P0`` for ``kalman_filter_1d_ekf``.
- :func:`transform_to_ekf_st` -- 2-D full ``P0`` for ``kalman_filter_1d_ekf_st``.
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np
import xarray as xr

logger = logging.getLogger("bunobee")


def _validate_common(ssp_priors: xr.Dataset) -> None:
    """Validate inputs shared by both EKF transforms."""
    if "positivity_idx" not in ssp_priors:
        raise ValueError("ssp_priors must contain a `positivity_idx` variable")

    has_a_obs = "a_obs" in ssp_priors
    has_P_obs = "P_obs" in ssp_priors
    if has_a_obs != has_P_obs:
        raise ValueError("a_obs and P_obs must both be present or both be absent")


def _compute_a0_a_space(
    a0_nat: jnp.ndarray,
    var_diag: jnp.ndarray,
    positivity: jnp.ndarray,
    k: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(a0_a, safe_init, sigma_y_sq_diag)``.

    ``var_diag`` is a 1-D vector of state variances (the caller passes ``P0``
    directly when 1-D, or ``jnp.diag(P0)`` when 2-D).
    """
    safe_init = jnp.where(positivity, a0_nat, 1.0)
    sigma_y_sq_diag = jnp.log1p(var_diag / jnp.square(safe_init))
    mu_y = jnp.log(safe_init) - 0.5 * sigma_y_sq_diag
    a0_a = jnp.where(positivity, mu_y / k, a0_nat)
    return a0_a, safe_init, sigma_y_sq_diag


def _transform_obs_block(
    ssp_priors: xr.Dataset,
    positivity: jnp.ndarray,
    k: float,
) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    """Transform ``a_obs`` / ``P_obs`` to a-space; return ``{}`` if absent.

    Element-wise on ``(time, state)``; independent of ``P0`` shape. Entries of
    ``P_obs`` equal to ``jnp.inf`` are preserved as undisclosed-step no-ops.
    """
    if "a_obs" not in ssp_priors:
        return {}

    a_obs_nat = jnp.asarray(ssp_priors["a_obs"].values)
    P_obs_nat = jnp.asarray(ssp_priors["P_obs"].values)

    finite_var = jnp.isfinite(P_obs_nat)
    safe_var_obs = jnp.where(finite_var, P_obs_nat, 0.0)
    safe_loc_obs = jnp.where(a_obs_nat > 0, a_obs_nat, 1.0)

    sigma_y_sq_obs = jnp.log1p(safe_var_obs / jnp.square(safe_loc_obs))
    mu_y_obs = jnp.log(safe_loc_obs) - 0.5 * sigma_y_sq_obs

    transform_mask = positivity[None, :] & finite_var
    a_obs = jnp.where(transform_mask, mu_y_obs / k, a_obs_nat)
    P_obs = jnp.where(transform_mask, sigma_y_sq_obs / (k * k), P_obs_nat)

    obs_dims = ssp_priors["a_obs"].dims
    return {
        "a_obs": (obs_dims, np.asarray(a_obs)),
        "P_obs": (obs_dims, np.asarray(P_obs)),
    }


def transform_to_ekf(
    ssp_priors: xr.Dataset,
    exponent: float = 0.5,
) -> xr.Dataset:
    """Transform natural-scale priors to a-space for ``kalman_filter_1d_ekf``.

    Diagonal-``P0`` variant: ``P0`` must be 1-D with dims ``(state,)``. The
    output ``P0`` is also 1-D. For full-covariance ``P0``, use
    :func:`transform_to_ekf_st`.

    See the module docstring for the moment-matching formulas and ``sigma_q``
    conventions. The ``sigma_q`` prior family is selected by
    ``ssp_priors.attrs["sigma_q_family"]`` (default ``"truncated_normal"``);
    the chosen family is propagated to ``attrs["sigma_q_family"]`` on the
    returned dataset.

    Parameters
    ----------
    ssp_priors : xr.Dataset
        Required: ``a0`` ``(state,)``, ``P0`` ``(state,)`` diagonal,
        ``positivity_idx`` ``(state,)`` boolean, and the ``sigma_q`` block
        for the chosen family (see :func:`transform_to_ekf_st` for the
        per-family schema, which is identical here).
        Optional: ``a_obs`` / ``P_obs`` ``(time, state)`` (both or neither);
        ``obs_idx`` (passed through).
    exponent : float, optional
        Exponent ``k`` in the forward map ``x = exp(k * a)``. Default 0.5.

    Returns
    -------
    xr.Dataset
        a-space prior dataset with ``P0`` dims ``(state,)``. Dimensions,
        coordinates, and ``attrs["sigma_q_family"]`` are preserved.

    Raises
    ------
    ValueError
        If ``P0`` is not 1-D, if ``positivity_idx`` is missing, if exactly one
        of ``a_obs`` / ``P_obs`` is present, or if the ``sigma_q`` block
        violates its family-specific constraints.
    """
    _validate_common(ssp_priors)

    P0 = jnp.asarray(ssp_priors["P0"].values)
    if P0.ndim != 1:
        raise ValueError(
            f"transform_to_ekf requires a 1-D diagonal P0 with dims (state,); "
            f"got ndim={P0.ndim}, shape={P0.shape}. "
            f"Use transform_to_ekf_st for full covariance."
        )

    family = ssp_priors.attrs.get("sigma_q_family", "truncated_normal")
    positivity = jnp.asarray(ssp_priors["positivity_idx"].values, dtype=bool)
    a0_nat = jnp.asarray(ssp_priors["a0"].values)
    k = exponent
    n_states = a0_nat.shape[0]

    a0_a, safe_init, sigma_y_sq_diag = _compute_a0_a_space(a0_nat, P0, positivity, k)
    P0_a = jnp.where(positivity, sigma_y_sq_diag / (k * k), P0)

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "a0": (ssp_priors["a0"].dims, np.asarray(a0_a)),
        "P0": (("state",), np.asarray(P0_a)),
    }
    data_vars.update(
        _transform_sigma_q_block(ssp_priors, family, n_states, safe_init, positivity, k)
    )
    data_vars.update(_transform_obs_block(ssp_priors, positivity, k))

    if "obs_idx" in ssp_priors:
        obs_idx_var = ssp_priors["obs_idx"]
        data_vars["obs_idx"] = (obs_idx_var.dims, np.asarray(obs_idx_var.values))

    pos_var = ssp_priors["positivity_idx"]
    data_vars["positivity_idx"] = (pos_var.dims, np.asarray(pos_var.values))

    return xr.Dataset(
        data_vars=data_vars,
        coords=dict(ssp_priors.coords),
        attrs={"sigma_q_family": family},
    )


def transform_to_ekf_st(
    ssp_priors: xr.Dataset,
    exponent: float = 0.5,
) -> xr.Dataset:
    """Transform natural-scale priors to a-space for ``kalman_filter_1d_ekf_st``.

    Full-covariance ``P0`` variant: ``P0`` must be 2-D with dims
    ``(state, state_dual)``. The output ``P0`` preserves the input dims and is
    symmetrised. For diagonal ``P0``, use :func:`transform_to_ekf`.

    See the module docstring for the moment-matching formulas on the diagonal.
    Off-diagonals are transformed as follows (with ``k = exponent``):

    * both states positivity: ``Cov(A_i, A_j) = log(1 + C_ij / (mu_x_i mu_x_j)) / k^2``
    * mixed (one positivity, one linear): ``Cov(A_i, X_j) = C_ij / (k * mu_x_i)``
    * both linear: unchanged

    Parameters
    ----------
    ssp_priors : xr.Dataset
        Required: ``a0`` ``(state,)``, ``P0`` ``(state, state_dual)`` full
        covariance, ``positivity_idx`` ``(state,)`` boolean, and the
        ``sigma_q`` block for the chosen family.

        ``sigma_q`` block, family ``"truncated_normal"`` (default):

        - ``sigma_q_loc_prior`` : dims ``(state,)`` or compressed ``(2,)``
          -- TruncatedNormal loc; transformed element-wise against ``a0`` or
          ``a0[:2]`` representatives (compressed callers must arrange states
          ``1:`` to share a common positivity flag and reference level).
        - ``sigma_q_scale_prior`` : same dims; transformed identically.
        - ``sigma_q_low_prior`` / ``sigma_q_high_prior`` (optional): bounds;
          transformed identically to the loc.

        ``sigma_q`` block, family ``"beta"``:

        - ``sigma_q_alpha_prior`` / ``sigma_q_beta_prior`` : same shape rules
          as ``sigma_q_scale_prior``; passed through unchanged. Both must be
          ``> 1`` element-wise so the Beta has a defined mode.
        - ``sigma_q_scale_prior`` : rescaled so the mode of
          ``scale_a * Beta(alpha, beta)`` in a-space equals the pushforward
          of the natural-scale mode ``scale * (alpha - 1) / (alpha + beta - 2)``.

        Optional: ``a_obs`` / ``P_obs`` ``(time, state)`` (both or neither);
        ``obs_idx`` (passed through).
    exponent : float, optional
        Exponent ``k`` in the forward map ``x = exp(k * a)``. Default 0.5.

    Returns
    -------
    xr.Dataset
        a-space prior dataset with ``P0`` dims ``(state, state_dual)``,
        symmetrised. Dimensions, coordinates, and ``attrs["sigma_q_family"]``
        are preserved.

    Raises
    ------
    ValueError
        If ``P0`` is not 2-D, if ``positivity_idx`` is missing, if exactly one
        of ``a_obs`` / ``P_obs`` is present, or if the ``sigma_q`` block
        violates its family-specific constraints.
    """
    _validate_common(ssp_priors)

    P0 = jnp.asarray(ssp_priors["P0"].values)
    if P0.ndim != 2:
        raise ValueError(
            f"transform_to_ekf_st requires a 2-D full P0 with dims (state, state_dual); "
            f"got ndim={P0.ndim}, shape={P0.shape}. "
            f"Use transform_to_ekf for diagonal."
        )

    family = ssp_priors.attrs.get("sigma_q_family", "truncated_normal")
    positivity = jnp.asarray(ssp_priors["positivity_idx"].values, dtype=bool)
    a0_nat = jnp.asarray(ssp_priors["a0"].values)
    k = exponent
    n_states = a0_nat.shape[0]

    a0_a, safe_init, _ = _compute_a0_a_space(a0_nat, jnp.diag(P0), positivity, k)

    safe_init_outer = safe_init[:, None] * safe_init[None, :]
    cov_both = jnp.log1p(P0 / safe_init_outer) / (k * k)
    denom_i = jnp.where(positivity, k * safe_init, 1.0)[:, None]
    denom_j = jnp.where(positivity, k * safe_init, 1.0)[None, :]
    cov_mixed = P0 / (denom_i * denom_j)
    both_pos = positivity[:, None] & positivity[None, :]
    P0_a = jnp.where(both_pos, cov_both, cov_mixed)
    P0_a = 0.5 * (P0_a + P0_a.T)

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "a0": (ssp_priors["a0"].dims, np.asarray(a0_a)),
        "P0": (ssp_priors["P0"].dims, np.asarray(P0_a)),
    }
    data_vars.update(
        _transform_sigma_q_block(ssp_priors, family, n_states, safe_init, positivity, k)
    )
    data_vars.update(_transform_obs_block(ssp_priors, positivity, k))

    if "obs_idx" in ssp_priors:
        obs_idx_var = ssp_priors["obs_idx"]
        data_vars["obs_idx"] = (obs_idx_var.dims, np.asarray(obs_idx_var.values))

    pos_var = ssp_priors["positivity_idx"]
    data_vars["positivity_idx"] = (pos_var.dims, np.asarray(pos_var.values))

    return xr.Dataset(
        data_vars=data_vars,
        coords=dict(ssp_priors.coords),
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
    as representatives -- ``state[1]`` stands in for the shared block at
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
    ssp_priors: xr.Dataset,
    family: str,
    n_states: int,
    safe_init: jnp.ndarray,
    positivity: jnp.ndarray,
    k: float,
) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    """Dispatch sigma_q natural to a-space conversion on the chosen prior family.

    See the module docstring for the per-family schema. Both branches reuse
    ``_moment_match_sigma`` for sigma-like quantities and
    ``_resolve_sigma_alignment`` for the per-state vs compressed ``(2,)``
    shape rules.
    """
    if family == "truncated_normal":
        return _transform_sigma_q_truncated_normal(
            ssp_priors, n_states, safe_init, positivity, k
        )
    if family == "beta":
        return _transform_sigma_q_beta(
            ssp_priors, n_states, safe_init, positivity, k
        )
    raise ValueError(
        f"unknown sigma_q_family: {family!r}; expected 'truncated_normal' or 'beta'"
    )


def _transform_sigma_q_truncated_normal(
    ssp_priors: xr.Dataset,
    n_states: int,
    safe_init: jnp.ndarray,
    positivity: jnp.ndarray,
    k: float,
) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    for name in ("sigma_q_loc_prior", "sigma_q_scale_prior"):
        if name not in ssp_priors:
            raise ValueError(f"sigma_q_family='truncated_normal' requires {name!r}")

    loc_nat = jnp.asarray(ssp_priors["sigma_q_loc_prior"].values)
    scale_nat = jnp.asarray(ssp_priors["sigma_q_scale_prior"].values)
    if loc_nat.shape != scale_nat.shape:
        raise ValueError(
            "sigma_q_loc_prior and sigma_q_scale_prior must share the same shape; "
            f"got {loc_nat.shape} and {scale_nat.shape}"
        )

    ref_level, sigma_positivity = _resolve_sigma_alignment(
        loc_nat.shape, n_states, safe_init, positivity
    )
    sigma_dims = ssp_priors["sigma_q_loc_prior"].dims

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
        ("sigma_q_low_prior", "sigma_q_low_prior"),
        ("sigma_q_high_prior", "sigma_q_high_prior"),
    ):
        if src_name in ssp_priors:
            arr = jnp.asarray(ssp_priors[src_name].values)
            out[out_name] = (
                sigma_dims,
                np.asarray(_moment_match_sigma(arr, ref_level, sigma_positivity, k)),
            )

    return out


def _transform_sigma_q_beta(
    ssp_priors: xr.Dataset,
    n_states: int,
    safe_init: jnp.ndarray,
    positivity: jnp.ndarray,
    k: float,
) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    required = ("sigma_q_alpha_prior", "sigma_q_beta_prior", "sigma_q_scale_prior")
    for name in required:
        if name not in ssp_priors:
            raise ValueError(f"sigma_q_family='beta' requires {name!r}")

    alpha = jnp.asarray(ssp_priors["sigma_q_alpha_prior"].values)
    beta = jnp.asarray(ssp_priors["sigma_q_beta_prior"].values)
    scale_nat = jnp.asarray(ssp_priors["sigma_q_scale_prior"].values)
    if alpha.shape != beta.shape or alpha.shape != scale_nat.shape:
        raise ValueError(
            "sigma_q_alpha_prior, sigma_q_beta_prior, and sigma_q_scale_prior must share the same shape; "
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
    sigma_dims = ssp_priors["sigma_q_scale_prior"].dims

    mode_frac = (alpha - 1.0) / (alpha + beta - 2.0)
    mode_nat = scale_nat * mode_frac
    mode_a = _moment_match_sigma(mode_nat, ref_level, sigma_positivity, k)
    scale_a = jnp.where(sigma_positivity, mode_a / mode_frac, scale_nat)

    return {
        "sigma_q_alpha_prior": (sigma_dims, np.asarray(alpha)),
        "sigma_q_beta_prior": (sigma_dims, np.asarray(beta)),
        "sigma_q_scale_prior": (sigma_dims, np.asarray(scale_a)),
    }
