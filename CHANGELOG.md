# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [v0.0.5] - 2026-10-01

Adds SSP forecasting, prior fusion and smoothed prior extension, and a packaged M5 dataset
with forecast-accuracy guards.

### Added

- `forecast_ssp` — multi-series SSP predictive draws from a fitted posterior. Launches from
  the last filtered state, propagates a driftless random walk forward, and returns
  `forecast_samples` / `mu_samples` (and `eps_samples` with `noise_embed`) over
  `(sample, time, series)`. `method="filter"` propagates the future state through the filter
  itself, so missing future observations become pure predict steps.
- `build_forecast_design` — continues a training design matrix past the end of the sample,
  producing the `Z_future` array `forecast_ssp` consumes.
- `mask` argument on `kalman_filter_1d_st` and `kalman_filter_1d_ekf_st` — an opt-in
  predict-only step wherever an observation is missing. `log_p` scores only observed entries;
  `mask=None` (the default) leaves outputs bitwise identical.
- `plot_prior_density` — per-state density panel for SSP priors.
- `bunobee.datasets` with `load_m5_aggregate()` — the aggregated M5 panel shipped as package
  data.
- Forecast-accuracy guards on the M5 aggregate for the SSP and DLT forecast paths, gated
  behind a registered `slow` pytest marker.
- Packaging CI (build, `twine check`, clean-venv install) and a version-driven publish
  workflow: `.devN` / pre-release versions go to Test PyPI, final versions to PyPI after
  approval. See `RELEASE.md`.
- README rewritten as the PyPI landing page, with an SSP forecast quickstart; new SSP
  notebooks covering forecasting, `combine_states_priors` and multi-anchor prior extension.
- PEP 561 `py.typed` marker so downstream type checkers honor the package's type hints.
- `extend_states_prior_smoothed` — exact KF-forward + RTS-backward anchor extension that
  fuses every anchor per state, for multi-anchor channels the nearest-anchor heuristic only
  approximates. Seeds the leading (pre-series) state with true zero precision (`P0 = inf`),
  the same "no information" encoding `P_obs` already uses, rather than a large-but-finite
  proxy; `kalman_filter_1d` and `kalman_rts_smoother_1d` handle that literal `inf` exactly, so
  the result is correct at JAX's actual default precision (float32), not only under
  `jax_enable_x64=True`.
- `combine_states_priors` — closed-form inverse-variance fusion of two independent states
  priors over the same `time` / `state` grid, returning a validated `SspPrior`. Precision
  adds, so `N(a, P)` fused with itself is `N(a, P/2)` and a tight prior dominates a loose
  one. `P_obs = inf` is zero precision and drops out; a step undisclosed on both sides stays
  undisclosed rather than going `NaN`. Mismatched coords, mismatched `positivity`, and a
  full-covariance `P0` raise rather than aligning or guessing; non-moment variables and
  attrs (`sdy`, the `sigma_q` block, ...) are taken from the left operand. Fusion assumes
  the two priors carry disjoint evidence, and is a product of densities, not a mixture.

### Changed

- `SspPrior` is now load-bearing across the SSP prior API. `extend_states_prior_nearest` and
  `extend_states_prior_smoothed` return an `SspPrior` instead of a bare `xr.Dataset` (values
  unchanged), and `validate_prior`, `disclosed_idx`, `transform_to_ekf`, and
  `transform_to_ekf_st` accept an `SspPrior` and an `xr.Dataset` interchangeably. The a-space
  transform outputs stay plain `xr.Dataset`s, as does `construct_states_prior`.
- `SspPrior.__contains__` now delegates to the wrapped dataset, so `"time" in prior` matches
  `"time" in prior.dataset` for coordinates as well as data vars (it previously fell back to
  iteration, which yields data-var names only). Added `SspPrior.from_dataset` /
  `SspPrior.to_dataset`, and documented the wrapper's two limits: `xr.merge` rejects it, and no
  xarray operation preserves it — the invariant holds at function boundaries only.
- Renamed `extend_states_prior` to `extend_states_prior_nearest` so the nearest-anchor
  heuristic and the new `extend_states_prior_smoothed` are self-describing.
- Both prior extensions now extend one step backward to `t = -1` and emit the initial-state
  moments `a0` / `P0` over dims `(state,)`, so their output satisfies the complete-prior
  contract and can be promoted to an `SspPrior`. The `(time, state)` rectangle keeps its
  shape. Unanchored states get `a0 = 0`, `P0 = inf`. The derived moments are a placeholder,
  so an `a0` / `P0` already present on the input is preserved rather than overwritten — pass
  `overwrite_init=True` to let the placeholder win instead.
- The M5 prediction path now runs on the shared SSP forecast core.

### Fixed

- `extend_states_prior_smoothed` no longer collapses pre-first-anchor variances toward zero
  under float32 (JAX's default precision); see the `P0 = inf` seeding under Added.
- The positivity mask is now disambiguated between the EKF and linear filters.
- Extending a states prior preserves an `a0` / `P0` already present on the input.

## [v0.0.4]

First public release on PyPI. `bunobee` is a time-series forecasting engine built on JAX
and NumPyro. Versions `0.0.1`–`0.0.3` were internal / Test PyPI pre-releases, so this entry
consolidates the state-space (SSP) engine work that landed across them.

### Added

- `SspPrior` dataclass for structured, validated SSP prior specification.
- `extend_states_prior` for two-sided random-walk anchor extension of state priors.
- `notebook` optional-dependency extra — `pip install bunobee[notebook]` (houses `ipywidgets`).
- Package metadata: project URLs, trove classifiers, and keywords.
- Unit-test coverage for the SSP Kalman filters; CI running `pytest` + `black`.
- Trusted-publishing (OIDC) GitHub Actions workflow to publish releases to PyPI.

### Changed

- Standardized prior/posterior naming on the arviz convention.
- Hardened the SSP time-point prior contract and renamed the positivity handling for clarity.
- Declared the license as SPDX `MIT`.

### Removed

- Unused `setuptools-scm` from build requirements.

[Unreleased]: https://github.com/edwinnglabs/bunobee/compare/v0.0.5...HEAD
[v0.0.5]: https://github.com/edwinnglabs/bunobee/compare/v0.0.4...v0.0.5
[v0.0.4]: https://github.com/edwinnglabs/bunobee/releases/tag/v0.0.4
