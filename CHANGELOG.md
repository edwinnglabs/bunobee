# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- PEP 561 `py.typed` marker so downstream type checkers honor the package's type hints.
- `extend_states_prior_smoothed` — exact KF-forward + RTS-backward anchor extension that
  fuses every anchor per state, for multi-anchor channels the nearest-anchor heuristic only
  approximates.

### Changed

- Renamed `extend_states_prior` to `extend_states_prior_nearest` so the nearest-anchor
  heuristic and the new `extend_states_prior_smoothed` are self-describing.
- Both prior extensions now extend one step backward to `t = -1` and emit the initial-state
  moments `a0` / `P0` over dims `(state,)`, so their output satisfies the complete-prior
  contract and can be promoted to an `SspPrior`. The `(time, state)` rectangle keeps its
  shape. Unanchored states get `a0 = 0`, `P0 = inf`; an `a0` / `P0` already present on the
  input is overwritten rather than passed through.

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

[Unreleased]: https://github.com/edwinnglabs/bunobee/compare/v0.0.4...HEAD
[v0.0.4]: https://github.com/edwinnglabs/bunobee/releases/tag/v0.0.4
