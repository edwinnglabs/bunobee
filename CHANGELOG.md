# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [v0.0.6] - 2026-10-30

Makes the DLT hyper-parameter tuner usable downstream (regressors, a configurable search space, a documented
WBIC approximation), adds conditional DLT forecasts, pluggable target transforms and sigma priors for the M5
MAP fitters, and `make_periodic_dummies`.

### Added

- `search_space` argument on `hyper_tuning_dlt_with_wbic` — `{name: (low, high)}` or `{name: (low, high, prior)}`
  with prior `"log-uniform"` or `"uniform"`. A name left out keeps its default range and prior, as does a prior
  left out (`lev_sm` and `slp_sm` log-uniform, `theta` uniform); an unknown name raises `ValueError`. With no
  search space the tuner searches the same space as before. The defaults are in `DEFAULT_SEARCH_SPACE`, and
  `resolve_search_space` merges a search space into them.

- `conditional_response` argument on DLT `make_inference` — actuals past the train end update the latent
  level and slope (NaN entries are forecast as usual); length and shape are validated.
- `tune` optional-dependency extra — `pip install bunobee[tune]` installs `scikit-optimize` for
  `hyper_tuning_dlt_with_wbic`.
- `make_periodic_dummies` in `bunobee.regression` — periodic dummy design matrix with a `phase` argument
  (the cycle position of step 0, e.g. the first day's weekday) and an `intercept` flag that prepends a
  column of ones. `build_forecast_design`, the M5 NUTS fit and the M5 run script all build their weekly
  dummies with it.
- `bunobee.models.m5.transforms` — a `Transform` protocol for the target of the M5 MAP fitters (`fit`
  per-series parameters on the fit days, `forward`, and `inverse` with the back-transform bias correction)
  and `LogTransform`, the previously hard-coded `log(max(sales, 0.1) / mean)` target. `fit_one_series_opt`
  and `fit_batch_series_opt` take a `transform` argument (default `LogTransform()`) and return its
  parameters and the transform; the MAP predictors invert the fit's transform. Forecasts are unchanged.
- `PowerTransform` in `bunobee.models.m5.transforms` — a Box-Cox or Yeo-Johnson target with λ fitted per
  series by scikit-learn's `PowerTransformer` on `max(sales, floor) + shift`, bounded to [−1, 2] by default,
  and standardised. Its inverse is the mean of the sales a Gaussian forecast implies, censored to the series'
  range from 0 to its largest fit-day sales, so forecasts are never negative.
- `SIGMA_H_BOUNDS` and `SIGMA_Q_BOUNDS` in `bunobee.models.m5.m5_ssp_optim` — the bounds of the uniform
  priors on the noise and the drifts of the MAP fitters, which use them. Their values are unchanged.
- `SigmaPrior` in `bunobee.models.m5.m5_ssp_optim` — the prior of a sigma of the MAP fitters: uniform (bounded),
  half-normal or log-normal. `fit_one_series_opt` and `fit_batch_series_opt` take `sigma_h_prior` and
  `sigma_q_prior` arguments, by default `SIGMA_H_PRIOR` = U(0.1, 0.5) and `SIGMA_Q_PRIOR` = U(0.01, 0.1), the
  previous priors, and start each sigma at its prior's mean (the previous midpoints for the defaults).
  `fit_one_series_opt` also returns `final_loss`, the loss of the returned parameters.

### Changed

- The `m5` extra installs `scikit-learn`, which `PowerTransform` fits with. It used to arrive only through
  `scikit-optimize`.
- `hyper_tuning_dlt_with_wbic` takes `y` plus optional `regression_scheme`, `covariates_df` and
  `mcmc_run_args` instead of a `data` dict, so models with regressors can be tuned. It returns a dict of
  the tuned `lev_sm`, `slp_sm` and `theta`, with the raw `skopt` result under `result`, and logs through
  the `bunobee` logger instead of printing.
- `hyper_tune` and the README document that the tuner's score samples the ordinary posterior, not the
  posterior tempered at `beta = 1 / log(n)`, so it is an approximation of WBIC used to rank candidates. A true
  tempered WBIC is tracked in [#135](https://github.com/edwinnglabs/bunobee/issues/135).

### Deprecated

- `make_peridoic_dummies` — misspelled name, kept as an alias of `make_periodic_dummies` that emits a
  `DeprecationWarning`.

### Fixed

- `fit_batch_series_opt` returns the parameters that achieved its best loss. It kept the parameters one Adam step
  after them, so a sigma its uniform prior's bound held came back just past the bound (e.g. `sigma_h` 0.5085
  under U(0.1, 0.5)). `fit_one_series_opt` returned the parameters after its last step, whose loss it never
  computed; it now returns those of its lowest loss too. The returned sigmas lie inside their prior's bounds.
  This changes the M5 forecasts slightly, as the capped `sigma_h` and its lognormal correction shrink: `ssp_dow`'s
  validation WRMSSE moves from 0.93487 to 0.94808 on the 3,000-series sample and from 1.08136 to 1.10175 on
  the full run.
- `hyper_tuning_dlt_with_wbic` runs again: it called `run_dlt_model` with arguments that no longer exist
  and raised a `TypeError`.
- DLT `make_inference` documents that with `noise_embed=False` (the default) the forecast band reflects
  only state-propagated uncertainty, zero at the first step, and is narrower than the predictive
  distribution of `y`; set `noise_embed=True` for intervals that cover future observations. New tests
  confirm the forecast uses no state past the train end.

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

[Unreleased]: https://github.com/edwinnglabs/bunobee/compare/v0.0.6...HEAD
[v0.0.6]: https://github.com/edwinnglabs/bunobee/compare/v0.0.5...v0.0.6
[v0.0.5]: https://github.com/edwinnglabs/bunobee/compare/v0.0.4...v0.0.5
[v0.0.4]: https://github.com/edwinnglabs/bunobee/releases/tag/v0.0.4
