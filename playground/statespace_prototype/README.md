# State-Space Prototype (`ssp`) notebooks

A staged series of prototypes for the state-space engine in `bunobee.models.ssp`. Each notebook builds on the
previous one, moving from a single-series linear Kalman filter to a multi-series extended Kalman filter (EKF), and
demonstrates how to assemble, disclose, and visualise **time-point priors** on the latent states.

## The `ssp_v1` … `ssp_v6` progression

- **`ssp_v1` — First Prototype** (`kalman_1d`): bare single-series linear Kalman filter, the starting point.
- **`ssp_v2` — Kalman Filter with Time-Point Priors and Positive Coefficients** (`kalman_1d`): time-point priors via
  `construct_states_prior`; positivity-constrained coefficients.
- **`ssp_v3` — Extended Kalman Filter, a Revisit on the Non-Linear State-Space Model** (`kalman_1d_ekf`): log-space
  (multiplicative) states through `transform_to_ekf`; `λ_t = exp(k·a_t)`.
- **`ssp_v4` — Linear Kalman Filter, Fit with Multiple Time-Series** (`kalman_1d_st`): multi-series (`_st`) linear
  filter; time-point priors derived from a beta-prior table.
- **`ssp_v5` — Extended Kalman Filter, Fit with Multiple Time-Series** (`kalman_1d_ekf_st`): multi-series EKF via
  `transform_to_ekf_st`; positivity on the natural scale.
- **`ssp_v6_forecast` — Extended Kalman Filter, Multi-Series Forecast** (`forecast_ssp`): the `v5` fit with the last
  28 days held out, its design continued by `build_forecast_design`, and the predictive path drawn by `forecast_ssp`
  — fan chart with 50% / 90% bands, `plot_states` on the in-sample path, and the fan faceted over all series. This is
  the only notebook here **committed with its rendered outputs** (`"keep_output": true` in the notebook metadata, which
  `nbstripout` honours), because the plots are the point.

`v2`/`v3` are single-series and disclose the ground-truth latent state over a few random windows. `v4`/`v5` are
multi-series and build their priors from a time-point beta table (`use_time_point_prior=True`), anchoring the media
states every four weeks.

## Prior construction & visualisation tools

All live in `bunobee.models.ssp` (re-exported from `.prior` / `.posterior` / `.plotting` / `.transforms` /
`.simulation`):

- **`construct_states_prior(...)`** (in `.simulation`, simulation-only — requires a ground-truth `true_states` that
  isn't available outside a synthetic/prototype setting) — assembles an `xarray.Dataset` prior with `a_obs` / `P_obs`
  over `(time, state)` and a `positivity` mask, disclosing the ground truth over `n_periods` random windows of
  `n_points` consecutive steps. Undisclosed steps carry `inf` variance (zero precision → the pure filter passes
  through untouched).
- **`SspPrior`** (in `.prior`) — frozen wrapper around a *complete*, filter-ready prior `xr.Dataset`. Validates
  `a0` / `P0` / `a_obs` / `P_obs` / `positivity` plus `time` / `state` coordinates at construction time, regardless
  of how the dataset was built; exposes each as a property alongside `.dataset` for interop with `xr.merge`,
  `az.InferenceData`, and the Kalman filter functions.
- **`disclosed_idx(ssp_priors)`** — derives the disclosure timesteps directly from `P_obs` (any step with at least one
  finite-variance state). Replaces the previously stored `obs_idx` variable, which was redundant with `P_obs`.
- **`extend_states_prior_nearest(ssp_priors, Q)`** — fills the `inf` (undisclosed) steps of each anchored state with the
  driftless random-walk marginal `a_obs[t] = a*`, `P_obs[t] = P* + |t−t*|·Q`, spread forward and backward from the
  nearest anchor. Grows each anchor into a symmetric variance cone; states with no anchor stay `inf`, and `Q → inf`
  recovers the anchors-only prior. Exact only for an isolated channel — feed the result into the augmented-measurement
  step, not as a posterior.
- **`extend_states_prior_smoothed(ssp_priors, Q)`** — the exact multi-anchor counterpart: drives `kalman_filter_1d` in
  extension mode (`Z = 0`) then `kalman_rts_smoother_1d`, fusing *every* anchor per state by inverse-variance weighting.
  For a single-anchor channel it matches `extend_states_prior_nearest` to numerical noise; for multiple anchors it blends
  means and tightens the variance (never wider). States with no anchor stay `inf`. See
  [`ssp_extend_prior_multi_anchor.ipynb`](./ssp_extend_prior_multi_anchor.ipynb).
- **`plot_prior_heatmap(ssp_priors, quantity="both")`** — renders the prior as a **states × time** heatmap. Rows are
  latent states, columns are timesteps; coloured cells are disclosed anchors (finite variance) and grey cells are
  undisclosed (`inf` variance). `quantity` selects `"mean"` (`a_obs`), `"var"` (`P_obs`), or `"both"`.
- **`plot_states(posterior, dates, state_labels, ...)`** — posterior quantile ribbons for the filtered / smoothed
  states (or EKF intensities), optionally overlaying disclosure anchors via `obs_idx=disclosed_idx(ssp_priors)`.
- **`transform_to_ekf` / `transform_to_ekf_st`** — reparameterise a natural-scale prior into EKF (`a`-space) form for
  the `_ekf` filters (`v3` / `v5`).
- **`validate_prior(ssp_priors)`** — contract check on the prior dataset (called inside `construct_states_prior`).

Each notebook now includes a short *"Time-point prior at a glance"* block that prints `disclosed_idx(...)` and draws
`plot_prior_heatmap(..., quantity="both")` right after the prior is assembled. In `v4`/`v5` the default config sets
`use_time_point_prior=False`, so the heatmap is fully grey — flip that flag to populate the anchor windows.

## Standalone demo — `ssp_extend_prior`

`ssp_extend_prior.ipynb` is a self-contained demo of `extend_states_prior_nearest`. It (1) builds a disclosed prior with
`construct_states_prior`, (2) extends the sparse anchors along the two-sided random walk, and (3) overlays the
**anchors-only** vs. **random-walk-extended** prior in a single `plot_states` figure — the extension shows up as a
continuous variance cone that pinches to each anchor and widens with lag, while the anchorless intercept stays `inf`.

## Running the notebooks

The notebooks require the `bunobee` package and its JAX / NumPyro stack (`requires-python >= 3.11`). From the repo
root:

```bash
pip install -e .          # install bunobee into your environment
jupyter lab               # then open playground/statespace_prototype/ssp_v*.ipynb
```

`v2`/`v3` are self-contained (they synthesise their own data). `v4`/`v5`/`v6` expect a multi-series dataset (and, for
`v4`/`v5`, a time-point beta table) loaded in their data cells.
