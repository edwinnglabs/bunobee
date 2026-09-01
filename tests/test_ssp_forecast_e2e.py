"""End-to-end SSP forecast test on the sparse dataset.

Covers the acceptance criteria of issue #28 -- the full path *fit -> build future design ->
forecast -> inspect draws* on real data, rather than the synthetic-posterior identity checks in
``tests/test_ssp_forecast.py`` (issue #26) and ``tests/test_build_forecast_design.py`` (issue #27).

The data is ``tests/fixtures/sparse_panel.csv``: an exact slice of the sparse dataset used by the
``ssp_05_extended_filter_multi_series`` prototype, ``playground/resource/sparse/sparse_df.csv``, which is
untracked (``*.csv`` is gitignored) and so unavailable in CI. The slice keeps the first ``N_STEPS + HORIZON`` dates, the four
series in ``SERIES``, and the columns ``outcome`` plus the ``MEDIA`` channels --
:class:`TestFixtureProvenance` re-derives it from the full file whenever that file is on hand, so the
committed copy cannot silently drift.

The fixture fits that panel with a deliberately tiny ``ssp_05``-style NUTS run: a 4-series / 120-step
training window, an ``intercept + weekly-dummy + media`` design, and a shared-state EKF likelihood whose
only sampled parameters are ``sigma_h`` and ``sigma_q``. That keeps the posterior geometry two-block and
low-dimensional, so 2 chains x (40 warmup + 100 draws) converge in roughly ten seconds -- no multi-minute
MCMC and no committed posterior blob.

Criteria, one class each:

1. Output shape is ``(sample, H, n_series)`` with the expected coords.
2. Forecast interval width is nondecreasing in ``time``.
3. The held-out tail of length ``H`` sits inside the 5-95% band for the majority of
   ``(time, series)`` cells.
4. ``forecast_ssp`` with a fixed ``seed`` is bitwise-stable across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from bunobee.models.ssp import build_forecast_design, forecast_ssp, transform_to_ekf_st
from bunobee.regression import make_peridoic_dummies

SPARSE_PANEL = Path(__file__).resolve().parent / "fixtures" / "sparse_panel.csv"
SPARSE_SOURCE = Path(__file__).resolve().parents[1] / "playground" / "resource" / "sparse" / "sparse_df.csv"

# Training window, horizon, and panel slice. Small on purpose: the point is the wiring, not the fit.
N_STEPS = 120
HORIZON = 14
PERIOD = 7
EXPONENT = 1.0
SERIES = ["region_1", "region_12", "region_14", "region_5"]
MEDIA = ["ch_1", "ch_3", "ch_7"]
RESPONSE = "outcome"

# Design layout: [intercept] + [weekly dummies] + [media]. The three blocks exercise the three
# resolution rules of build_forecast_design (constant carry-forward, periodic continuation, covariate).
N_MEDIA = len(MEDIA)
N_SERIES = len(SERIES)
N_DUMMIES = PERIOD - 1
N_STATES = 1 + N_DUMMIES + N_MEDIA
MEDIA_COL_0 = 1 + N_DUMMIES
WEEKLY_SPEC = {"columns": slice(1, PERIOD), "period": PERIOD, "drop_first": True}

NUM_WARMUP = 40
NUM_SAMPLES = 100
NUM_CHAINS = 2
N_POSTERIOR = NUM_CHAINS * NUM_SAMPLES


@dataclass(frozen=True)
class SparseFit:
    """Everything the end-to-end assertions need from one fit of the sparse panel."""

    idata: xr.Dataset
    Z_future: np.ndarray
    Z_future_truth: np.ndarray
    y_holdout: np.ndarray


def _slice_sparse(df: pd.DataFrame) -> pd.DataFrame:
    """Cut the sparse dataset down to the series, dates, and columns this test needs."""
    dates = sorted(df.loc[df["dma"].isin(SERIES), "date"].unique())[: N_STEPS + HORIZON]
    sliced = df[df["dma"].isin(SERIES) & df["date"].isin(dates)]
    return sliced[["date", "dma", RESPONSE, *MEDIA]].sort_values(["date", "dma"]).reset_index(drop=True)


def _to_panels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pivot the long panel into ``(y, media)`` arrays indexed ``(time, series[, channel])``."""
    dates = sorted(df["date"].unique())

    def _pivot(column: str) -> np.ndarray:
        return df.pivot(index="date", columns="dma", values=column).reindex(index=dates, columns=SERIES).values

    y = np.asarray(_pivot(RESPONSE), dtype=float)
    media = np.stack([np.asarray(_pivot(name), dtype=float) for name in MEDIA], axis=-1)
    return y, media


def _build_design(media: np.ndarray) -> np.ndarray:
    """Assemble ``[intercept | weekly dummies | media]`` over the full date range."""
    n_all = media.shape[0]
    dummies = np.asarray(make_peridoic_dummies(n_all, period=PERIOD, drop_first=True), dtype=float)
    intercept = np.ones((n_all, N_SERIES, 1))
    dummies_b = np.broadcast_to(dummies[:, None, :], (n_all, N_SERIES, N_DUMMIES))
    return np.concatenate([intercept, dummies_b, media], axis=-1)


def _ekf_prior(sigma_q_scale: np.ndarray, positivity: np.ndarray) -> xr.Dataset:
    """Natural-scale ``ssp_05``-style prior, transformed to EKF a-space."""
    a0 = np.concatenate([[1.0], np.zeros(N_DUMMIES), np.full(N_MEDIA, 0.1)])
    P0_diag = np.concatenate([[0.25], np.full(N_DUMMIES, 0.05), np.full(N_MEDIA, 0.001)])
    prior = xr.Dataset(
        {
            "a0": (("state",), a0),
            "P0": (("state", "state_dual"), np.diag(P0_diag)),
            "positivity": (("state",), positivity),
            "sigma_q_alpha_prior": (("state",), np.full(N_STATES, 2.0)),
            "sigma_q_beta_prior": (("state",), np.full(N_STATES, 10.0)),
            "sigma_q_scale_prior": (("state",), sigma_q_scale),
        }
    ).assign_attrs({"sigma_q_family": "beta"})
    return transform_to_ekf_st(prior, exponent=EXPONENT)


def _run_nuts(prior_ekf: xr.Dataset, y_train: np.ndarray, Z_train: np.ndarray, positivity: np.ndarray) -> xr.Dataset:
    """Fit the shared-state EKF likelihood and package the posterior the way ``forecast_ssp`` reads it."""
    import jax.numpy as jnp
    import numpyro
    from jax import random
    from numpyro import distributions as dist
    from numpyro.infer import MCMC, NUTS

    from bunobee.models.ssp.kalman_1d_st_ekf import kalman_filter_1d_ekf_st

    a0 = jnp.asarray(prior_ekf["a0"].values)
    P0 = jnp.asarray(prior_ekf["P0"].values)
    positivity_j = jnp.asarray(positivity)
    sigma_q_alpha = jnp.asarray(prior_ekf["sigma_q_alpha_prior"].values)
    sigma_q_beta = jnp.asarray(prior_ekf["sigma_q_beta_prior"].values)
    sigma_q_scale = jnp.asarray(prior_ekf["sigma_q_scale_prior"].values)
    sdy = jnp.asarray(y_train.std(axis=0))
    Z_j = jnp.asarray(Z_train)
    y_j = jnp.asarray(y_train)

    def model():
        with numpyro.plate("series", N_SERIES):
            sigma_h_raw = numpyro.sample("sigma_h_raw", dist.Beta(2.0, 10.0))
        sigma_h = numpyro.deterministic("sigma_h", sigma_h_raw * sdy)
        sigma_q_raw = numpyro.sample("sigma_q_raw", dist.Beta(sigma_q_alpha, sigma_q_beta))
        sigma_q = numpyro.deterministic("sigma_q", sigma_q_raw * sigma_q_scale)
        lp, at, _, _, _, _ = kalman_filter_1d_ekf_st(
            a0=a0,
            P0=P0,
            Z=Z_j,
            sigma_h=sigma_h,
            sigma_q=sigma_q,
            y=y_j,
            logp=True,
            exponent=EXPONENT,
            positivity=positivity_j,
        )
        numpyro.factor("lp", lp)
        numpyro.deterministic("at", at)

    mcmc = MCMC(
        NUTS(model),
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(0))
    samples = mcmc.get_samples(group_by_chain=True)

    return xr.Dataset(
        {
            "at": (("chain", "draw", "step", "state"), np.asarray(samples["at"])),
            "sigma_q": (("chain", "draw", "state"), np.asarray(samples["sigma_q"])),
            "sigma_h": (("chain", "draw", "series"), np.asarray(samples["sigma_h"])),
            "positivity": (("state",), positivity),
        }
    ).assign_attrs({"exponent": EXPONENT})


@pytest.fixture(scope="module")
def fit() -> SparseFit:
    """Fit the sparse panel once, then continue its design ``HORIZON`` steps past the sample."""
    pytest.importorskip("numpyro")

    y_raw, media_raw = _to_panels(pd.read_csv(SPARSE_PANEL, parse_dates=["date"]))

    # Per-series mean normalization, fitted on the training window only so the holdout stays out of sample.
    y = y_raw / y_raw[:N_STEPS].mean(axis=0)
    media = media_raw / media_raw[:N_STEPS].mean(axis=0)

    Z_all = _build_design(media)
    Z_train = Z_all[:N_STEPS]
    y_train = y[:N_STEPS]

    positivity = np.zeros(N_STATES, dtype=bool)
    positivity[MEDIA_COL_0:] = True  # media states get the exp(exponent * a) map; level/seasonal stay linear
    sigma_q_scale = np.concatenate([[0.05], np.full(N_DUMMIES, 0.02), np.full(N_MEDIA, 0.01)])

    idata = _run_nuts(_ekf_prior(sigma_q_scale, positivity), y_train, Z_train, positivity)

    Z_future = build_forecast_design(
        Z_train,
        HORIZON,
        periodic=WEEKLY_SPEC,
        covariates_future={MEDIA_COL_0 + i: media[N_STEPS:, :, i] for i in range(N_MEDIA)},
    )

    return SparseFit(
        idata=idata,
        Z_future=Z_future,
        Z_future_truth=Z_all[N_STEPS:],
        y_holdout=y[N_STEPS:],
    )


def _band(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """5-95% band over the leading ``sample`` axis."""
    lo, hi = np.quantile(samples, [0.05, 0.95], axis=0)
    return lo, hi


# ---------------------------------------------------------------------------
# The committed fixture is a verbatim slice of the sparse dataset
# ---------------------------------------------------------------------------


class TestFixtureProvenance:
    def test_panel_has_the_expected_grid(self):
        panel = pd.read_csv(SPARSE_PANEL, parse_dates=["date"])

        assert list(panel.columns) == ["date", "dma", RESPONSE, *MEDIA]
        assert sorted(panel["dma"].unique()) == sorted(SERIES)
        assert panel["date"].nunique() == N_STEPS + HORIZON
        assert len(panel) == (N_STEPS + HORIZON) * N_SERIES
        assert panel.notna().all().all()

    def test_matches_the_sparse_dataset_when_available(self):
        # The full dataset is gitignored, so this only runs where a working copy has it.
        if not SPARSE_SOURCE.exists():
            pytest.skip(f"sparse dataset not available at {SPARSE_SOURCE}")

        expected = _slice_sparse(pd.read_csv(SPARSE_SOURCE, parse_dates=["date"]))
        committed = pd.read_csv(SPARSE_PANEL, parse_dates=["date"])

        pd.testing.assert_frame_equal(committed, expected, check_exact=False, rtol=1e-12)


# ---------------------------------------------------------------------------
# The future design itself — build_forecast_design must reconstruct the real future Z
# ---------------------------------------------------------------------------


class TestFutureDesign:
    def test_matches_the_true_future_design(self, fit: SparseFit):
        assert fit.Z_future.shape == (HORIZON, N_SERIES, N_STATES)
        np.testing.assert_allclose(fit.Z_future, fit.Z_future_truth, rtol=1e-12, atol=1e-12)

    def test_intercept_and_weekly_blocks_continue_the_cycle(self, fit: SparseFit):
        np.testing.assert_allclose(fit.Z_future[:, :, 0], 1.0)
        # Every forecast step lands on exactly one weekday, except the dropped reference day.
        weekly = fit.Z_future[:, 0, 1:PERIOD]
        assert set(np.unique(weekly)) <= {0.0, 1.0}
        np.testing.assert_array_equal(np.unique(weekly.sum(axis=1)), np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# Criterion 1 — output shape is (sample, H, n_series) with the expected coords
# ---------------------------------------------------------------------------


class TestOutputContract:
    def test_dims_sizes_and_coords(self, fit: SparseFit):
        out = forecast_ssp(fit.idata, fit.Z_future, noise_embed=True, seed=42)

        for name in ("forecast_samples", "mu_samples", "eps_samples"):
            assert out[name].dims == ("sample", "time", "series")
            assert out[name].shape == (N_POSTERIOR, HORIZON, N_SERIES)
        assert dict(out.sizes) == {"sample": N_POSTERIOR, "time": HORIZON, "series": N_SERIES}
        np.testing.assert_array_equal(out.coords["sample"].values, np.arange(N_POSTERIOR))
        np.testing.assert_array_equal(out.coords["time"].values, np.arange(HORIZON))
        np.testing.assert_array_equal(out.coords["series"].values, np.arange(N_SERIES))

    def test_draws_are_finite_and_on_the_response_scale(self, fit: SparseFit):
        out = forecast_ssp(fit.idata, fit.Z_future, noise_embed=True, seed=42)

        forecast = out["forecast_samples"].values
        assert np.isfinite(forecast).all()
        # y is mean-normalized per series over the training window, so the median forecast must sit in
        # the same order of magnitude as the holdout rather than collapsing to zero or blowing up.
        median = np.median(forecast, axis=0)
        assert 0.2 < median.mean() < 5.0 * fit.y_holdout.mean()

    def test_idata_positivity_and_exponent_drive_the_forecast(self, fit: SparseFit):
        # The posterior carries positivity / exponent, so a conflicting argument must not change anything.
        baseline = forecast_ssp(fit.idata, fit.Z_future, seed=7)
        overridden = forecast_ssp(
            fit.idata,
            fit.Z_future,
            exponent=0.5,
            positivity=np.zeros(N_STATES, dtype=bool),
            seed=7,
        )
        xr.testing.assert_identical(baseline, overridden)


# ---------------------------------------------------------------------------
# Criterion 2 — forecast interval width is nondecreasing in time
# ---------------------------------------------------------------------------


class TestIntervalWidening:
    """Widening is a property of the horizon, so it is measured on a frozen design.

    On the real ``Z_future`` the band also moves with the weekly dummies and the media covariates,
    which vary step to step; freezing the design at its first future row isolates the random-walk
    variance growth that the forecast is supposed to produce.
    """

    @staticmethod
    def _frozen(fit: SparseFit) -> np.ndarray:
        return np.repeat(fit.Z_future[:1], HORIZON, axis=0)

    def test_variance_nondecreasing_under_a_frozen_design(self, fit: SparseFit):
        out = forecast_ssp(fit.idata, self._frozen(fit), noise_embed=False, seed=0)

        var_t = out["mu_samples"].var(dim="sample").values  # (time, series)
        assert np.all(np.diff(var_t, axis=0) >= -1e-12)
        assert np.all(var_t[-1] > var_t[0])

    def test_band_width_grows_with_horizon(self, fit: SparseFit):
        # Measured on mu: the fitted sigma_h is an order of magnitude above the random-walk spread, so
        # an eps-inflated band is dominated by a horizon-flat noise term.
        out = forecast_ssp(fit.idata, self._frozen(fit), noise_embed=False, seed=0)

        lo, hi = _band(out["forecast_samples"].values)
        width = hi - lo  # (time, series)
        assert np.all(width[-1] > width[0])
        # Monte-Carlo noise on a quantile can dent a single step; the trend must still be upward.
        assert np.all(width[-HORIZON // 2 :].mean(axis=0) > width[: HORIZON // 2].mean(axis=0))

    def test_observation_noise_widens_every_step(self, fit: SparseFit):
        frozen = self._frozen(fit)
        plain = forecast_ssp(fit.idata, frozen, noise_embed=False, seed=0)
        noisy = forecast_ssp(fit.idata, frozen, noise_embed=True, seed=0)

        var_plain = plain["forecast_samples"].var(dim="sample").values
        var_noisy = noisy["forecast_samples"].var(dim="sample").values
        assert np.all(var_noisy > var_plain)


# ---------------------------------------------------------------------------
# Criterion 3 — the held-out tail sits inside the 5-95% band for most cells
# ---------------------------------------------------------------------------


class TestHoldoutCoverage:
    def test_majority_of_cells_are_covered(self, fit: SparseFit):
        out = forecast_ssp(fit.idata, fit.Z_future, noise_embed=True, seed=42)

        lo, hi = _band(out["forecast_samples"].values)
        covered = (fit.y_holdout >= lo) & (fit.y_holdout <= hi)
        assert covered.mean() > 0.5, f"holdout coverage {covered.mean():.3f} of the 5-95% band"
        assert np.all(covered.mean(axis=0) > 0.5), f"per-series coverage {covered.mean(axis=0)}"

    def test_predictive_band_covers_more_than_the_mu_band(self, fit: SparseFit):
        out = forecast_ssp(fit.idata, fit.Z_future, noise_embed=True, seed=42)

        lo_y, hi_y = _band(out["forecast_samples"].values)
        lo_mu, hi_mu = _band(out["mu_samples"].values)
        covered_y = ((fit.y_holdout >= lo_y) & (fit.y_holdout <= hi_y)).mean()
        covered_mu = ((fit.y_holdout >= lo_mu) & (fit.y_holdout <= hi_mu)).mean()
        assert covered_y > covered_mu

    def test_median_forecast_beats_a_zero_forecast(self, fit: SparseFit):
        out = forecast_ssp(fit.idata, fit.Z_future, noise_embed=False, seed=42)

        median = np.median(out["mu_samples"].values, axis=0)
        assert np.abs(median - fit.y_holdout).mean() < np.abs(fit.y_holdout).mean()


# ---------------------------------------------------------------------------
# Criterion 4 — a fixed seed is bitwise-stable across runs
# ---------------------------------------------------------------------------


class TestSeedStability:
    def test_same_seed_reproduces_the_dataset(self, fit: SparseFit):
        first = forecast_ssp(fit.idata, fit.Z_future, noise_embed=True, seed=2024)
        # Consume the global RNG and run an unrelated forecast in between: forecast_ssp must depend on
        # nothing but (idata, Z_future, seed).
        np.random.default_rng().standard_normal(1000)
        forecast_ssp(fit.idata, fit.Z_future, noise_embed=True, seed=99)
        second = forecast_ssp(fit.idata, fit.Z_future, noise_embed=True, seed=2024)

        xr.testing.assert_identical(first, second)
        np.testing.assert_array_equal(first["forecast_samples"].values, second["forecast_samples"].values)

    def test_a_different_seed_moves_the_draws(self, fit: SparseFit):
        first = forecast_ssp(fit.idata, fit.Z_future, noise_embed=True, seed=2024)
        other = forecast_ssp(fit.idata, fit.Z_future, noise_embed=True, seed=2025)

        assert not np.allclose(first["forecast_samples"].values, other["forecast_samples"].values)
        # ... while the posterior-implied band is stable to the sampling seed.
        lo_a, hi_a = _band(first["forecast_samples"].values)
        lo_b, hi_b = _band(other["forecast_samples"].values)
        np.testing.assert_allclose(hi_a - lo_a, hi_b - lo_b, rtol=0.35)
