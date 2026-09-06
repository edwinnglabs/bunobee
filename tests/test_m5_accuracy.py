"""Forecast-accuracy guard on the aggregated M5 panel.

This module holds the hermetic benchmark behind issue #59: a small, dense, deterministic panel that a
broken forecast can be scored against. Issue #60 lands the fixture and its provenance guard; issue #61
lands the scoring harness on top of it — the fit, the forecast, the metrics, and the per-series table;
issue #62 lands the three thresholds that turn the measurement into a guard, so a quality regression
now fails the suite instead of merely printing a worse number.

The data is ``bunobee.datasets.load_m5_aggregate()``, backed by the packaged
``src/bunobee/datasets/data/m5_aggregate.csv``, built by
``playground/m5_prototype/m5_make_accuracy_fixture.py`` from the M5 Forecasting — Accuracy competition
dump. That dump is 980 MB and untracked (``*.csv`` / ``*.nc`` are gitignored), so CI cannot see it; the
committed file is a derived aggregate of it — 3 states x 3 categories plus a national ``TOTAL``, summed
over all 30 490 item-store series, for the last ``N_STEPS + HORIZON`` observed days — not a
redistribution of the raw competition data. It is packaged rather than kept test-only so the same panel
is one import away in a demo notebook or a prototype script: ``from bunobee.datasets import
load_m5_aggregate``.

Aggregating is what makes the panel scoreable at all: raw M5 item-store series are intermittent and
zero-inflated, and MAPE on a series that spends half its life at zero means nothing. The aggregates run
in the thousands of units per day with no zeros anywhere, which the grid checks below assert rather than
assume.

:class:`TestFixtureProvenance` mirrors the class of the same name in ``tests/test_ssp_forecast_e2e.py``:
the grid checks always run, and the re-derivation check rebuilds the aggregate from the raw dump
whenever a working copy has it, so the committed copy cannot silently drift.

:class:`TestAccuracyHarness` scores the panel. The path is the one
``tests/test_ssp_forecast_e2e.py`` already walks — ``_ekf_prior`` -> :func:`transform_to_ekf_st` ->
``_run_nuts`` -> :func:`build_forecast_design` -> :func:`forecast_ssp` — behind a single
``scope="module"`` fixture, because the fit is expensive and the assertions are cheap. Everything about
it is pinned: the fixture, the 728/28 split, the NUTS configuration, and both PRNG seeds, so two
consecutive runs log the same numbers to the last digit.

:class:`TestAccuracyThresholds` is the tripwire itself: two generous absolute ceilings
(:data:`PANEL_MAPE_MAX`, :data:`SERIES_MAPE_MAX`) plus the scale-free
:data:`MAX_NAIVE_RATIO`. It closes with a test that breaks the forecast on purpose and asserts the
guard fires, because a tripwire nobody has tripped is not known to work.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from bunobee.datasets import M5_AGGREGATE_PATH, load_m5_aggregate
from bunobee.models.ssp import build_forecast_design, forecast_ssp, transform_to_ekf_st
from bunobee.regression import make_peridoic_dummies

logger = logging.getLogger(__name__)

# The accuracy guard runs a real NUTS fit, so it is materially slower than the rest of the suite. It
# stays in the default CI run; local iteration can deselect it with ``pytest -m "not slow"``.
pytestmark = pytest.mark.slow

_ROOT = Path(__file__).resolve().parents[1]
M5_PANEL = M5_AGGREGATE_PATH
M5_SOURCE = _ROOT / "playground" / "resource" / "m5-forecasting-accuracy"
M5_BUILDER = _ROOT / "playground" / "m5_prototype" / "m5_make_accuracy_fixture.py"

# Window and panel shape. Mirrors the builder's constants on purpose: declaring them again here means a
# builder change that never reached the committed CSV trips the grid checks instead of sliding through.
N_STEPS = 728
HORIZON = 28
N_DATES = N_STEPS + HORIZON

SERIES = [
    "CA_FOODS",
    "CA_HOBBIES",
    "CA_HOUSEHOLD",
    "TOTAL",
    "TX_FOODS",
    "TX_HOBBIES",
    "TX_HOUSEHOLD",
    "WI_FOODS",
    "WI_HOBBIES",
    "WI_HOUSEHOLD",
]
N_SERIES = len(SERIES)

# SHA-256 of the committed fixture, as printed by the builder on 2026-09-03. Pinning it makes a
# hand-edited value fail everywhere, not only where the raw dump is on hand to re-derive against.
FIXTURE_SHA256 = "afadf2e1f47b171767c2a7c2e8de80c9d812ec1b0347a36f2495a28af7cb0d35"

# The fixture is a committed test asset, not a data release; keep it small enough to stay one.
MAX_FIXTURE_BYTES = 500_000

# ---------------------------------------------------------------------------
# Harness configuration
# ---------------------------------------------------------------------------

# Design layout: ``[intercept | 6 weekly dummies]``, no covariates. Deliberately minimal — this module
# guards the forecasting engine, not feature engineering, so anything a richer design would add is
# noise between a break and the number that is supposed to reveal it.
PERIOD = 7
N_DUMMIES = PERIOD - 1
N_STATES = 1 + N_DUMMIES
WEEKLY_SPEC = {"columns": slice(1, PERIOD), "period": PERIOD, "drop_first": True}

# ``exp(EXPONENT * a)`` is the EKF's state map, but ``POSITIVITY`` is all-False here: on the
# mean-normalized scale the level and the weekly states are linear, so no state is ever exponentiated
# and the exponent only has to be carried consistently between the prior transform and the forecast.
EXPONENT = 1.0
POSITIVITY = np.zeros(N_STATES, dtype=bool)

# Natural-scale prior. The level starts at the training mean (1.0 after normalization) and the weekly
# deviations start at zero; the ``sigma_q`` scales let the level drift about ten times faster than the
# seasonal block, which is the usual ordering for a daily retail aggregate.
A0 = np.concatenate([[1.0], np.zeros(N_DUMMIES)])
P0_DIAG = np.concatenate([[0.25], np.full(N_DUMMIES, 0.05)])
SIGMA_Q_SCALE = np.concatenate([[0.05], np.full(N_DUMMIES, 0.005)])

# NUTS budget. Larger than the e2e smoke fit (2 x (40 + 100)) because this one has to produce a point
# forecast worth scoring, and bounded by the runtime budget: the likelihood is a 728-step scan over 10
# series, so one gradient costs ~20 ms and the iteration count is the whole cost model.
#
# ``MAX_TREE_DEPTH`` is the cheap half of that. The posterior is 17 well-identified parameters and the
# sampling phase settles at a constant 7 leapfrog steps per iteration -- tree depth 3, measured, well
# under the cap -- so a cap of 5 binds only the exploratory trees of early warmup, which is where the
# unbounded default spends the bulk of the fit. Measured on one chain of 100 + 100: 57.8 s uncapped
# against 33.5 s at depth 5, with the posterior means of ``sigma_q`` and ``sigma_h`` unchanged to three
# significant figures. Halving the draws on top of that lands the whole module near 70 s, and 200
# posterior draws is ample for a *median* point forecast, which is all this module scores.
NUM_WARMUP = 100
NUM_SAMPLES = 100
NUM_CHAINS = 2
MAX_TREE_DEPTH = 5
N_POSTERIOR = NUM_CHAINS * NUM_SAMPLES

#: Seed for :func:`forecast_ssp`. The NUTS run is pinned separately at ``PRNGKey(0)``.
FORECAST_SEED = 0

#: Seasonal period of the naive baseline: "same weekday last week", carried across the horizon.
NAIVE_PERIOD = 7

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Calibrated 2026-09-03 on the committed fixture, at the NUTS configuration above. The full run:
#
#   series            MAPE%   naive%   ratio
#   CA_FOODS           6.20     6.70   0.925
#   CA_HOBBIES         6.21     8.38   0.742
#   CA_HOUSEHOLD       6.15     5.28   1.166
#   TOTAL              7.38     8.31   0.889
#   TX_FOODS           7.90     9.70   0.815
#   TX_HOBBIES        11.96    12.02   0.995
#   TX_HOUSEHOLD       6.87    10.72   0.641
#   WI_FOODS          14.23    13.19   1.079
#   WI_HOBBIES        10.45     8.79   1.189
#   WI_HOUSEHOLD       9.84     7.99   1.233
#   panel mean: MAPE 8.72% | seasonal-naive MAPE 9.11% | ratio 0.958
#
# The two absolute ceilings carry 50% relative headroom over their calibrated value, rounded up to a
# clean number: they are meant to pass through ordinary tuning and fail on a broken algorithm, not to
# rank fits. Re-derive them by re-running this module and applying the same rule; do not nudge one
# because a change moved the number a little.

#: Panel-mean MAPE ceiling, in percent. Calibrated 8.72; 8.72 x 1.5 = 13.08, rounded up.
PANEL_MAPE_MAX = 14.0

#: Per-series MAPE ceiling, in percent, so one blown-up series cannot hide behind nine good ones.
#: Calibrated on the worst series, ``WI_FOODS`` at 14.23; 14.23 x 1.5 = 21.35, rounded up.
SERIES_MAPE_MAX = 22.0

#: Panel-mean model MAPE must be at or below seasonal-naive. Scale-free, so unlike the two ceilings it
#: survives a fixture change untouched, and it cannot silently absorb a regression that a lucky
#: calibration run baked into an absolute number.
#:
#: Calibrated at 0.958 — the model does beat "same weekday last week", but by only 4%, and four of the
#: ten series (CA_HOUSEHOLD, WI_FOODS, WI_HOBBIES, WI_HOUSEHOLD) score *worse* than naive. That is a
#: real property of the deliberately minimal design here — a random-walk level plus six weekly dummies,
#: no covariates, no trend, no holidays — not of the forecasting engine, and it is why the guard is on
#: the panel mean rather than per series. The thin margin is the point of the guard, not a reason to
#: loosen it: the ceilings above are the loose half of the pair.
MAX_NAIVE_RATIO = 1.0


def _load_builder() -> ModuleType:
    """Import the playground fixture builder by path.

    Returns
    -------
    ModuleType
        The imported ``m5_make_accuracy_fixture`` module.
    """
    spec = importlib.util.spec_from_file_location("m5_make_accuracy_fixture", M5_BUILDER)
    if spec is None or spec.loader is None:  # pragma: no cover - only on a broken checkout
        raise ImportError(f"cannot load the fixture builder from {M5_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_panel() -> pd.DataFrame:
    """Read the committed fixture into a long panel, via the packaged loader.

    Returns
    -------
    pd.DataFrame
        Columns ``date`` (datetime64), ``series_id`` (str), ``sales`` (int).
    """
    return load_m5_aggregate()


class TestFixtureProvenance:
    def test_panel_has_the_expected_grid(self):
        panel = _read_panel()

        assert list(panel.columns) == ["date", "series_id", "sales"]
        assert sorted(panel["series_id"].unique()) == SERIES
        assert panel["date"].nunique() == N_DATES
        assert len(panel) == N_DATES * N_SERIES
        assert panel.notna().all().all()

    def test_every_series_is_dense_and_strictly_positive(self):
        # The whole point of aggregating to state x cat is that MAPE becomes meaningful. A zero would
        # make it undefined, and a near-zero would make it explode, so assert the density claim here
        # rather than trusting the builder to have held it.
        panel = _read_panel()

        assert (panel["sales"] > 0).all()
        assert panel["sales"].dtype.kind in "iu"
        assert panel.groupby("series_id")["sales"].min().min() >= 1

    def test_dates_form_a_contiguous_daily_grid(self):
        dates = pd.DatetimeIndex(sorted(_read_panel()["date"].unique()))

        assert len(dates) == N_DATES
        # Contiguous days: the weekly design the accuracy harness builds is positional, so a missing
        # date would silently shift every weekday dummy after it.
        assert (dates.to_series().diff().dropna() == pd.Timedelta(days=1)).all()

    def test_file_is_committed_and_small(self):
        # ``.gitignore`` ignores ``*.csv`` but re-includes ``src/bunobee/datasets/data/*.csv``; if that
        # exception ever goes away the file vanishes from CI and every check above fails for the wrong
        # reason.
        assert M5_PANEL.exists()
        assert M5_PANEL.stat().st_size < MAX_FIXTURE_BYTES

    def test_content_matches_the_pinned_digest(self):
        # Normalised to LF so a CRLF checkout does not fail a content check for a whitespace reason.
        content = M5_PANEL.read_bytes().replace(b"\r\n", b"\n")

        assert hashlib.sha256(content).hexdigest() == FIXTURE_SHA256

    def test_matches_the_m5_dump_when_available(self):
        # The raw competition dump is gitignored, so this only runs where a working copy has it.
        if not M5_SOURCE.exists():
            pytest.skip(f"raw M5 data not available at {M5_SOURCE}")

        expected = _load_builder().build_m5_aggregate(M5_SOURCE)
        committed = _read_panel()

        pd.testing.assert_frame_equal(committed, expected, check_dtype=False)


# ---------------------------------------------------------------------------
# The accuracy harness — fit, forecast, score
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class M5Fit:
    """One fit of the aggregated M5 panel, scored on its 28-day holdout.

    Every array is on the **un-normalized** scale the fixture is stored in — units of sales per day —
    so a threshold read off this object means the same thing as a number read off the raw CSV.

    Attributes
    ----------
    idata : xr.Dataset
        Posterior packaged the way :func:`forecast_ssp` reads it.
    Z_future : np.ndarray, shape (HORIZON, N_SERIES, N_STATES)
        Future design, continued from the training design by :func:`build_forecast_design`.
    forecast : np.ndarray, shape (HORIZON, N_SERIES)
        Point forecast: the posterior **median** over the ``sample`` axis, un-normalized.
    truth : np.ndarray, shape (HORIZON, N_SERIES)
        Held-out actuals over the same window.
    naive : np.ndarray, shape (HORIZON, N_SERIES)
        Seasonal-naive baseline, ``y[t - 7]`` carried across the horizon.
    scale : np.ndarray, shape (N_SERIES,)
        Per-series training-window mean used to normalize, kept so the un-normalization is auditable.
    """

    idata: xr.Dataset
    Z_future: np.ndarray
    forecast: np.ndarray
    truth: np.ndarray
    naive: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class M5Scores:
    """Per-series and panel-mean error of the model and its seasonal-naive baseline.

    Attributes
    ----------
    mape_model, smape_model : np.ndarray, shape (N_SERIES,)
        Model error per series, in percent.
    mape_naive, smape_naive : np.ndarray, shape (N_SERIES,)
        Seasonal-naive error per series, in percent.
    ratio : np.ndarray, shape (N_SERIES,)
        ``mape_model / mape_naive`` per series. Scale-free, so it survives a fixture change that
        moves the absolute MAPE.
    """

    mape_model: np.ndarray
    mape_naive: np.ndarray
    smape_model: np.ndarray
    smape_naive: np.ndarray
    ratio: np.ndarray

    @property
    def panel_mape_model(self) -> float:
        """float: Panel-mean model MAPE, in percent."""
        return float(self.mape_model.mean())

    @property
    def panel_mape_naive(self) -> float:
        """float: Panel-mean seasonal-naive MAPE, in percent."""
        return float(self.mape_naive.mean())

    @property
    def panel_ratio(self) -> float:
        """float: Ratio of the two panel-mean MAPEs."""
        return self.panel_mape_model / self.panel_mape_naive


def _to_panel(df: pd.DataFrame) -> np.ndarray:
    """Pivot the long fixture into a ``(time, series)`` array in :data:`SERIES` order.

    Parameters
    ----------
    df : pd.DataFrame
        Long panel with columns ``date``, ``series_id``, ``sales``.

    Returns
    -------
    np.ndarray, shape (N_DATES, N_SERIES)
        Sales indexed by date (ascending) and series.
    """
    dates = sorted(df["date"].unique())
    wide = df.pivot(index="date", columns="series_id", values="sales").reindex(index=dates, columns=SERIES)
    return np.asarray(wide.values, dtype=float)


def _build_design(n_all: int) -> np.ndarray:
    """Assemble ``[intercept | weekly dummies]`` over ``n_all`` steps, broadcast across series.

    Parameters
    ----------
    n_all : int
        Number of time steps to build.

    Returns
    -------
    np.ndarray, shape (n_all, N_SERIES, N_STATES)
        Design matrix. The weekly block is positional, which is why the fixture's contiguous daily
        grid is asserted in :class:`TestFixtureProvenance`.
    """
    dummies = np.asarray(make_peridoic_dummies(n_all, period=PERIOD, drop_first=True), dtype=float)
    intercept = np.ones((n_all, N_SERIES, 1))
    dummies_b = np.broadcast_to(dummies[:, None, :], (n_all, N_SERIES, N_DUMMIES))
    return np.concatenate([intercept, dummies_b], axis=-1)


def _ekf_prior() -> xr.Dataset:
    """Natural-scale prior for the level + weekly design, transformed to EKF a-space.

    Returns
    -------
    xr.Dataset
        Prior with ``a0``, ``P0``, ``positivity`` and the Beta ``sigma_q`` block, as
        :func:`transform_to_ekf_st` returns it.
    """
    prior = xr.Dataset(
        {
            "a0": (("state",), A0),
            "P0": (("state", "state_dual"), np.diag(P0_DIAG)),
            "positivity": (("state",), POSITIVITY),
            "sigma_q_alpha_prior": (("state",), np.full(N_STATES, 2.0)),
            "sigma_q_beta_prior": (("state",), np.full(N_STATES, 10.0)),
            "sigma_q_scale_prior": (("state",), SIGMA_Q_SCALE),
        }
    ).assign_attrs({"sigma_q_family": "beta"})
    return transform_to_ekf_st(prior, exponent=EXPONENT)


def _run_nuts(prior_ekf: xr.Dataset, y_train: np.ndarray, Z_train: np.ndarray) -> xr.Dataset:
    """Fit the shared-state EKF likelihood and package the posterior for :func:`forecast_ssp`.

    Mirrors ``tests/test_ssp_forecast_e2e.py::_run_nuts``: the only sampled parameters are the
    per-series observation noise and the per-state process noise, so the geometry stays two-block and
    low-dimensional and a fixed ``PRNGKey`` reproduces the chain exactly.

    Parameters
    ----------
    prior_ekf : xr.Dataset
        Output of :func:`_ekf_prior`.
    y_train : np.ndarray, shape (N_STEPS, N_SERIES)
        Normalized training response.
    Z_train : np.ndarray, shape (N_STEPS, N_SERIES, N_STATES)
        Training design.

    Returns
    -------
    xr.Dataset
        Posterior with ``at``, ``sigma_q``, ``sigma_h``, ``positivity`` and the ``exponent`` attribute.
    """
    import jax.numpy as jnp
    import numpyro
    from jax import random
    from numpyro import distributions as dist
    from numpyro.infer import MCMC, NUTS

    from bunobee.models.ssp.kalman_1d_st_ekf import kalman_filter_1d_ekf_st

    a0 = jnp.asarray(prior_ekf["a0"].values)
    P0 = jnp.asarray(prior_ekf["P0"].values)
    positivity_j = jnp.asarray(POSITIVITY)
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
        NUTS(model, max_tree_depth=MAX_TREE_DEPTH),
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
        # Sequential, not vectorized: vmapped chains run to the slowest chain's tree at every step,
        # which measured ~25% *slower* here than running the two chains one after the other.
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
            "positivity": (("state",), POSITIVITY),
        }
    ).assign_attrs({"exponent": EXPONENT})


def _mape(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Mean absolute percentage error per series, in percent.

    Well-behaved here precisely because the panel is a dense aggregate: every denominator is in the
    thousands, so no term of the mean can blow up.

    Parameters
    ----------
    truth, pred : np.ndarray, shape (horizon, n_series)
        Actuals and point forecast on the same scale.

    Returns
    -------
    np.ndarray, shape (n_series,)
        Per-series MAPE in percent, averaged over the horizon.
    """
    return 100.0 * np.mean(np.abs(truth - pred) / np.abs(truth), axis=0)


def _smape(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Symmetric mean absolute percentage error per series, in percent.

    Reported alongside :func:`_mape` because it is bounded by 200 and penalizes over- and
    under-forecasting alike, so a MAPE that moves while the sMAPE does not points at a scale problem
    rather than a broken forecast.

    Parameters
    ----------
    truth, pred : np.ndarray, shape (horizon, n_series)
        Actuals and point forecast on the same scale.

    Returns
    -------
    np.ndarray, shape (n_series,)
        Per-series sMAPE in percent, averaged over the horizon.
    """
    denom = np.abs(truth) + np.abs(pred)
    return 200.0 * np.mean(np.abs(truth - pred) / denom, axis=0)


def _seasonal_naive(y_train: np.ndarray, horizon: int, period: int = NAIVE_PERIOD) -> np.ndarray:
    """Carry the last ``period`` training observations across the horizon.

    The baseline the model has to beat: "same weekday last week", repeated as often as the horizon
    needs. With ``horizon = 28`` and ``period = 7`` that is the final training week, tiled four times.

    Parameters
    ----------
    y_train : np.ndarray, shape (n_steps, n_series)
        Training response; only its last ``period`` rows are used.
    horizon : int
        Number of steps to produce.
    period : int, optional
        Seasonal period. Default :data:`NAIVE_PERIOD`.

    Returns
    -------
    np.ndarray, shape (horizon, n_series)
        Seasonal-naive forecast on the scale of ``y_train``.
    """
    return y_train[-period:][np.arange(horizon) % period]


def _score(fit: M5Fit) -> M5Scores:
    """Score a fit against its holdout and its seasonal-naive baseline.

    Parameters
    ----------
    fit : M5Fit
        The module fixture.

    Returns
    -------
    M5Scores
        Per-series and panel-mean error of both forecasts.
    """
    mape_model = _mape(fit.truth, fit.forecast)
    mape_naive = _mape(fit.truth, fit.naive)
    return M5Scores(
        mape_model=mape_model,
        mape_naive=mape_naive,
        smape_model=_smape(fit.truth, fit.forecast),
        smape_naive=_smape(fit.truth, fit.naive),
        ratio=mape_model / mape_naive,
    )


def _score_table(scores: M5Scores) -> str:
    """Render the per-series error table plus its panel summary line.

    ``log_cli = true`` / ``log_cli_level = "INFO"`` is already set in ``pyproject.toml``, so logging
    this once per run means a future threshold failure names *which* series moved without a re-run.

    Parameters
    ----------
    scores : M5Scores
        Output of :func:`_score`.

    Returns
    -------
    str
        Multi-line table, one row per series, ordered as :data:`SERIES`.
    """
    header = f"{'series':<14}{'MAPE%':>9}{'naive%':>9}{'ratio':>8}{'sMAPE%':>9}{'naive%':>9}"
    rows = [
        f"{name:<14}{scores.mape_model[i]:>9.2f}{scores.mape_naive[i]:>9.2f}"
        f"{scores.ratio[i]:>8.3f}{scores.smape_model[i]:>9.2f}{scores.smape_naive[i]:>9.2f}"
        for i, name in enumerate(SERIES)
    ]
    summary = (
        f"panel mean: MAPE {scores.panel_mape_model:.2f}% | "
        f"seasonal-naive MAPE {scores.panel_mape_naive:.2f}% | ratio {scores.panel_ratio:.3f}"
    )
    return "\n".join([f"M5 aggregate accuracy ({HORIZON}-day holdout, {N_SERIES} series)", header, *rows, summary])


def _threshold_failure(headline: str, scores: M5Scores) -> str:
    """Compose a self-explaining threshold-failure message.

    The headline carries the observed value and the threshold it broke; the table underneath carries
    every series, so whoever reads the failure can tell a panel-wide regression from one series moving
    without re-running a 75-second fit.

    Parameters
    ----------
    headline : str
        One line naming what was observed, what the limit was, and — where the check is per-series —
        which series broke it.
    scores : M5Scores
        The scores the assertion read.

    Returns
    -------
    str
        The headline followed by :func:`_score_table`.
    """
    return f"{headline}\n{_score_table(scores)}"


@pytest.fixture(scope="module")
def fit() -> M5Fit:
    """Fit the aggregated panel once, forecast the holdout, and return everything un-normalized.

    Returns
    -------
    M5Fit
        Posterior, future design, point forecast, actuals, and the seasonal-naive baseline.
    """
    pytest.importorskip("numpyro")

    y_raw = _to_panel(_read_panel())
    assert y_raw.shape == (N_DATES, N_SERIES)

    # Per-series mean normalization fitted on the training window only, so the holdout stays out of
    # sample. Scoring happens after un-normalizing, on the fixture's own units.
    scale = y_raw[:N_STEPS].mean(axis=0)
    y = y_raw / scale

    Z_all = _build_design(N_DATES)
    Z_train = Z_all[:N_STEPS]
    y_train = y[:N_STEPS]

    idata = _run_nuts(_ekf_prior(), y_train, Z_train)
    Z_future = build_forecast_design(Z_train, HORIZON, periodic=WEEKLY_SPEC)

    out = forecast_ssp(idata, Z_future, noise_embed=False, seed=FORECAST_SEED)
    # Median over the posterior, not the mean: robust to a stray draw, and what the M5 prototype's
    # ``predict_one_series`` reports.
    forecast = np.median(out["forecast_samples"].values, axis=0) * scale

    return M5Fit(
        idata=idata,
        Z_future=Z_future,
        forecast=forecast,
        truth=y_raw[N_STEPS:],
        naive=_seasonal_naive(y_raw[:N_STEPS], HORIZON),
        scale=scale,
    )


@pytest.fixture(scope="module")
def scores(fit: M5Fit) -> M5Scores:
    """Score the module fit once; every assertion below reads this.

    Parameters
    ----------
    fit : M5Fit
        The module fixture.

    Returns
    -------
    M5Scores
        Per-series and panel-mean error.
    """
    return _score(fit)


class TestAccuracyHarness:
    """The machinery the thresholds assert on: shapes, scale, and seed stability.

    No error ceiling lives here on purpose — those are :class:`TestAccuracyThresholds`. What is
    asserted here is everything a threshold silently depends on: that the arrays line up, that they
    are on the fixture's own scale, and that the numbers do not move between runs. When a threshold
    fails, these tests are what says whether the error moved or the harness did.
    """

    def test_fixture_shapes_and_scale(self, fit: M5Fit):
        assert fit.forecast.shape == (HORIZON, N_SERIES)
        assert fit.truth.shape == (HORIZON, N_SERIES)
        assert fit.naive.shape == (HORIZON, N_SERIES)
        assert fit.Z_future.shape == (HORIZON, N_SERIES, N_STATES)
        assert np.isfinite(fit.forecast).all()
        # Un-normalized: the panel runs in the thousands of units per day, so a forecast still sitting
        # on the ~1.0 normalized scale would be an un-normalization bug, not a modelling one.
        np.testing.assert_allclose(fit.scale, _to_panel(_read_panel())[:N_STEPS].mean(axis=0))
        assert fit.forecast.min() > 1.0
        assert 0.2 < fit.forecast.mean() / fit.truth.mean() < 5.0

    def test_holdout_truth_matches_the_fixture_tail(self, fit: M5Fit):
        # The split is the load-bearing part of the whole guard: an off-by-one here would score the
        # model against data it trained on and every threshold would read as passing.
        panel = _read_panel()
        tail_dates = sorted(panel["date"].unique())[N_STEPS:]
        expected = _to_panel(panel[panel["date"].isin(tail_dates)])

        assert len(tail_dates) == HORIZON
        np.testing.assert_array_equal(fit.truth, expected)

    def test_seasonal_naive_repeats_the_last_training_week(self, fit: M5Fit):
        y_raw = _to_panel(_read_panel())

        np.testing.assert_array_equal(fit.naive[:NAIVE_PERIOD], y_raw[N_STEPS - NAIVE_PERIOD : N_STEPS])
        # 28 = 4 x 7, so the whole horizon is the final training week tiled four times.
        for block in range(1, HORIZON // NAIVE_PERIOD):
            start = block * NAIVE_PERIOD
            np.testing.assert_array_equal(fit.naive[start : start + NAIVE_PERIOD], fit.naive[:NAIVE_PERIOD])

    def test_future_design_continues_the_weekly_cycle(self, fit: M5Fit):
        np.testing.assert_allclose(fit.Z_future[:, :, 0], 1.0)
        weekly = fit.Z_future[:, 0, 1:PERIOD]
        assert set(np.unique(weekly)) <= {0.0, 1.0}
        # Every forecast step lands on exactly one weekday, except the dropped reference day.
        np.testing.assert_array_equal(np.unique(weekly.sum(axis=1)), np.array([0.0, 1.0]))
        # ... and the phase continues unbroken from the training design.
        np.testing.assert_allclose(fit.Z_future, _build_design(N_DATES)[N_STEPS:])

    def test_metrics_are_finite_and_well_posed(self, scores: M5Scores):
        for name in ("mape_model", "mape_naive", "smape_model", "smape_naive", "ratio"):
            values = getattr(scores, name)
            assert values.shape == (N_SERIES,)
            assert np.isfinite(values).all(), f"{name} is not finite: {values}"
            assert (values > 0).all(), f"{name} has a non-positive entry: {values}"
        np.testing.assert_allclose(scores.ratio, scores.mape_model / scores.mape_naive)
        np.testing.assert_allclose(scores.panel_mape_model, scores.mape_model.mean())

    def test_metric_helpers_are_exact_on_a_known_case(self):
        truth = np.array([[100.0, 200.0], [100.0, 200.0]])
        pred = np.array([[110.0, 200.0], [90.0, 100.0]])

        np.testing.assert_allclose(_mape(truth, pred), [10.0, 25.0])
        np.testing.assert_allclose(_smape(truth, pred), [200.0 * (10 / 210 + 10 / 190) / 2, 200.0 * (100 / 300) / 2])
        np.testing.assert_allclose(_mape(truth, truth), [0.0, 0.0])

    def test_point_forecast_is_the_posterior_median(self, fit: M5Fit):
        # The un-normalization and the median are the two steps between forecast_ssp and every number
        # this module reports, so pin both against a fresh draw from the same posterior and seed.
        out = forecast_ssp(fit.idata, fit.Z_future, noise_embed=False, seed=FORECAST_SEED)
        expected = np.median(out["forecast_samples"].values, axis=0) * fit.scale

        assert out["forecast_samples"].shape == (N_POSTERIOR, HORIZON, N_SERIES)
        np.testing.assert_array_equal(fit.forecast, expected)

    def test_scoring_is_stable_within_a_run(self, fit: M5Fit, scores: M5Scores):
        # Consume the global RNG and forecast under a different seed in between: nothing but
        # (idata, Z_future, seed) may reach the reported numbers.
        np.random.default_rng().standard_normal(1000)
        forecast_ssp(fit.idata, fit.Z_future, noise_embed=False, seed=FORECAST_SEED + 99)

        again = forecast_ssp(fit.idata, fit.Z_future, noise_embed=False, seed=FORECAST_SEED)
        repeated = _score(
            M5Fit(
                idata=fit.idata,
                Z_future=fit.Z_future,
                forecast=np.median(again["forecast_samples"].values, axis=0) * fit.scale,
                truth=fit.truth,
                naive=fit.naive,
                scale=fit.scale,
            )
        )

        np.testing.assert_array_equal(repeated.mape_model, scores.mape_model)
        np.testing.assert_array_equal(repeated.ratio, scores.ratio)
        assert _score_table(repeated) == _score_table(scores)

    def test_per_series_table_is_logged(self, caplog, scores: M5Scores):
        with caplog.at_level(logging.INFO, logger=__name__):
            logger.info("\n%s", _score_table(scores))

        captured = caplog.text
        for name in SERIES:
            assert name in captured, f"{name} missing from the logged table"
        assert "panel mean:" in captured
        assert f"{scores.panel_mape_model:.2f}%" in captured


class TestAccuracyThresholds:
    """The tripwire: three assertions, loosest first, plus a check that they actually fire.

    The two absolute ceilings are the loose half of the pair — 50% relative headroom over the
    calibrated number, so ordinary tuning passes — and :data:`MAX_NAIVE_RATIO` is the half that cannot
    rot, because it is measured against a baseline recomputed from the same split on every run.

    They are not redundant. Measured 2026-09-03: zeroing the weekly block in ``Z_future`` moves the
    panel mean to 13.57% and the worst series to 19.47%, both *inside* the ceilings — only the ratio
    guard catches it, at 1.490. Forecasting from ``at[:, :, 0, :]`` instead of the last state breaks
    all three (24.72% panel, 31.75% worst, ratio 2.714).
    """

    def test_panel_mape_is_under_the_ceiling(self, scores: M5Scores):
        observed = scores.panel_mape_model

        assert observed <= PANEL_MAPE_MAX, _threshold_failure(
            f"panel-mean MAPE {observed:.2f}% exceeds PANEL_MAPE_MAX {PANEL_MAPE_MAX:.2f}%",
            scores,
        )

    def test_no_series_is_over_the_per_series_ceiling(self, scores: M5Scores):
        # Per-series, not just the mean: nine good series can hide one that blew up, and a broken
        # state or a mis-continued seasonal block often shows on one series before the panel moves.
        offenders = [
            (name, scores.mape_model[i]) for i, name in enumerate(SERIES) if scores.mape_model[i] > SERIES_MAPE_MAX
        ]

        assert not offenders, _threshold_failure(
            f"{len(offenders)} series over SERIES_MAPE_MAX {SERIES_MAPE_MAX:.2f}%: "
            + ", ".join(f"{name} {value:.2f}%" for name, value in offenders),
            scores,
        )

    def test_panel_beats_the_seasonal_naive_baseline(self, scores: M5Scores):
        # Scale-free, so it survives a fixture change that moves every absolute MAPE, and it is the
        # only one of the three that asserts the model is doing something rather than merely not
        # doing something terrible.
        observed = scores.panel_ratio

        assert observed <= MAX_NAIVE_RATIO, _threshold_failure(
            f"panel-mean model MAPE {scores.panel_mape_model:.2f}% is {observed:.3f}x the "
            f"seasonal-naive {scores.panel_mape_naive:.2f}%, over MAX_NAIVE_RATIO {MAX_NAIVE_RATIO:.3f}",
            scores,
        )

    def test_the_thresholds_fire_on_a_broken_forecast(self, fit: M5Fit, scores: M5Scores):
        # A tripwire nobody has tripped is not known to work. Zeroing the weekly block in Z_future is
        # the cheapest realistic break — the shapes, the identities and the fit all stay valid, and
        # only the seasonal continuation is gone, which is exactly the class of regression the rest of
        # the suite lands green on. Reuses the module fit, so this costs one forecast, not another
        # NUTS run.
        Z_broken = fit.Z_future.copy()
        Z_broken[:, :, 1:PERIOD] = 0.0
        out = forecast_ssp(fit.idata, Z_broken, noise_embed=False, seed=FORECAST_SEED)
        broken = _score(
            M5Fit(
                idata=fit.idata,
                Z_future=Z_broken,
                forecast=np.median(out["forecast_samples"].values, axis=0) * fit.scale,
                truth=fit.truth,
                naive=fit.naive,
                scale=fit.scale,
            )
        )

        assert broken.panel_mape_model > scores.panel_mape_model, _threshold_failure(
            f"a forecast with no weekly block scored {broken.panel_mape_model:.2f}%, no worse than "
            f"the healthy {scores.panel_mape_model:.2f}% — the weekly states are not reaching the "
            "forecast, so this module is guarding nothing",
            broken,
        )
        assert broken.panel_ratio > MAX_NAIVE_RATIO, _threshold_failure(
            f"a forecast with no weekly block scored ratio {broken.panel_ratio:.3f}, still within "
            f"MAX_NAIVE_RATIO {MAX_NAIVE_RATIO:.3f} — the guard would not catch this break",
            broken,
        )
