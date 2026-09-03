"""Forecast-accuracy guard on the aggregated M5 panel.

This module holds the hermetic benchmark behind issue #59: a small, dense, deterministic panel that a
broken forecast can be scored against. Issue #60 lands the fixture and its provenance guard; the fit,
the metrics, and the thresholds arrive on top of it.

The data is ``tests/fixtures/m5_aggregate.csv``, built by
``playground/m5_prototype/m5_make_accuracy_fixture.py`` from the M5 Forecasting — Accuracy competition
dump. That dump is 980 MB and untracked (``*.csv`` / ``*.nc`` are gitignored), so CI cannot see it; the
committed file is a derived aggregate of it — 3 states x 3 categories plus a national ``TOTAL``, summed
over all 30 490 item-store series, for the last ``N_STEPS + HORIZON`` observed days — not a
redistribution of the raw competition data.

Aggregating is what makes the panel scoreable at all: raw M5 item-store series are intermittent and
zero-inflated, and MAPE on a series that spends half its life at zero means nothing. The aggregates run
in the thousands of units per day with no zeros anywhere, which the grid checks below assert rather than
assume.

:class:`TestFixtureProvenance` mirrors the class of the same name in ``tests/test_ssp_forecast_e2e.py``:
the grid checks always run, and the re-derivation check rebuilds the aggregate from the raw dump
whenever a working copy has it, so the committed copy cannot silently drift.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
M5_PANEL = _ROOT / "tests" / "fixtures" / "m5_aggregate.csv"
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
    """Read the committed fixture into a long panel.

    Returns
    -------
    pd.DataFrame
        Columns ``date`` (datetime64), ``series_id`` (str), ``sales`` (int).
    """
    return pd.read_csv(M5_PANEL, parse_dates=["date"])


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
        # ``.gitignore`` ignores ``*.csv`` but re-includes ``tests/fixtures/*.csv``; if that exception
        # ever goes away the file vanishes from CI and every check above fails for the wrong reason.
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
