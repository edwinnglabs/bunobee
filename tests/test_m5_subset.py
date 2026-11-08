"""Unit tests for the M5 benchmark subset, ``playground/m5_prototype/m5_subset.py`` (issue #154).

The subset is three items in all their stores, one per volume tier and category, picked by a rule on the
pre-validation fit window: a listed price in every store on each of the last days, ranked by units, the top item
without stockout-like zeros for the high tier and the items nearest the median and the 10th percentile of units for
the others. These tests check the rule on a tiny synthetic dataset built in this module, that the committed items
follow the tiers, the order of the subset's series, and that each item is scored on its own series.

The script is a playground script, not part of the package, so it is loaded by path as ``tests/test_m5_score.py``
loads the scorer.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import xarray as xr

M5_DIR = Path(__file__).resolve().parents[1] / "playground" / "m5_prototype"


def _load_script(name: str) -> ModuleType:
    """Import a playground M5 script by path and register it under its own name.

    Registering matters for the scripts imported by name: ``m5_subset`` imports ``m5_common`` and
    ``m5_score_submission``, which do not resolve from ``tests/`` until they are in ``sys.modules``.

    Parameters
    ----------
    name : str
        Module name, i.e. the file stem under ``playground/m5_prototype/``.

    Returns
    -------
    ModuleType
        The imported module.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, M5_DIR / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover - only on a broken checkout
        raise ImportError(f"cannot load {name} from {M5_DIR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


_load_script("m5_common")
m5_score_submission = _load_script("m5_score_submission")
m5_subset = _load_script("m5_subset")

# Calendar: N_PRE_FIT pre-validation fit days, then the 28 pre-validation days, both labelled "train", then the 28
# validation days. Items are picked on the last N_DAYS pre-validation fit days.
N_PRE_FIT = 40
HORIZON = 28
N_DAYS = 10
STORES = ("CA_1", "CA_2")
# Item -> units per day in every store. Over the N_DAYS days and both stores an item sells 20 x its rate.
RATES = {
    "FOODS_1_001": 30.0,  # the top seller, but 0 on STOCKOUT_DAYS: skipped by the high tier
    "FOODS_1_002": 15.0,  # the high item
    "FOODS_1_003": 50.0,  # no price on one of the N_DAYS days in one store: not eligible
    "FOODS_1_004": 1.0,  # no price on day 0, before the N_DAYS days: still eligible
    "HOUSEHOLD_1_001": 5.0,
    "HOUSEHOLD_1_002": 2.0,  # nearest the median of the eligible units, 60: the mid item
    "HOBBIES_1_001": 3.0,
    "HOBBIES_1_002": 0.5,  # nearest their 10th percentile, 16: the low item
}
STOCKOUT_DAYS = (N_PRE_FIT - 3, N_PRE_FIT - 2, N_PRE_FIT - 1)
# Eligible items by units, highest first: 420, 300, 100, 60, 40, 20, 10.
RANKED = (
    "FOODS_1_001",
    "FOODS_1_002",
    "HOUSEHOLD_1_001",
    "HOBBIES_1_001",
    "HOUSEHOLD_1_002",
    "FOODS_1_004",
    "HOBBIES_1_002",
)
PICKED = {"high": "FOODS_1_002", "mid": "HOUSEHOLD_1_002", "low": "HOBBIES_1_002"}


@pytest.fixture(scope="module")
def ds() -> xr.Dataset:
    """M5-shaped dataset of the RATES items in both STORES, items in RATES order, stores in STORES order."""
    n_dates = N_PRE_FIT + 2 * HORIZON
    items = [item for item in RATES for _ in STORES]
    sales = np.repeat(np.array([RATES[item] for item in items])[:, None], n_dates, axis=1)
    # Every other day one unit more, so every series has a non-zero RMSSE scale.
    sales = sales + (np.arange(n_dates) % 2)[None, :]
    prices = np.full(sales.shape, 2.0)
    for row, item in enumerate(items):
        if item == "FOODS_1_001":
            sales[row, list(STOCKOUT_DAYS)] = 0.0
        if item == "FOODS_1_003" and row % 2:
            prices[row, N_PRE_FIT - 5] = np.nan
        if item == "FOODS_1_004":
            prices[row, 0] = np.nan
    return xr.Dataset(
        {
            "sales": (("series_id", "date"), sales.astype(np.float32)),
            "sell_price": (("series_id", "date"), prices.astype(np.float32)),
        },
        coords={
            "series_id": [f"{item}_{STORES[row % 2]}" for row, item in enumerate(items)],
            "item_id": ("series_id", items),
            "dept_id": ("series_id", [item.rsplit("_", 1)[0] for item in items]),
            "cat_id": ("series_id", [item.split("_")[0] for item in items]),
            "store_id": ("series_id", [STORES[row % 2] for row in range(len(items))]),
            "state_id": ("series_id", ["CA"] * len(items)),
            "date": pd.date_range("2016-01-01", periods=n_dates),
            "split": ("date", np.array(["train"] * (N_PRE_FIT + HORIZON) + ["validation"] * HORIZON)),
        },
    )


@pytest.fixture(scope="module")
def eligible(ds) -> pd.DataFrame:
    return m5_subset.eligible_items(ds, n_days=N_DAYS)


class TestEligible:
    def test_an_item_needs_a_price_in_every_store_on_every_day(self, eligible):
        assert "FOODS_1_003" not in eligible.index
        # A missing price before the last N_DAYS fit days does not count.
        assert "FOODS_1_004" in eligible.index

    def test_items_are_ranked_by_units(self, eligible):
        assert tuple(eligible.index) == RANKED
        np.testing.assert_array_equal(eligible["rank"], np.arange(1, len(RANKED) + 1))

    def test_units_are_over_the_last_fit_days_of_every_store(self, eligible):
        # The last N_DAYS fit days hold 5 days with the extra unit, in each of the 2 stores.
        assert eligible.loc["FOODS_1_002", "units"] == 15.0 * 20 + 10
        assert eligible.loc["FOODS_1_002", "units_per_store_day"] == pytest.approx((15.0 * 20 + 10) / 20)

    def test_zero_share_is_the_median_stores(self, eligible):
        assert eligible.loc["FOODS_1_001", "zero_share"] == pytest.approx(len(STOCKOUT_DAYS) / N_DAYS)
        assert eligible.loc["FOODS_1_002", "zero_share"] == 0.0

    def test_ties_go_by_item_id(self, ds):
        tied = ds.copy(deep=True)
        sales = tied["sales"].values
        sales[tied["item_id"].values == "HOBBIES_1_001"] = sales[tied["item_id"].values == "HOUSEHOLD_1_002"]
        ranked = list(m5_subset.eligible_items(tied, n_days=N_DAYS).index)
        assert ranked.index("HOBBIES_1_001") + 1 == ranked.index("HOUSEHOLD_1_002")


class TestPick:
    def test_one_item_per_tier(self, eligible):
        assert m5_subset.pick_items(eligible) == PICKED

    def test_the_high_tier_skips_stockout_zeros(self, eligible):
        assert eligible.index[0] == "FOODS_1_001"
        assert m5_subset.pick_items(eligible)["high"] != "FOODS_1_001"

    def test_a_category_without_an_eligible_item_is_an_error(self, eligible):
        with pytest.raises(ValueError, match="No eligible FOODS item for the high tier"):
            m5_subset.pick_items(eligible[eligible["cat_id"] != "FOODS"])


class TestSubsetItems:
    def test_one_item_per_tier_and_category(self):
        assert tuple(m5_subset.SUBSET_ITEMS) == tuple(tier.name for tier in m5_subset.TIERS)
        for tier in m5_subset.TIERS:
            assert m5_subset.SUBSET_ITEMS[tier.name].startswith(f"{tier.cat_id}_")

    def test_the_issue_items(self):
        assert m5_subset.SUBSET_ITEMS == {"high": "FOODS_3_586", "mid": "HOUSEHOLD_1_069", "low": "HOBBIES_1_197"}


class TestSeries:
    def test_every_store_of_each_item_in_item_order(self, ds):
        items = tuple(PICKED.values())
        idx = m5_subset.subset_series_idx(ds, items)

        assert idx.size == len(items) * len(STORES)
        np.testing.assert_array_equal(ds["item_id"].values[idx], np.repeat(items, len(STORES)))
        np.testing.assert_array_equal(ds["store_id"].values[idx], np.tile(STORES, len(items)))

    def test_an_unknown_item_is_an_error(self, ds):
        with pytest.raises(ValueError, match="FOODS_9_999"):
            m5_subset.subset_series_idx(ds, ("FOODS_1_002", "FOODS_9_999"))


@pytest.fixture(scope="module")
def scored(ds) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Pre-validation forecasts of the PICKED items' series, off their actuals, and their per-item scores."""
    idx = m5_subset.subset_series_idx(ds, tuple(PICKED.values()))
    _, score_mask = m5_score_submission.window_masks(ds["split"].values, "prevalidation")
    offsets = np.linspace(-1.0, 2.0, idx.size)[:, None]
    forecasts = ds["sales"].values[idx][:, score_mask].astype(np.float64) + offsets
    series_order = ds["series_id"].values[idx]
    return forecasts, series_order, m5_subset.per_item_wrmsse(ds, forecasts, series_order, "prevalidation")


class TestPerItem:
    def test_each_item_is_scored_on_its_own_series(self, ds, scored):
        forecasts, series_order, per_item = scored
        assert list(per_item["item_id"]) == list(PICKED.values())
        for row, item in enumerate(PICKED.values()):
            rows = np.char.startswith(series_order.astype(str), item)
            alone = m5_score_submission.compute_hierarchical_wrmsse(
                ds, forecasts[rows], series_order[rows], window="prevalidation"
            )
            assert per_item.loc[row, "wrmsse"] == pytest.approx(alone["wrmsse"].mean())
            assert per_item.loc[row, "n_series"] == len(STORES)

    def test_weights_are_dollar_shares(self, scored):
        *_, per_item = scored
        # Every price is 2, so the dollar shares are the unit shares of the last 28 fit days: 28 x rate + 14.
        units = np.array([28 * RATES[item] + 14 for item in PICKED.values()])
        np.testing.assert_allclose(per_item["weight"], units / units.sum())

    def test_items_outside_the_subset_have_no_tier(self, scored):
        *_, per_item = scored
        # The synthetic items are not SUBSET_ITEMS.
        assert (per_item["tier"] == "").all()
