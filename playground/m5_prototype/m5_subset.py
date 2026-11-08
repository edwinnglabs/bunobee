#!/usr/bin/env python3
"""The 3-item × 10-store M5 benchmark subset (issue #154): 30 series that fit in seconds.

A full M5 run is too slow to iterate on, and the random 3,000-series sample has no item × store structure. The
subset is three items, each in all 10 stores, one per volume tier and category, picked on purpose. ``m5_run.py
--sample-mode subset`` fits its 30 series, and its pre-validation WRMSSE is the acceptance score of the #143 re-plan.

The rule that picked the items, which :func:`pick_items` applies to :func:`eligible_items`:

1. **Eligible:** the item has a listed sell price in all 10 stores on each of the last :data:`ELIGIBLE_DAYS` days of
   the pre-validation fit window (d_1521-d_1885, ending 2016-03-27). 2,717 of the 3,049 items qualify.
2. **Ranked** by units sold over those days, summed over the 10 stores, highest first; ties go by ``item_id``.
3. **One item per tier, each tier its own category** (:data:`TIERS`):

   - *high* (``FOODS``): the top-ranked item whose median store has zero sales on under
     :data:`MAX_HIGH_ZERO_SHARE` of the days. The top item, ``FOODS_3_090``, sells 0 on 10-19% of the days in every
     store despite its volume, which looks like stockouts, so the rank 2 item is taken.
   - *mid* (``HOUSEHOLD``) and *low* (``HOBBIES``): the item of the category whose units are nearest the median and
     the 10th percentile of the eligible items' units, so the median rank and the 90th percentile rank.

Re-derived from ``m5_ds.nc`` (``python m5_subset.py``):

=====  ===================  ============  ===================  ========================
Tier   Item                 Rank          Units / store / day  Zero days (median store)
=====  ===================  ============  ===================  ========================
high   ``FOODS_3_586``      2             45.17                0.3%
mid    ``HOUSEHOLD_1_069``  1,360         0.63                 64.1%
low    ``HOBBIES_1_197``    2,442 (tied)  0.16                 85.9%
=====  ===================  ============  ===================  ========================

The high item holds most of the subset's dollar sales, so :func:`per_item_wrmsse` scores each item's 10 series on
their own, beside the subset's 12-level score.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from m5_common import DATA_DIR, load_m5_ds
from m5_score_submission import compute_hierarchical_wrmsse, dollar_sales, window_masks

logger = logging.getLogger(__name__)

#: The forecast origin whose fit window the items are picked on.
SUBSET_WINDOW = "prevalidation"

#: Days at the end of that fit window an eligible item must have a listed price on, in every store.
ELIGIBLE_DAYS = 365

#: Largest zero-sales share of the high item's median store: above it, zeros at that volume look like stockouts.
MAX_HIGH_ZERO_SHARE = 0.01


@dataclass(frozen=True)
class Tier:
    """A volume tier of the subset.

    Attributes
    ----------
    name : str
        Tier name, ``"high"``, ``"mid"`` or ``"low"``.
    cat_id : str
        Category the tier's item is taken from.
    quantile : float or None
        Quantile of the eligible items' units the item's units are nearest; None takes the top-ranked item with no
        stockout-like zeros (:data:`MAX_HIGH_ZERO_SHARE`).
    """

    name: str
    cat_id: str
    quantile: float | None


#: The subset's tiers, highest volume first.
TIERS: tuple[Tier, ...] = (
    Tier("high", "FOODS", None),
    Tier("mid", "HOUSEHOLD", 0.5),
    Tier("low", "HOBBIES", 0.1),
)

#: The subset's items by tier: what :func:`pick_items` returns on ``m5_ds.nc``, committed so a run needs no re-derive.
SUBSET_ITEMS: dict[str, str] = {"high": "FOODS_3_586", "mid": "HOUSEHOLD_1_069", "low": "HOBBIES_1_197"}


def eligible_items(ds: xr.Dataset, window: str = SUBSET_WINDOW, n_days: int = ELIGIBLE_DAYS) -> pd.DataFrame:
    """The items with a listed price in every store on each of the last ``n_days`` fit days, ranked by units.

    Parameters
    ----------
    ds : xr.Dataset
        The M5 dataset: ``sales`` and ``sell_price`` on ``(series_id, date)``, ``item_id``, ``cat_id`` and
        ``store_id`` on ``series_id`` and ``split`` on ``date``.
    window : str
        Forecast origin whose fit window is used, a key of ``m5_score_submission.WINDOWS``.
    n_days : int
        Days at the end of the fit window the price must be listed on.

    Returns
    -------
    pd.DataFrame
        One row per eligible item, indexed by ``item_id`` in rank order: ``cat_id``, ``rank`` (1 = most units; ties go
        by ``item_id``), ``units`` (summed over the stores and days), ``units_per_store_day`` and ``zero_share``, the
        median over the stores of the share of days with zero sales.
    """
    fit_mask, _ = window_masks(ds["split"].values, window)
    days = np.flatnonzero(fit_mask)[-n_days:]
    sales = ds["sales"].values[:, days]
    stores = ds["store_id"].values
    series = pd.DataFrame(
        {
            "item_id": ds["item_id"].values,
            "cat_id": ds["cat_id"].values,
            "store_id": stores,
            "listed": np.isfinite(ds["sell_price"].values[:, days]).all(axis=1),
            "units": sales.sum(axis=1, dtype=np.float64),
            "zero_share": (sales == 0).mean(axis=1),
        }
    )
    items = series.groupby("item_id").agg(
        cat_id=("cat_id", "first"),
        n_stores=("store_id", "nunique"),
        listed=("listed", "all"),
        units=("units", "sum"),
        zero_share=("zero_share", "median"),
    )
    n_stores = np.unique(stores).size
    items = items[items["listed"] & (items["n_stores"] == n_stores)]
    # groupby sorts by item_id, so a stable sort breaks ties in units by item_id.
    items = items.sort_values("units", ascending=False, kind="stable")
    items.insert(1, "rank", np.arange(1, len(items) + 1))
    items["units_per_store_day"] = items["units"] / (n_stores * days.size)
    return items[["cat_id", "rank", "units", "units_per_store_day", "zero_share"]]


def pick_items(eligible: pd.DataFrame, tiers: tuple[Tier, ...] = TIERS) -> dict[str, str]:
    """The item of each tier, by the rule in the module docstring.

    Parameters
    ----------
    eligible : pd.DataFrame
        Output of :func:`eligible_items`.
    tiers : tuple of Tier
        The tiers to fill.

    Returns
    -------
    dict of str to str
        Tier name -> ``item_id``, in ``tiers`` order.

    Raises
    ------
    ValueError
        If a tier's category has no eligible item, or no item the high tier accepts.
    """
    picked = {}
    for tier in tiers:
        candidates = eligible[eligible["cat_id"] == tier.cat_id]
        if tier.quantile is None:
            candidates = candidates[candidates["zero_share"] < MAX_HIGH_ZERO_SHARE]
        if candidates.empty:
            raise ValueError(f"No eligible {tier.cat_id} item for the {tier.name} tier")
        if tier.quantile is None:
            picked[tier.name] = str(candidates.index[0])  # rank order, so the top-ranked one
        else:
            target = np.quantile(eligible["units"], tier.quantile)
            # idxmin keeps the first of equally near items, which is the higher-ranked one.
            picked[tier.name] = str((candidates["units"] - target).abs().idxmin())
    return picked


def subset_series_idx(ds: xr.Dataset, items: tuple[str, ...] | None = None) -> np.ndarray:
    """Positions along ``series_id`` of every store series of the items, item by item, stores in dataset order.

    Parameters
    ----------
    ds : xr.Dataset
        The M5 dataset.
    items : tuple of str, optional
        ``item_id`` of each item, by default the :data:`SUBSET_ITEMS` from high to low.

    Returns
    -------
    np.ndarray
        Integer positions: ``n_stores`` per item.

    Raises
    ------
    ValueError
        If an item is not in the dataset.
    """
    items = tuple(SUBSET_ITEMS.values()) if items is None else items
    item_ids = ds["item_id"].values
    missing = [item for item in items if item not in set(item_ids)]
    if missing:
        raise ValueError(f"Items not in the dataset: {missing}")
    return np.concatenate([np.flatnonzero(item_ids == item) for item in items])


def per_item_wrmsse(ds: xr.Dataset, forecasts: np.ndarray, series_order: np.ndarray, window: str) -> pd.DataFrame:
    """The 12-level WRMSSE of each item's series on their own, with the item's share of the dollar sales.

    Each item is scored by ``compute_hierarchical_wrmsse`` on its own series, so its total, store and item levels
    are sums of those series only. The weight is the item's share of the dollar sales of all the given series over
    the last 28 fit days, as the scorer weights the items at level 10.

    Parameters
    ----------
    ds : xr.Dataset
        The M5 dataset.
    forecasts : np.ndarray
        Point forecasts over the window, ``(n_series, 28)``, one row per entry of ``series_order``.
    series_order : np.ndarray
        ``series_id`` of each forecast row.
    window : str
        Scoring window, a key of ``m5_score_submission.WINDOWS``.

    Returns
    -------
    pd.DataFrame
        One row per item, in order of first appearance: ``item_id``, ``tier`` (its :data:`SUBSET_ITEMS` tier, or
        ``""``), ``n_series``, ``weight`` and ``wrmsse``.
    """
    position = {sid: i for i, sid in enumerate(ds["series_id"].values)}
    idx = np.array([position[sid] for sid in series_order])
    item_ids = ds["item_id"].values[idx]
    train_mask, _ = window_masks(ds["split"].values, window)
    dollars = dollar_sales(ds["sales"].values[idx], ds["sell_price"].values[idx], train_mask)
    tier_of = {item: tier for tier, item in SUBSET_ITEMS.items()}
    rows = []
    for item in pd.unique(item_ids):
        rows_of_item = item_ids == item
        per_level = compute_hierarchical_wrmsse(ds, forecasts[rows_of_item], series_order[rows_of_item], window=window)
        rows.append(
            {
                "item_id": item,
                "tier": tier_of.get(item, ""),
                "n_series": int(rows_of_item.sum()),
                "weight": float(dollars[rows_of_item].sum() / dollars.sum()) if dollars.sum() > 0 else np.nan,
                "wrmsse": float(per_level["wrmsse"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-derive the items of the 3-item x 10-store M5 benchmark subset.")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR, metavar="DIR", help="Directory containing m5_ds.nc.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    eligible = eligible_items(load_m5_ds(args.data_dir))
    logger.info(
        "%d eligible items (a listed price in every store on the last %d fit days)", len(eligible), ELIGIBLE_DAYS
    )
    picked = pick_items(eligible)
    for tier, item in picked.items():
        row = eligible.loc[item]
        logger.info(
            "  %-4s  %-16s  rank %5d  %6.2f units / store / day  zero days (median store) %5.1f%%",
            tier,
            item,
            row["rank"],
            row["units_per_store_day"],
            100 * row["zero_share"],
        )
    if picked != SUBSET_ITEMS:
        raise SystemExit(f"m5_subset.py: the rule picks {picked}, but SUBSET_ITEMS is {SUBSET_ITEMS}")
    logger.info("The rule picks SUBSET_ITEMS.")


if __name__ == "__main__":
    main()
