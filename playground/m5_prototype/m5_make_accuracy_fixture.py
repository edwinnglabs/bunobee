#!/usr/bin/env python3
"""Build ``src/bunobee/datasets/data/m5_aggregate.csv`` — the hermetic panel behind the accuracy guard.

Source
------
M5 Forecasting — Accuracy (Kaggle, https://www.kaggle.com/competitions/m5-forecasting-accuracy).
The raw competition dump lives untracked in ``playground/resource/m5-forecasting-accuracy/`` (980 MB,
gitignored by ``*.csv`` / ``*.nc``), so CI never sees it.

The committed CSV is a **derived aggregate** — 10 summed series over a 756-day window — not a
redistribution of the raw competition data. It is packaged (not test-only) so
``tests/test_m5_accuracy.py`` and any notebook or prototype script share the same panel via
``bunobee.datasets.load_m5_aggregate()``.

What it contains
----------------
Raw M5 item-store series are intermittent and zero-inflated, which makes MAPE meaningless on them.
Summing to ``state_id x cat_id`` gives 9 dense series (thousands of units/day); a national ``TOTAL``
over all 30 490 item-store series makes 10. Only ``split in {"train", "validation"}`` dates are used
(``d_1``–``d_1941``); ``d_1942`` onward carries NaN sales in the competition files.

Christmas closure
-----------------
Every US store in M5 is closed on 25 December: on those five dates the whole panel collapses to a
handful of units (e.g. ``TOTAL = 14`` against a ~30 000 baseline), and six of the nine
``state x cat`` aggregates read exactly zero. Those are the only zeros anywhere in the panel. A hard
zero makes MAPE undefined and the near-zero neighbours make it explode, so the closure dates are
repaired in place with the same-weekday average of the surrounding weeks — ``(y[t - 7] + y[t + 7]) /
2``, rounded to the nearest unit. The repair is deterministic and is re-applied by the provenance
test, so the committed copy still cannot drift.

Calendar covariates
-------------------
Stage 1 of #143 (``m5_models.CALENDAR_COVARIATES``) in columns a demo can read without the raw dump, one value per
date, repeated on each of its rows:

- ``annual_sin_1``, ``annual_cos_1``: the annual Fourier terms, ``make_fourier_series`` columns for the stage 1 period
  and order. Step 0 is the first day of the M5 calendar, 2011-01-29, as in the stage 1 design, not the first day of
  this window.
- ``event_name_1``, ``event_name_2``: the M5 event names, ``""`` on a day without one. They stay names, because one
  0/1 column per event would not fit the file's size budget.
- ``month_of_year`` (1-12), ``day_of_month`` (1-31).

Price index
-----------
``price_index`` is a fixed-basket (Lowe) price index of each aggregate: what the units the aggregate sold in the first
``N_STEPS`` days of the window would cost at the day's prices, over what they cost at their mean listed price in those
days. With ``q_i`` the item-store's units in those days and ``p_i`` its mean listed price in them,

    price_index[t] = sum_i(q_i * price_i[t]) / sum_i(q_i * p_i)

over the item-store series in the aggregate that are listed on day ``t``, so a listing starting or ending does not move
the index by itself. A fixed basket is what makes it a price measure: the basket does not move with the day's sales, so
a shift in the sales mix cannot read as a price change, and the index is near 1 over the ``N_STEPS`` days. The basket
and the mean prices stop at ``N_STEPS``, so the last ``HORIZON`` days' sales stay out of it; those days' prices are in
it, since M5 publishes prices ahead of the sales. The units are the raw ones, before the Christmas repair. The values
are rounded to 6 decimals, as are the Fourier terms.

Usage
-----
    python playground/m5_prototype/m5_make_accuracy_fixture.py

Writes the CSV (long format, sorted by ``(date, series_id)``, columns :data:`PANEL_COLUMNS`) and prints its SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import xarray as xr

from bunobee.regression import make_fourier_series

_HERE = Path(__file__).resolve().parent


def _load_sibling(name: str) -> ModuleType:
    """Import a sibling playground module by path and register it under its own name.

    Loading by path rather than by name keeps this script importable from anywhere — the test suite
    pulls it in from ``tests/`` — without mutating ``sys.path``. Registering the module in
    ``sys.modules`` lets a sibling's own ``from m5_common import ...`` resolve to it.

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
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover - only on a broken checkout
        raise ImportError(f"cannot load {name} from {_HERE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]  # as the import system does, so a retry does not see a half-run module
        raise
    return module


_DEFAULT_DATA_DIR = _load_sibling("m5_common").DATA_DIR
_DEFAULT_OUTPUT = _HERE.parents[1] / "src" / "bunobee" / "datasets" / "data" / "m5_aggregate.csv"

# Window: the training span plus one M5 evaluation window, ending at d_1941.
N_STEPS = 728
HORIZON = 28
N_DATES = N_STEPS + HORIZON

TOTAL_SERIES = "TOTAL"
N_SERIES = 10  # 3 states x 3 categories, plus the national total
PERIOD = 7  # weekly cycle, used to repair the Christmas closure

# Splits carrying observed sales; d_1942+ ("test") is NaN in the competition files.
_OBSERVED_SPLITS = ("train", "validation")

_EVENT_COLUMNS = ("event_name_1", "event_name_2")
_DECIMALS = 6  # of the float columns: enough for a covariate, and a platform's last bits cannot reach the digest

#: The file's columns, in order. ``annual_*`` follows ``m5_models.CALENDAR_COVARIATES``.
PANEL_COLUMNS = (
    "date",
    "series_id",
    "sales",
    "price_index",
    "annual_sin_1",
    "annual_cos_1",
    *_EVENT_COLUMNS,
    "month_of_year",
    "day_of_month",
)


def _aggregate_labels(ds: xr.Dataset) -> np.ndarray:
    """Name the ``state_id x cat_id`` aggregate of every item-store series.

    Parameters
    ----------
    ds : xr.Dataset
        M5 dataset with ``state_id`` and ``cat_id`` coords on ``series_id``.

    Returns
    -------
    np.ndarray
        Aggregate names such as ``"CA_FOODS"``, one per series.
    """
    states = np.asarray(ds["state_id"].values, dtype=str)
    categories = np.asarray(ds["cat_id"].values, dtype=str)
    return np.char.add(np.char.add(states, "_"), categories)


def aggregate_state_cat(ds: xr.Dataset) -> pd.DataFrame:
    """Sum item-store sales into the 9 ``state_id x cat_id`` series plus a national total.

    Parameters
    ----------
    ds : xr.Dataset
        M5 dataset from ``m5_preprocess.build_m5_dataset``, with ``sales`` on
        ``(series_id, date)`` and ``state_id`` / ``cat_id`` coords on ``series_id``.

    Returns
    -------
    pd.DataFrame
        Wide panel indexed by date (observed splits only) with 10 integer columns, sorted by name.
    """
    observed = np.isin(np.asarray(ds["split"].values), _OBSERVED_SPLITS)
    sales = np.asarray(ds["sales"].values, dtype=np.float64)[:, observed]
    dates = pd.DatetimeIndex(np.asarray(ds["date"].values)[observed])

    labels = _aggregate_labels(ds)
    groups = sorted(np.unique(labels))
    stacked = np.stack([sales[labels == name].sum(axis=0) for name in groups])

    wide = pd.DataFrame(stacked.T, index=dates, columns=groups)
    wide[TOTAL_SERIES] = sales.sum(axis=0)
    wide.index.name = "date"
    return wide[sorted(wide.columns)].round().astype(np.int64)


def closure_dates(wide: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the store-closure dates — the dates on which any aggregate reads exactly zero.

    Parameters
    ----------
    wide : pd.DataFrame
        Wide panel indexed by date, as produced by :func:`aggregate_state_cat`.

    Returns
    -------
    pd.DatetimeIndex
        Closure dates, ascending.

    Raises
    ------
    ValueError
        If a zero shows up on a date that is not 25 December — the panel is then not the
        all-stores-closed panel this builder assumes, and the repair below would be guesswork.
    """
    dates = pd.DatetimeIndex(wide.index[(wide == 0).any(axis=1)])
    unexpected = dates[~((dates.month == 12) & (dates.day == 25))]
    if len(unexpected) > 0:
        raise ValueError(f"zero sales outside the 25 December closure: {list(unexpected.date)}")
    return dates


def repair_closures(wide: pd.DataFrame) -> pd.DataFrame:
    """Replace the closure days with the same-weekday average of the neighbouring weeks.

    Every series is repaired on a closure date, not just the ones that hit zero: the closure is a
    whole-panel event, and the surviving values (a handful of units against a ~30 000 baseline) are
    just as unusable for a MAPE guard as the zeros.

    Parameters
    ----------
    wide : pd.DataFrame
        Wide panel indexed by date, as produced by :func:`aggregate_state_cat`.

    Returns
    -------
    pd.DataFrame
        Copy of ``wide`` with closure rows replaced by ``round((y[t - 7] + y[t + 7]) / 2)``.

    Raises
    ------
    ValueError
        If a closure date lacks a same-weekday neighbour on either side, or if a repaired row still
        holds a non-positive value.
    """
    repaired = wide.copy()
    week = pd.Timedelta(days=PERIOD)
    for date in closure_dates(wide):
        before, after = date - week, date + week
        if before not in wide.index or after not in wide.index:
            raise ValueError(f"closure {date.date()} has no same-weekday neighbour inside the panel")
        repaired.loc[date] = np.rint((wide.loc[before] + wide.loc[after]) / 2.0).astype(np.int64)

    if not (repaired > 0).all().all():
        raise ValueError("repaired panel still holds non-positive sales")
    return repaired


def price_index(ds: xr.Dataset, dates: pd.DatetimeIndex, reference_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute the fixed-basket price index of every aggregate, defined in the module docstring.

    Parameters
    ----------
    ds : xr.Dataset
        M5 dataset with ``sales`` and ``sell_price`` on ``(series_id, date)`` (``sell_price`` NaN where a series is not
        listed) and ``state_id`` / ``cat_id`` coords on ``series_id``.
    dates : pd.DatetimeIndex
        Days to compute the index on.
    reference_dates : pd.DatetimeIndex
        Days whose units and mean listed prices fix the basket.

    Returns
    -------
    pd.DataFrame
        Wide panel indexed by ``dates`` with one column per aggregate plus :data:`TOTAL_SERIES`, sorted by name, rounded
        to 6 decimals.

    Raises
    ------
    ValueError
        If a series sold units in ``reference_dates`` without being listed in any of them, or if an aggregate has no
        basket on a day, so that its index is undefined.
    """
    reference_prices = np.asarray(ds["sell_price"].sel(date=reference_dates).values, dtype=np.float64)
    units = np.nansum(np.asarray(ds["sales"].sel(date=reference_dates).values, dtype=np.float64), axis=1)
    days_listed = np.isfinite(reference_prices).sum(axis=1)
    unpriced = (units > 0) & (days_listed == 0)
    if unpriced.any():
        raise ValueError(f"{int(unpriced.sum())} series sold units without a listed price in the reference days")
    reference_cost = units * np.nansum(reference_prices, axis=1) / np.maximum(days_listed, 1)

    prices = np.asarray(ds["sell_price"].sel(date=dates).values, dtype=np.float64)
    # Only the series with a basket count: one not sold in the reference days has no mean price to measure against.
    listed = np.isfinite(prices) & (units > 0)[:, None]
    cost = np.where(listed, units[:, None] * prices, 0.0)
    cost_at_reference = np.where(listed, reference_cost[:, None], 0.0)

    labels = _aggregate_labels(ds)
    members = {name: labels == name for name in np.unique(labels)} | {TOTAL_SERIES: np.ones(labels.size, dtype=bool)}
    with np.errstate(divide="ignore", invalid="ignore"):
        index = {name: cost[mask].sum(axis=0) / cost_at_reference[mask].sum(axis=0) for name, mask in members.items()}

    wide = pd.DataFrame(index, index=dates)[sorted(members)].round(_DECIMALS)
    wide.index.name = "date"
    if not np.isfinite(wide.to_numpy()).all():
        raise ValueError("an aggregate has no basket on some day, so its price index is undefined")
    return wide


def calendar_covariates(ds: xr.Dataset) -> pd.DataFrame:
    """Collect the calendar columns of the panel over every day of the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        M5 dataset from the first day of the M5 calendar, with ``event_name_1`` and ``event_name_2`` (``""`` or NaN on a
        day without an event), ``month_of_year`` and ``day_of_month`` along ``date``.

    Returns
    -------
    pd.DataFrame
        Frame indexed by ``date``, with the calendar columns of :data:`PANEL_COLUMNS`.
    """
    stage_1 = _load_sibling("m5_models").CALENDAR_COVARIATES
    terms = make_fourier_series(ds.sizes["date"], period=stage_1.annual_period, order=stage_1.annual_order)
    columns = {
        f"annual_{kind}_{k}": np.round(terms[:, offset + k - 1], _DECIMALS)
        for offset, kind in ((0, "sin"), (stage_1.annual_order, "cos"))
        for k in range(1, stage_1.annual_order + 1)
    }
    for name in _EVENT_COLUMNS:
        values = ds[name].values
        columns[name] = np.where(pd.isna(values), "", values.astype(str))
    for name in ("month_of_year", "day_of_month"):
        columns[name] = ds[name].values.astype(np.int64)
    return pd.DataFrame(columns, index=pd.DatetimeIndex(ds["date"].values, name="date"))


def build_m5_aggregate(data_dir: Path | str = _DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Build the committed fixture's contents from the raw M5 dump.

    Parameters
    ----------
    data_dir : Path | str
        Directory holding the raw M5 CSV files.

    Returns
    -------
    pd.DataFrame
        Long panel with columns :data:`PANEL_COLUMNS`: ``date`` (datetime64), ``series_id`` (str), ``sales`` (int64),
        ``price_index`` (float64), and the calendar columns of :func:`calendar_covariates`, sorted by
        ``(date, series_id)``, covering the last ``N_DATES`` observed days.
    """
    ds = _load_sibling("m5_preprocess").build_m5_dataset(data_dir)
    wide = repair_closures(aggregate_state_cat(ds)).iloc[-N_DATES:]

    long = wide.stack().rename("sales").reset_index()
    long.columns = ["date", "series_id", "sales"]
    prices = price_index(ds, wide.index, wide.index[:N_STEPS]).stack().rename("price_index").reset_index()
    prices.columns = ["date", "series_id", "price_index"]
    panel = long.merge(prices, on=["date", "series_id"]).merge(calendar_covariates(ds).reset_index(), on="date")
    return panel[list(PANEL_COLUMNS)].sort_values(["date", "series_id"]).reset_index(drop=True)


def write_fixture(panel: pd.DataFrame, output_path: Path | str = _DEFAULT_OUTPUT) -> str:
    """Write the long panel as CSV and return the file's SHA-256.

    Parameters
    ----------
    panel : pd.DataFrame
        Long panel from :func:`build_m5_aggregate`.
    output_path : Path | str
        Destination CSV path.

    Returns
    -------
    str
        Hex SHA-256 digest of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def main() -> None:
    """Build the aggregate, write the fixture, and report its shape, size, and SHA-256."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR, help="raw M5 CSV directory")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, help="fixture CSV path")
    args = parser.parse_args()

    panel = build_m5_aggregate(args.data_dir)
    digest = write_fixture(panel, args.output)

    dates = panel["date"]
    print(f"series      : {panel['series_id'].nunique()} — {', '.join(sorted(panel['series_id'].unique()))}")
    print(f"dates       : {dates.nunique()} ({dates.min().date()} … {dates.max().date()})")
    print(f"rows        : {len(panel)}")
    print(f"min sales   : {panel['sales'].min()}")
    print(f"price index : {panel['price_index'].min():.6f} … {panel['price_index'].max():.6f}")
    print(f"file        : {args.output} ({args.output.stat().st_size / 1e3:.0f} KB)")
    print(f"sha256      : {digest}")


if __name__ == "__main__":
    main()
