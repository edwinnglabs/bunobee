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

Usage
-----
    python playground/m5_prototype/m5_make_accuracy_fixture.py

Writes the CSV (long format ``date,series_id,sales``, sorted by ``(date, series_id)``) and prints its
SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import xarray as xr

_HERE = Path(__file__).resolve().parent
_DEFAULT_DATA_DIR = _HERE.parent / "resource" / "m5-forecasting-accuracy"
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


def _load_m5_preprocess() -> ModuleType:
    """Import the sibling ``m5_preprocess`` module by path.

    Loading by path rather than by name keeps this script importable from anywhere — the test suite
    pulls it in from ``tests/`` — without mutating ``sys.path``.

    Returns
    -------
    ModuleType
        The imported ``m5_preprocess`` module.
    """
    spec = importlib.util.spec_from_file_location("m5_preprocess", _HERE / "m5_preprocess.py")
    if spec is None or spec.loader is None:  # pragma: no cover - only on a broken checkout
        raise ImportError(f"cannot load m5_preprocess from {_HERE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    states = np.asarray(ds["state_id"].values, dtype=str)
    categories = np.asarray(ds["cat_id"].values, dtype=str)
    labels = np.char.add(np.char.add(states, "_"), categories)
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


def build_m5_aggregate(data_dir: Path | str = _DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Build the committed fixture's contents from the raw M5 dump.

    Parameters
    ----------
    data_dir : Path | str
        Directory holding the raw M5 CSV files.

    Returns
    -------
    pd.DataFrame
        Long panel with columns ``date`` (datetime64), ``series_id`` (str), ``sales`` (int64),
        sorted by ``(date, series_id)``, covering the last ``N_DATES`` observed days.
    """
    build_m5_dataset = _load_m5_preprocess().build_m5_dataset
    wide = repair_closures(aggregate_state_cat(build_m5_dataset(data_dir))).iloc[-N_DATES:]

    long = wide.stack().rename("sales").reset_index()
    long.columns = ["date", "series_id", "sales"]
    return long.sort_values(["date", "series_id"]).reset_index(drop=True)


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
    print(f"file        : {args.output} ({args.output.stat().st_size / 1e3:.0f} KB)")
    print(f"sha256      : {digest}")


if __name__ == "__main__":
    main()
