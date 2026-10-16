#!/usr/bin/env python3
"""Check the local WRMSSE weights against the official per-series weights.

The organisers published the weight of every one of the 42,840 series (12 levels) for both scoring
windows as ``weights_validation.csv`` / ``weights_evaluation.csv``; ``m5_fetch.py`` downloads them.
This script compares them with :func:`m5_score_submission.level_weights`, which runs the scorer's own
weighting code, and exits non-zero if any weight differs by more than ``--tol``.

Usage
-----
    python playground/m5_prototype/m5_check_weights.py                      # both windows
    python playground/m5_prototype/m5_check_weights.py --window validation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from m5_common import DATA_DIR, load_m5_ds
from m5_score_submission import WINDOWS, level_weights

# The official files name each group by (Agg_Level_1, Agg_Level_2), with "X" for an unused slot. Their key
# order matches ``_LEVELS`` except at level 11, which the official file lists as (state, item).
_SWAPPED_LEVELS = {11}


def official_weights(path: Path | str) -> pd.DataFrame:
    """Read an official weights file into the ``level`` / ``group`` / ``weight`` layout of ``level_weights``.

    Parameters
    ----------
    path : Path | str
        ``weights_validation.csv`` or ``weights_evaluation.csv``.

    Returns
    -------
    pd.DataFrame
        Columns ``level`` (int), ``group`` (keys in ``_LEVELS`` order joined by ``"|"``), ``dollar_sales``
        and ``weight``.
    """
    w = pd.read_csv(path, dtype={"Agg_Level_1": str, "Agg_Level_2": str})
    level = w["Level_id"].str.removeprefix("Level").astype(int)
    first, second = w["Agg_Level_1"], w["Agg_Level_2"]
    swapped = level.isin(_SWAPPED_LEVELS)
    first, second = first.where(~swapped, w["Agg_Level_2"]), second.where(~swapped, w["Agg_Level_1"])
    group = first.where(second == "X", first + "|" + second)
    return pd.DataFrame({"level": level, "group": group, "dollar_sales": w["Dollar_Sales"], "weight": w["weight"]})


def compare(ds, window: str, data_dir: Path) -> pd.DataFrame:
    """Return per-level differences between the local and the official weights for one window."""
    path = data_dir / f"weights_{window}.csv"
    if not path.is_file():
        raise SystemExit(f"{path} not found; run m5_fetch.py first.")
    merged = level_weights(ds, window).merge(
        official_weights(path), on=["level", "group"], how="outer", suffixes=("_local", "_official"), indicator=True
    )
    unmatched = merged[merged["_merge"] != "both"]
    if len(unmatched):
        raise SystemExit(f"{len(unmatched)} groups match on one side only, e.g. {unmatched.iloc[0].to_dict()}")
    merged["weight_diff"] = (merged["weight_local"] - merged["weight_official"]).abs()
    merged["dollar_diff"] = (merged["dollar_sales_local"] - merged["dollar_sales_official"]).abs()
    return merged.groupby("level").agg(
        n_groups=("group", "size"), max_weight_diff=("weight_diff", "max"), max_dollar_diff=("dollar_diff", "max")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="directory with m5_ds.nc and weights_*.csv")
    parser.add_argument("--window", choices=(*WINDOWS, "both"), default="both")
    parser.add_argument("--tol", type=float, default=1e-8, help="max allowed absolute weight difference")
    args = parser.parse_args()

    ds = load_m5_ds(args.data_dir)
    ok = True
    for window in WINDOWS if args.window == "both" else (args.window,):
        table = compare(ds, window, args.data_dir)
        worst = table["max_weight_diff"].max()
        passed = worst <= args.tol
        ok &= passed
        print(
            f"{window}: {int(table['n_groups'].sum())} groups, max |weight diff| {worst:.2e} "
            f"(tol {args.tol:.0e}) -> {'OK' if passed else 'MISMATCH'}"
        )
        print(table.to_string(float_format=lambda v: f"{v:.2e}"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
