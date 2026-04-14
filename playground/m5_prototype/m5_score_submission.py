#!/usr/bin/env python3
"""Score an M5 submission CSV against known validation labels (d_1914-d_1941).

Computes the full 12-level hierarchical WRMSSE that Kaggle uses:

    Level 1  : Total                        (1 series)
    Level 2  : state_id                     (3)
    Level 3  : store_id                     (10)
    Level 4  : cat_id                       (3)
    Level 5  : dept_id                      (7)
    Level 6  : state_id x cat_id            (9)
    Level 7  : state_id x dept_id           (21)
    Level 8  : store_id x cat_id            (30)
    Level 9  : store_id x dept_id           (70)
    Level 10 : item_id                      (3,049)
    Level 11 : item_id x state_id           (9,147)
    Level 12 : item_id x store_id           (30,490, bottom)

For each level: aggregate sales and forecasts by sum, compute per-group RMSSE
(training-difference scale), weight by dollar sales over the last 28 train days
normalised within the level, and take the weighted mean. Final WRMSSE is the
simple mean of the 12 level scores.

Test-window scoring (d_1942-d_1969) is skipped — the evaluation labels were
never publicly released.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


logger = logging.getLogger(__name__)

N_STEPS = 28
_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "resource" / "m5-forecasting-accuracy"

# (level_number, human_name, grouping_keys). None = no grouping (Total level).
_LEVELS: list[tuple[int, str, tuple[str, ...] | None]] = [
    (1, "Total", None),
    (2, "state_id", ("state_id",)),
    (3, "store_id", ("store_id",)),
    (4, "cat_id", ("cat_id",)),
    (5, "dept_id", ("dept_id",)),
    (6, "state_id x cat_id", ("state_id", "cat_id")),
    (7, "state_id x dept_id", ("state_id", "dept_id")),
    (8, "store_id x cat_id", ("store_id", "cat_id")),
    (9, "store_id x dept_id", ("store_id", "dept_id")),
    (10, "item_id", ("item_id",)),
    (11, "item_id x state_id", ("item_id", "state_id")),
    (12, "item_id x store_id", ("item_id", "store_id")),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score an M5 submission CSV against validation labels.")
    p.add_argument("--submission", type=Path, required=True, metavar="CSV", help="Submission CSV to score.")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        metavar="DIR",
        help="Directory containing m5_ds.nc.",
    )
    return p.parse_args()


def _group_index(keys: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """Return (group_index_per_series, n_groups) for the given key columns."""
    if len(keys) == 1:
        labels = keys[0]
    else:
        labels = np.array(["|".join(k[i] for k in keys) for i in range(len(keys[0]))])
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse, int(inverse.max()) + 1


def _aggregate(matrix: np.ndarray, group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum rows of `matrix` by group index. matrix: (n_series, T) -> (n_groups, T)."""
    out = np.zeros((n_groups, matrix.shape[1]), dtype=np.float64)
    np.add.at(out, group_idx, matrix)
    return out


def _level_stats(
    y_train_agg: np.ndarray,
    y_val_agg: np.ndarray,
    f_val_agg: np.ndarray,
    dollar_agg: np.ndarray,
) -> dict[str, float]:
    """Return WRMSSE plus bias diagnostics for one level."""
    diff = np.diff(y_train_agg, axis=1)
    scale = np.mean(diff ** 2, axis=1)
    scale = np.where(scale <= 0.0, 1.0, scale)
    denom = np.sqrt(scale)

    err = f_val_agg - y_val_agg
    mse = np.mean(err ** 2, axis=1)
    rmsse = np.sqrt(mse) / denom

    weights = dollar_agg / dollar_agg.sum() if dollar_agg.sum() > 0 else np.ones_like(dollar_agg) / len(dollar_agg)

    actual_mean = y_val_agg.mean(axis=1)
    forecast_mean = f_val_agg.mean(axis=1)
    bias = forecast_mean - actual_mean
    rel_bias = bias / np.where(actual_mean != 0.0, actual_mean, 1.0)

    return {
        "wrmsse": float(np.sum(weights * rmsse)),
        "mean_rmsse": float(rmsse.mean()),
        "actual_mean": float((weights * actual_mean).sum()),
        "forecast_mean": float((weights * forecast_mean).sum()),
        "bias_signed": float((weights * bias).sum()),
        "bias_pct": float((weights * rel_bias).sum() * 100.0),
    }


def compute_hierarchical_wrmsse(
    ds: xr.Dataset,
    forecasts: np.ndarray,
    series_order: np.ndarray,
) -> pd.DataFrame:
    """Compute WRMSSE across all 12 M5 aggregation levels."""
    ds_ids = ds["series_id"].values
    id_to_idx = {sid: i for i, sid in enumerate(ds_ids)}
    idx = np.array([id_to_idx[sid] for sid in series_order])

    split = ds["split"].values
    train_mask = split == "train"
    val_mask = split == "validation"

    sales = ds["sales"].values[idx].astype(np.float64)
    prices = ds["sell_price"].values[idx].astype(np.float64)
    y_train = sales[:, train_mask]
    y_val = sales[:, val_mask]

    if y_val.shape[1] != N_STEPS:
        raise RuntimeError(f"Expected validation length {N_STEPS}, got {y_val.shape[1]}")

    train_indices = np.where(train_mask)[0]
    last_28_idx = train_indices[-N_STEPS:]
    dollar_per_series = (sales[:, last_28_idx] * prices[:, last_28_idx]).sum(axis=1)
    dollar_per_series = np.where(np.isnan(dollar_per_series), 0.0, dollar_per_series)

    meta = {key: ds[key].values[idx] for key in ("state_id", "store_id", "cat_id", "dept_id", "item_id")}

    rows: list[dict] = []
    for level_no, name, keys in _LEVELS:
        if keys is None:
            group_idx = np.zeros(len(idx), dtype=np.int64)
            n_groups = 1
        else:
            group_idx, n_groups = _group_index([meta[k] for k in keys])

        y_train_agg = _aggregate(y_train, group_idx, n_groups)
        y_val_agg = _aggregate(y_val, group_idx, n_groups)
        f_val_agg = _aggregate(forecasts, group_idx, n_groups)
        dollar_agg = np.zeros(n_groups, dtype=np.float64)
        np.add.at(dollar_agg, group_idx, dollar_per_series)

        stats = _level_stats(y_train_agg, y_val_agg, f_val_agg, dollar_agg)
        rows.append({"level": level_no, "name": name, "n_series": n_groups, **stats})

    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.submission.exists():
        raise SystemExit(f"Submission not found: {args.submission}")

    logger.info("Loading submission from %s", args.submission)
    sub = pd.read_csv(args.submission)
    f_cols = [f"F{i}" for i in range(1, N_STEPS + 1)]

    val_rows = sub[sub["id"].str.endswith("_validation")].copy()
    logger.info("Found %d _validation rows", len(val_rows))

    val_rows["series_id"] = val_rows["id"].str.replace("_validation$", "", regex=True)
    forecasts = val_rows[f_cols].values.astype(np.float64)
    series_order = val_rows["series_id"].values

    ds_path = args.data_dir / "m5_ds.nc"
    logger.info("Loading dataset from %s", ds_path)
    ds = xr.open_dataset(ds_path)

    per_level = compute_hierarchical_wrmsse(ds, forecasts, series_order)
    final_wrmsse = float(per_level["wrmsse"].mean())

    logger.info("Per-level WRMSSE + bias (validation d_1914-d_1941):")
    logger.info(
        "  %-3s  %-22s  %-6s  %-8s  %-10s  %-10s  %-10s  %-8s",
        "lvl", "name", "n", "wrmsse", "actual_mean", "fcst_mean", "bias", "bias_%",
    )
    for _, row in per_level.iterrows():
        logger.info(
            "  L%-2d  %-22s  n=%-6d  %.4f    %10.2f  %10.2f  %+10.2f  %+7.1f%%",
            int(row["level"]), row["name"], int(row["n_series"]), row["wrmsse"],
            row["actual_mean"], row["forecast_mean"], row["bias_signed"], row["bias_pct"],
        )
    logger.info("Hierarchical WRMSSE (mean of 12 levels): %.6f", final_wrmsse)


if __name__ == "__main__":
    main()
