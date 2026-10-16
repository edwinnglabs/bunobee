#!/usr/bin/env python3
"""Score an M5 submission CSV against known labels for one scoring window.

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

Two details follow the M5 Competitors Guide, so a local score is comparable to a
published one:

- The RMSSE scale of each (aggregated) series uses its training history from its
  first non-zero sale on; the leading zeros of a late launch are not counted.
- Dollar sales skip days with no listed price, so a price gap drops those days
  from a series' weight instead of zeroing all of it.

Scoring windows (``--window``):

    validation : actuals d_1914-d_1941, trained on d_1-d_1913  (``_validation`` rows)
    evaluation : actuals d_1942-d_1969, trained on d_1-d_1941  (``_evaluation`` rows)

The Kaggle files stop at d_1941 (see ``m5_fetch.py``), so the evaluation window
only scores once the dataset's ``test`` split carries the released actuals.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from m5_common import DATA_DIR, load_m5_ds

logger = logging.getLogger(__name__)

N_STEPS = 28

# Scoring window -> (splits forming its training span, split holding its actuals, day range for logs). The
# training span feeds the RMSSE scale, and its last N_STEPS days feed the dollar-sales weights.
_WINDOWS: dict[str, tuple[tuple[str, ...], str, str]] = {
    "validation": (("train",), "validation", "d_1914-d_1941"),
    "evaluation": (("train", "validation"), "test", "d_1942-d_1969"),
}

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
    p = argparse.ArgumentParser(description="Score an M5 submission CSV against known labels.")
    p.add_argument("--submission", type=Path, required=True, metavar="CSV", help="Submission CSV to score.")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        metavar="DIR",
        help="Directory containing m5_ds.nc.",
    )
    p.add_argument(
        "--window",
        choices=tuple(_WINDOWS),
        default="validation",
        help="Scoring window: validation (d_1914-d_1941) or evaluation (d_1942-d_1969).",
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


def _rmsse_scale(y_train: np.ndarray) -> np.ndarray:
    """Per-series RMSSE scale: the mean squared one-step difference from the first non-zero sale on.

    The M5 Competitors Guide starts each series' scale at its first non-zero observation, so the leading
    zeros of a late launch add neither zero-difference terms nor the launch jump to the denominator.

    Parameters
    ----------
    y_train : np.ndarray, shape (n_series, T)
        Training history, one (possibly aggregated) series per row.

    Returns
    -------
    np.ndarray, shape (n_series,)
        Scale per series. A series with no one-step difference left after trimming, or a flat one,
        gets 1.0 so its RMSSE stays finite.
    """
    started = np.logical_or.accumulate(y_train != 0.0, axis=1)
    # ``started`` is monotone, so a difference lies in the trimmed span exactly when its earlier day does.
    in_span = started[:, :-1]
    sq_diff = np.where(in_span, np.diff(y_train, axis=1) ** 2, 0.0)
    scale = sq_diff.sum(axis=1) / np.maximum(in_span.sum(axis=1), 1)
    return np.where(scale <= 0.0, 1.0, scale)


def _level_stats(
    y_train_agg: np.ndarray,
    y_score_agg: np.ndarray,
    f_score_agg: np.ndarray,
    dollar_agg: np.ndarray,
) -> dict[str, float]:
    """Return WRMSSE plus bias diagnostics for one level."""
    denom = np.sqrt(_rmsse_scale(y_train_agg))

    err = f_score_agg - y_score_agg
    mse = np.mean(err**2, axis=1)
    rmsse = np.sqrt(mse) / denom

    weights = dollar_agg / dollar_agg.sum() if dollar_agg.sum() > 0 else np.ones_like(dollar_agg) / len(dollar_agg)

    actual_mean = y_score_agg.mean(axis=1)
    forecast_mean = f_score_agg.mean(axis=1)
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


def _submission_rows(sub: pd.DataFrame, window: str) -> tuple[np.ndarray, np.ndarray]:
    """Pick one scoring window's rows out of a submission frame.

    Parameters
    ----------
    sub : pd.DataFrame
        Submission with an ``id`` column (``<series_id>_validation`` / ``<series_id>_evaluation``) and
        forecast columns ``F1`` ... ``F28``.
    window : {"validation", "evaluation"}
        Scoring window; its name is the ``id`` suffix of its rows.

    Returns
    -------
    forecasts : np.ndarray, shape (n_rows, N_STEPS)
        Forecasts of the window's rows.
    series_order : np.ndarray, shape (n_rows,)
        ``series_id`` of each row, with the window suffix stripped.
    """
    suffix = f"_{window}"
    rows = sub[sub["id"].str.endswith(suffix)]
    f_cols = [f"F{i}" for i in range(1, N_STEPS + 1)]
    return rows[f_cols].values.astype(np.float64), rows["id"].str.removesuffix(suffix).values


def compute_hierarchical_wrmsse(
    ds: xr.Dataset,
    forecasts: np.ndarray,
    series_order: np.ndarray,
    window: str = "validation",
) -> pd.DataFrame:
    """Compute WRMSSE across all 12 M5 aggregation levels for one scoring window.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from ``m5_preprocess.build_m5_dataset``: ``sales`` and ``sell_price`` on
        ``(series_id, date)``, the hierarchy keys on ``series_id`` and ``split`` on ``date``.
    forecasts : np.ndarray, shape (n_series, N_STEPS)
        Point forecasts over the scoring window, one row per entry of ``series_order``.
    series_order : np.ndarray, shape (n_series,)
        ``series_id`` of each forecast row.
    window : {"validation", "evaluation"}, optional
        Scoring window, a key of ``_WINDOWS``. Default ``"validation"``.

    Returns
    -------
    pd.DataFrame
        One row per level: ``level``, ``name``, ``n_series``, ``wrmsse`` and the bias diagnostics.

    Raises
    ------
    ValueError
        If ``window`` is not a key of ``_WINDOWS``.
    RuntimeError
        If the window is not ``N_STEPS`` days long or the dataset has no actuals for it.
    """
    if window not in _WINDOWS:
        raise ValueError(f"Unknown window {window!r}; expected one of {sorted(_WINDOWS)}")
    train_splits, score_split, days = _WINDOWS[window]

    ds_ids = ds["series_id"].values
    id_to_idx = {sid: i for i, sid in enumerate(ds_ids)}
    idx = np.array([id_to_idx[sid] for sid in series_order])

    split = ds["split"].values
    train_mask = np.isin(split, train_splits)
    score_mask = split == score_split

    sales = ds["sales"].values[idx].astype(np.float64)
    prices = ds["sell_price"].values[idx].astype(np.float64)
    y_train = sales[:, train_mask]
    y_score = sales[:, score_mask]

    if y_score.shape[1] != N_STEPS:
        raise RuntimeError(f"Expected {window} length {N_STEPS}, got {y_score.shape[1]}")
    if not np.isfinite(y_score).all():
        raise RuntimeError(f"The dataset has no {window} actuals ({days}); see m5_fetch.py for where they come from")

    # nansum, not sum: a day with no listed price drops out of the weight instead of zeroing the series.
    last_28_idx = np.where(train_mask)[0][-N_STEPS:]
    dollar_per_series = np.nansum(sales[:, last_28_idx] * prices[:, last_28_idx], axis=1)

    meta = {key: ds[key].values[idx] for key in ("state_id", "store_id", "cat_id", "dept_id", "item_id")}

    rows: list[dict] = []
    for level_no, name, keys in _LEVELS:
        if keys is None:
            group_idx = np.zeros(len(idx), dtype=np.int64)
            n_groups = 1
        else:
            group_idx, n_groups = _group_index([meta[k] for k in keys])

        y_train_agg = _aggregate(y_train, group_idx, n_groups)
        y_score_agg = _aggregate(y_score, group_idx, n_groups)
        f_score_agg = _aggregate(forecasts, group_idx, n_groups)
        dollar_agg = np.zeros(n_groups, dtype=np.float64)
        np.add.at(dollar_agg, group_idx, dollar_per_series)

        stats = _level_stats(y_train_agg, y_score_agg, f_score_agg, dollar_agg)
        rows.append({"level": level_no, "name": name, "n_series": n_groups, **stats})

    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.submission.exists():
        raise SystemExit(f"Submission not found: {args.submission}")

    logger.info("Loading submission from %s", args.submission)
    sub = pd.read_csv(args.submission)
    forecasts, series_order = _submission_rows(sub, args.window)
    logger.info("Found %d _%s rows", len(series_order), args.window)

    logger.info("Loading dataset from %s", args.data_dir)
    ds = load_m5_ds(args.data_dir)

    per_level = compute_hierarchical_wrmsse(ds, forecasts, series_order, window=args.window)
    final_wrmsse = float(per_level["wrmsse"].mean())

    logger.info("Per-level WRMSSE + bias (%s %s):", args.window, _WINDOWS[args.window][2])
    logger.info(
        "  %-3s  %-22s  %-6s  %-8s  %-10s  %-10s  %-10s  %-8s",
        "lvl",
        "name",
        "n",
        "wrmsse",
        "actual_mean",
        "fcst_mean",
        "bias",
        "bias_%",
    )
    for _, row in per_level.iterrows():
        logger.info(
            "  L%-2d  %-22s  n=%-6d  %.4f    %10.2f  %10.2f  %+10.2f  %+7.1f%%",
            int(row["level"]),
            row["name"],
            int(row["n_series"]),
            row["wrmsse"],
            row["actual_mean"],
            row["forecast_mean"],
            row["bias_signed"],
            row["bias_pct"],
        )
    logger.info("Hierarchical WRMSSE (mean of 12 levels): %.6f", final_wrmsse)


if __name__ == "__main__":
    main()
