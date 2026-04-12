#!/usr/bin/env python3
"""Diagnostic diff between a run's submission.csv and a reference submission.

Given a timestamped ``output/<TIMESTAMP>`` run directory with a ``submission.csv``
and a reference benchmark submission (default: ``sample_submission2.csv`` in the
resource directory), this script:

1. Aligns the two 60,981-row submissions by ``id``.
2. Computes per-series disagreement stats (MAE, RMSE, Pearson correlation of
   the 28-day forecast vector) and global distribution summaries.
3. Attaches per-series total training sales (from ``m5_ds.nc``) for bucketing.
4. Writes three summary CSVs alongside ``submission.csv``:
   - ``submission_diff_summary.csv`` — global distribution and diff stats.
   - ``submission_diff_top_n.csv``   — top-N series with largest disagreement.
   - ``submission_diff_by_volume.csv`` — stats grouped by training-sales bucket.

This does not treat the reference as ground truth; all metrics are framed as
pairwise disagreement between two forecast sets.
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
EXPECTED_ROWS = 60_980
_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "resource" / "m5-forecasting-accuracy"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diff an M5 submission.csv against a benchmark submission.")
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="m5_prototype/output run directory containing submission.csv",
    )
    p.add_argument(
        "--reference",
        type=Path,
        default=_DEFAULT_DATA_DIR / "sample_submission2.csv",
        metavar="CSV",
        help="Reference benchmark submission CSV to diff against.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        metavar="DIR",
        help="Directory containing m5_ds.nc (for volume bucketing).",
    )
    p.add_argument("--top-n", type=int, default=20, metavar="N", help="Number of worst-disagreement series to list.")
    p.add_argument("--n-buckets", type=int, default=5, metavar="N", help="Number of volume quantile buckets.")
    return p.parse_args()


def _load_submission(path: Path, label: str) -> pd.DataFrame:
    """Load a submission CSV and validate its shape."""
    logger.info("Loading %s submission from %s", label, path)
    df = pd.read_csv(path)
    f_cols = [f"F{i}" for i in range(1, N_STEPS + 1)]
    expected_cols = ["id", *f_cols]
    if list(df.columns) != expected_cols:
        raise ValueError(f"{label} has unexpected columns: {list(df.columns)[:5]}...")
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"{label} has {len(df)} rows, expected {EXPECTED_ROWS}")
    return df


def _align_submissions(run: pd.DataFrame, ref: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align two submissions by id and return (ids, A, B) forecast arrays."""
    if set(run["id"]) != set(ref["id"]):
        missing_in_ref = set(run["id"]) - set(ref["id"])
        missing_in_run = set(ref["id"]) - set(run["id"])
        raise ValueError(
            f"Submission id sets differ — {len(missing_in_ref)} missing in reference, "
            f"{len(missing_in_run)} missing in run."
        )

    ref_sorted = ref.set_index("id").loc[run["id"]].reset_index()
    f_cols = [f"F{i}" for i in range(1, N_STEPS + 1)]
    ids = run["id"].values
    a = run[f_cols].values.astype(np.float64)
    b = ref_sorted[f_cols].values.astype(np.float64)
    return ids, a, b


def _row_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation between rows of a and b; NaN where either row has zero variance."""
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    num = (a_centered * b_centered).sum(axis=1)
    denom = np.sqrt((a_centered ** 2).sum(axis=1) * (b_centered ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0.0, num / denom, np.nan)
    return corr


def _flat_dist_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Summary statistics on a flattened forecast array."""
    flat = values.ravel()
    q25, q50, q75 = np.quantile(flat, [0.25, 0.5, 0.75])
    return {
        f"{prefix}_count": float(flat.size),
        f"{prefix}_mean": float(flat.mean()),
        f"{prefix}_std": float(flat.std()),
        f"{prefix}_min": float(flat.min()),
        f"{prefix}_p25": float(q25),
        f"{prefix}_p50": float(q50),
        f"{prefix}_p75": float(q75),
        f"{prefix}_max": float(flat.max()),
        f"{prefix}_frac_zero": float(np.mean(flat == 0.0)),
    }


def _global_diff_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Global pairwise disagreement stats between two forecast matrices."""
    diff = a - b
    abs_diff = np.abs(diff)
    flat_a = a.ravel()
    flat_b = b.ravel()
    flat_a_c = flat_a - flat_a.mean()
    flat_b_c = flat_b - flat_b.mean()
    denom = np.sqrt((flat_a_c ** 2).sum() * (flat_b_c ** 2).sum())
    corr = float((flat_a_c * flat_b_c).sum() / denom) if denom > 0.0 else float("nan")
    return {
        "diff_mae": float(abs_diff.mean()),
        "diff_rmse": float(np.sqrt((diff ** 2).mean())),
        "diff_median_abs": float(np.median(abs_diff)),
        "diff_bias_signed": float(diff.mean()),
        "diff_pearson_corr": corr,
    }


def _load_train_sales(data_dir: Path) -> pd.Series:
    """Load per-series total training sales from m5_ds.nc, indexed by series_id."""
    ds_path = data_dir / "m5_ds.nc"
    if not ds_path.exists():
        raise FileNotFoundError(f"m5_ds.nc not found at {ds_path}")
    logger.info("Loading training sales from %s", ds_path)
    ds = xr.open_dataset(ds_path)
    train_mask = ds["split"].values == "train"
    totals = np.nansum(ds["sales"].values[:, train_mask], axis=1)
    return pd.Series(totals, index=ds["series_id"].values, name="total_train_sales")


def _strip_suffix(ids: np.ndarray) -> np.ndarray:
    """Strip the _validation / _evaluation suffix from submission ids."""
    return np.array(
        [sid.rsplit("_validation", 1)[0].rsplit("_evaluation", 1)[0] for sid in ids],
        dtype=object,
    )


def main() -> None:
    args = _parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_submission_path = args.run_dir / "submission.csv"
    if not run_submission_path.exists():
        raise SystemExit(f"submission.csv not found in {args.run_dir}")
    if not args.reference.exists():
        raise SystemExit(f"Reference submission not found at {args.reference}")

    run_df = _load_submission(run_submission_path, "run")
    ref_df = _load_submission(args.reference, "reference")

    ids, a, b = _align_submissions(run_df, ref_df)
    logger.info("Aligned %d rows × %d forecast steps", a.shape[0], a.shape[1])

    abs_diff = np.abs(a - b)
    mae_row = abs_diff.mean(axis=1)
    rmse_row = np.sqrt(((a - b) ** 2).mean(axis=1))
    corr_row = _row_pearson(a, b)

    train_sales = _load_train_sales(args.data_dir)
    base_ids = _strip_suffix(ids)
    missing = set(base_ids) - set(train_sales.index.values)
    if missing:
        raise RuntimeError(f"{len(missing)} submission series missing from m5_ds.nc (first: {next(iter(missing))})")
    total_sales_per_row = train_sales.reindex(base_ids).values

    # Global summary (one row)
    summary = {
        **_flat_dist_stats(a, "run"),
        **_flat_dist_stats(b, "ref"),
        **_global_diff_stats(a, b),
    }
    summary_path = args.run_dir / "submission_diff_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    logger.info("Wrote global diff summary to %s", summary_path)

    # Top-N worst series
    per_row = pd.DataFrame(
        {
            "id": ids,
            "mean_run": a.mean(axis=1),
            "mean_ref": b.mean(axis=1),
            "mae_ab": mae_row,
            "rmse_ab": rmse_row,
            "corr_ab": corr_row,
            "total_train_sales": total_sales_per_row,
        }
    )
    top_n = per_row.sort_values("mae_ab", ascending=False).head(args.top_n).reset_index(drop=True)
    top_path = args.run_dir / "submission_diff_top_n.csv"
    top_n.to_csv(top_path, index=False)
    logger.info("Wrote top-%d worst series to %s", args.top_n, top_path)

    # Volume buckets
    bucket_labels = pd.qcut(
        per_row["total_train_sales"],
        q=args.n_buckets,
        labels=[f"q{i + 1}" for i in range(args.n_buckets)],
        duplicates="drop",
    )
    per_row["bucket"] = bucket_labels
    bucket_summary = (
        per_row.groupby("bucket", observed=True)
        .agg(
            count=("id", "size"),
            min_total_sales=("total_train_sales", "min"),
            max_total_sales=("total_train_sales", "max"),
            mean_mae_ab=("mae_ab", "mean"),
            mean_rmse_ab=("rmse_ab", "mean"),
            median_corr_ab=("corr_ab", "median"),
        )
        .reset_index()
    )
    bucket_path = args.run_dir / "submission_diff_by_volume.csv"
    bucket_summary.to_csv(bucket_path, index=False)
    logger.info("Wrote volume-bucket summary to %s", bucket_path)

    # Console summary
    logger.info(
        "Global | MAE: %.4f | RMSE: %.4f | bias: %+.4f | Pearson: %.4f",
        summary["diff_mae"],
        summary["diff_rmse"],
        summary["diff_bias_signed"],
        summary["diff_pearson_corr"],
    )
    logger.info("Top-5 worst series by MAE:")
    for _, row in top_n.head(5).iterrows():
        logger.info(
            "  %s  mae=%.4f  rmse=%.4f  corr=%.3f  run_mean=%.3f  ref_mean=%.3f  train_sales=%.0f",
            row["id"],
            row["mae_ab"],
            row["rmse_ab"],
            row["corr_ab"],
            row["mean_run"],
            row["mean_ref"],
            row["total_train_sales"],
        )
    logger.info("Per-bucket mean MAE (low → high volume):")
    for _, row in bucket_summary.iterrows():
        logger.info(
            "  %s  n=%d  sales=[%.0f, %.0f]  mean_mae=%.4f  median_corr=%.3f",
            row["bucket"],
            row["count"],
            row["min_total_sales"],
            row["max_total_sales"],
            row["mean_mae_ab"],
            row["median_corr_ab"],
        )


if __name__ == "__main__":
    main()
