#!/usr/bin/env python3
"""M5 SSP forecasting script — batch MAP estimation.

Usage examples
--------------
# Fit 10 random series (demo mode):
    python m5_run.py

# Fit 50 top-volume series:
    python m5_run.py --n-demo 50 --sample-mode top

# Fit all 30,490 series and generate a submission CSV:
    python m5_run.py --n-demo 0
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from wunkui.models.m5.m5_ssp_optim import fit_batch_series_opt, predict_batch_series_opt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HORIZON = 28
_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "resource" / "m5-forecasting-accuracy"
_DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit M5 SSP model and generate forecasts.")
    p.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR, metavar="DIR",
                   help="Directory containing m5_ds.nc and sample_submission.csv.")
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR, metavar="DIR",
                   help="Root output directory; a timestamped sub-dir is created inside.")
    p.add_argument("--n-demo", type=int, default=10, metavar="N",
                   help="Number of series to fit. 0 = all 30,490 series.")
    p.add_argument("--sample-mode", choices=["random", "top"], default="random",
                   help="How to select demo series: random sample or top by total volume.")
    p.add_argument("--seed", type=int, default=2026,
                   help="RNG seed for reproducible random sampling.")
    p.add_argument("--n-iter", type=int, default=1500,
                   help="Adam optimisation steps per series.")
    p.add_argument("--lr", type=float, default=3e-2,
                   help="Adam learning rate.")
    p.add_argument("--chunk-size", type=int, default=4096, metavar="N",
                   help="Max series per vmap chunk (memory guard). 0 = no chunking.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dow_dummies(dow: np.ndarray) -> np.ndarray:
    """Day-of-week dummies with Monday dropped as the reference category."""
    dummies = pd.get_dummies(dow, dtype=np.float32).reindex(columns=range(7), fill_value=0.0)
    # Drop Monday (col 0) → shape (n, 6)
    return dummies.iloc[:, 1:].values


def _make_submission(
    predictions: pd.DataFrame,
    sample_submission: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Build a competition-ready submission CSV.

    Parameters
    ----------
    predictions : pd.DataFrame
        Columns: ``id`` (validation ids) + F1–F28.
    sample_submission : pd.DataFrame
        Official sample_submission template for row ordering.
    output_path : Path
        Where to write the CSV.

    Returns
    -------
    pd.DataFrame
        The final submission dataframe (60,980 rows).
    """
    f_cols = [f"F{i}" for i in range(1, HORIZON + 1)]

    eval_preds = predictions.copy()
    eval_preds["id"] = eval_preds["id"].str.replace("_validation", "_evaluation", regex=False)

    full = pd.concat([predictions, eval_preds], ignore_index=True)

    sub = sample_submission[["id"]].merge(full, on="id", how="left")

    assert sub.shape[0] == sample_submission.shape[0], (
        f"Row count mismatch: got {sub.shape[0]}, expected {sample_submission.shape[0]}"
    )
    assert not sub[f_cols].isna().any().any(), "Missing forecasts detected — ensure all series are fitted."

    sub.to_csv(output_path, index=False)
    logger.info("Submission written to %s  (%d rows)", output_path, sub.shape[0])
    return sub


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir: Path = args.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading dataset from %s", args.data_dir)
    ds = xr.open_dataset(args.data_dir / "m5_ds.nc")

    train_mask = ds["split"].values == "train"
    val_mask = ds["split"].values == "validation"
    sales_matrix = ds["sales"].values[:, train_mask]          # (n_series, n_train_steps)
    series_ids = np.array([f"{sid}_validation" for sid in ds["series_id"].values])
    n_series_total = sales_matrix.shape[0]
    logger.info("sales_matrix: %s  (n_series=%d, n_steps=%d)",
                sales_matrix.shape, sales_matrix.shape[0], sales_matrix.shape[1])

    # ------------------------------------------------------------------
    # 2. Build design matrices
    # ------------------------------------------------------------------
    dow = ds["day_of_week"].values
    Z_shared = jnp.asarray(_dow_dummies(dow[train_mask]))          # (n_train_steps, 6)
    Z_future_shared = jnp.asarray(_dow_dummies(dow[val_mask]))     # (horizon, 6)
    logger.info("Z_shared: %s  Z_future_shared: %s", Z_shared.shape, Z_future_shared.shape)

    # ------------------------------------------------------------------
    # 3. Select series to fit
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    total_sales = sales_matrix.sum(axis=1)
    n_fit = n_series_total if args.n_demo == 0 else args.n_demo

    if args.sample_mode == "random":
        fit_idx = rng.choice(n_series_total, size=n_fit, replace=False)
        label = f"random sample (seed={args.seed})"
    else:
        fit_idx = np.argsort(total_sales)[::-1][:n_fit]
        label = "top by volume"

    logger.info("Selected %d series [%s]", n_fit, label)
    for rank, idx in enumerate(fit_idx[:10]):
        logger.info("  #%d  %s  total=%s", rank + 1, series_ids[idx], f"{total_sales[idx]:,.0f}")
    if n_fit > 10:
        logger.info("  … (%d more)", n_fit - 10)

    # ------------------------------------------------------------------
    # 4. Fit + predict
    # ------------------------------------------------------------------
    chunk_size = args.chunk_size if args.chunk_size > 0 else None
    logger.info("Starting batch fit: n_iter=%d  lr=%g  chunk_size=%s", args.n_iter, args.lr, chunk_size)
    t0 = time.time()
    batch_result = fit_batch_series_opt(
        sales_matrix[fit_idx],
        n_iter=args.n_iter,
        lr=args.lr,
        Z=Z_shared,
        chunk_size=chunk_size,
        log_every=100,
        show_progress=True,
    )
    forecasts = predict_batch_series_opt(batch_result, Z_future=Z_future_shared)
    elapsed = time.time() - t0
    logger.info("Fitted %d series in %.1fs  (%.1f ms/series)", n_fit, elapsed, elapsed / n_fit * 1000)

    # ------------------------------------------------------------------
    # 5. Diagnostics plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    axes[0].hist(batch_result["final_loss"], bins=100, edgecolor="none")
    axes[0].set_xlabel("Final neg-log-posterior")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Loss distribution")

    axes[1].hist(batch_result["sigma_h"], bins=100, edgecolor="none", alpha=0.7, label="σ_h")
    axes[1].hist(batch_result["sigma_q"][:, 0], bins=100, edgecolor="none", alpha=0.7, label="σ_q level")
    axes[1].hist(batch_result["sigma_q"][:, 1], bins=100, edgecolor="none", alpha=0.7, label="σ_q seas")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Count")
    axes[1].set_title("MAP parameter distributions")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(run_dir / "diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved diagnostics.png")

    # ------------------------------------------------------------------
    # 6. Forecast plot (first n_plot series)
    # ------------------------------------------------------------------
    n_plot = min(n_fit, 10)
    tail = 90
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 3 * n_plot), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for rank, (ax, idx) in enumerate(zip(axes, fit_idx[:n_plot])):
        actual = sales_matrix[idx]
        forecast = forecasts[rank]
        ax.plot(range(tail), actual[-tail:], label="Actual", alpha=0.7)
        ax.plot(range(tail, tail + HORIZON), forecast, label="Forecast", linestyle="--", color="tomato")
        ax.axvline(tail, color="grey", linestyle=":", alpha=0.5)
        ax.set_title(series_ids[idx], fontsize=9, loc="left")
        ax.set_ylabel("Units")
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Day")
    fig.suptitle(f"{n_plot} series ({label}) — SSP batch forecast", y=1.01)
    plt.tight_layout()
    fig.savefig(run_dir / "forecasts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved forecasts.png")

    # ------------------------------------------------------------------
    # 7. Save fitted parameters
    # ------------------------------------------------------------------
    np.savez(
        run_dir / "batch_params.npz",
        sigma_h=batch_result["sigma_h"],
        sigma_q=batch_result["sigma_q"],
        response_norm=batch_result["response_norm"],
        final_loss=batch_result["final_loss"],
        fit_idx=fit_idx,
    )
    logger.info("Saved batch_params.npz")

    # ------------------------------------------------------------------
    # 8. Submission CSV (only when all series are fitted)
    # ------------------------------------------------------------------
    if n_fit == n_series_total:
        f_cols = [f"F{i}" for i in range(1, HORIZON + 1)]
        predictions = pd.DataFrame(forecasts, columns=f_cols)
        predictions.insert(0, "id", series_ids[fit_idx])
        sample_sub = pd.read_csv(args.data_dir / "sample_submission.csv")
        _make_submission(predictions, sample_sub, output_path=run_dir / "submission.csv")
    else:
        logger.info("Skipping submission — only %d / %d series fitted (use --n-demo 0 to fit all).",
                    n_fit, n_series_total)

    logger.info("Done. Outputs in %s", run_dir)


if __name__ == "__main__":
    main()
