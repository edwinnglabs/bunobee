#!/usr/bin/env python3
"""Offline validation script for M5 SSP prototype.

Given a saved ``batch_params.npz`` from ``m5_run.py``, this script:

1. Reconstructs the filtered states via ``kalman_filter_1d_batch``.
2. Generates 56‑day forecasts for the fitted series (28 validation + 28 test).
3. Computes bottom‑level M5‑style WRMSSE scores for validation and test.
4. Writes 56‑step forecasts and metric summary CSVs into the run directory.

This does not implement the full 42,840‑series hierarchy used in the
official competition, but the metric closely mimics the competition
objective at the item×store level.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

from wunkui.models.ssp.univariate import kalman_filter_1d_batch


logger = logging.getLogger(__name__)

HORIZON_VAL = 28
HORIZON_TEST = 28
HORIZON_TOTAL = HORIZON_VAL + HORIZON_TEST
_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "resource" / "m5-forecasting-accuracy"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate M5 SSP forecasts from batch_params.npz")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        metavar="DIR",
        help="Directory containing m5_ds.nc",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="m5_prototype/output run directory containing batch_params.npz",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        metavar="N",
        help="Max series per Kalman batch; 0 = no chunking",
    )
    return p.parse_args()


def _dow_dummies(dow: np.ndarray) -> np.ndarray:
    """Day-of-week dummies with Monday dropped (replicates m5_run.py)."""
    dummies = pd.get_dummies(dow, dtype=np.float32).reindex(columns=range(7), fill_value=0.0)
    return dummies.iloc[:, 1:].values  # drop Monday → (n, 6)


def _reconstruct_forecasts(
    ds: xr.Dataset,
    batch_params_path: Path,
    chunk_size: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild filtered states from saved params and generate 56-step forecasts.

    Returns
    -------
    forecasts : np.ndarray
        Array of shape (B, HORIZON_TOTAL) with 28‑day validation + 28‑day test
        forecasts for each fitted series.
    fit_idx : np.ndarray
        Indices of the fitted series into ds["series_id"].
    """
    data = np.load(batch_params_path, allow_pickle=True)
    sigma_h = data["sigma_h"]  # (B,)
    sigma_q = data["sigma_q"]  # (B, n_states)
    response_norm = data["response_norm"]  # (B,)
    fit_idx = data["fit_idx"].astype(int)  # (B,)

    # Masks and matrices
    split = ds["split"].values
    train_mask = split == "train"
    val_mask = split == "validation"
    test_mask = split == "test"

    sales = ds["sales"].values
    sales_train = sales[fit_idx][:, train_mask]  # (B, n_train)

    if sales_train.shape[1] <= 1:
        raise RuntimeError("Training window too short to compute scale and states.")

    dow = ds["day_of_week"].values
    Z_shared = jnp.asarray(_dow_dummies(dow[train_mask]))          # (n_train, 6)
    Z_val = jnp.asarray(_dow_dummies(dow[val_mask]))               # (HORIZON_VAL, 6)
    Z_test = jnp.asarray(_dow_dummies(dow[test_mask]))             # (HORIZON_TEST, 6)

    if Z_val.shape[0] != HORIZON_VAL or Z_test.shape[0] != HORIZON_TEST:
        raise RuntimeError(
            f"Expected validation/test horizon {HORIZON_VAL}+{HORIZON_TEST}, "
            f"got {Z_val.shape[0]}+{Z_test.shape[0]}"
        )

    Z_future = jnp.concatenate([Z_val, Z_test], axis=0)            # (HORIZON_TOTAL, 6)

    # Rebuild y used during fitting
    sales_clipped = np.clip(sales_train, 1e-1, None).astype(np.float32)
    y = jnp.asarray(np.log(sales_clipped / response_norm[:, None]))

    n_states = Z_shared.shape[1]
    a0 = jnp.zeros(n_states)
    P0 = jnp.ones(n_states)

    # Run batched Kalman filter to recover filtered states and covariances at MAP params
    _, at, Pt, _, _, _ = kalman_filter_1d_batch(
        a0=a0,
        P0=P0,
        Z=Z_shared,
        sigma_h=jnp.asarray(sigma_h),
        sigma_q=jnp.asarray(sigma_q),
        y=y,
        logp=False,
        chunk_size=None if chunk_size is None or chunk_size <= 0 else chunk_size,
    )

    at_np = np.asarray(at)  # (B, n_train, n_states)
    Pt_np = np.asarray(Pt)  # (B, n_train, n_states) — diagonal variances
    a_last = at_np[:, -1, :]  # (B, n_states)
    P_last = Pt_np[:, -1, :]  # (B, n_states)

    Z_future_np = np.asarray(Z_future)  # (HORIZON_TOTAL, n_states)
    mu_future = a_last @ Z_future_np.T  # (B, HORIZON_TOTAL)

    # Predictive variance under random-walk states: P_{T+h} = P_T + h · σ_q²
    # Var(y_{T+h}) = Σ_i Z_{T+h,i}² · (P_T,i + h · σ_q,i²) + σ_h²
    # Lognormal back-transform needs +0.5·Var to be unbiased (Jensen correction).
    h_steps = np.arange(1, Z_future_np.shape[0] + 1, dtype=np.float64)  # (HORIZON_TOTAL,)
    sigma_q_sq = np.asarray(sigma_q) ** 2  # (B, n_states)
    Z_sq = Z_future_np ** 2  # (HORIZON_TOTAL, n_states)
    # (B, HORIZON_TOTAL): contribution from P_T (constant over h) + from accumulated process noise
    var_state = P_last @ Z_sq.T + sigma_q_sq @ Z_sq.T * h_steps[None, :]
    var_future = var_state + (np.asarray(sigma_h) ** 2)[:, None]  # (B, HORIZON_TOTAL)

    forecasts = np.exp(mu_future + 0.5 * var_future) * response_norm[:, None]
    return forecasts, fit_idx


def _compute_bottom_wrmsse(
    ds: xr.Dataset,
    forecasts: np.ndarray,
    fit_idx: np.ndarray,
) -> dict[str, float]:
    """Compute bottom-level WRMSSE-style scores on validation and test windows.

    This mirrors the M5 definition at the item×store level:

    - RMSSE denominator uses training-period differences.
    - Weights are proportional to sales × price over the last 28 train days.
    """
    split = ds["split"].values
    train_mask = split == "train"
    val_mask = split == "validation"
    test_mask = split == "test"

    sales = ds["sales"].values[fit_idx]
    prices = ds["sell_price"].values[fit_idx]

    y_train = sales[:, train_mask]
    y_val = sales[:, val_mask]
    y_test = sales[:, test_mask]

    if y_val.shape[1] != HORIZON_VAL or y_test.shape[1] != HORIZON_TEST:
        raise RuntimeError(
            f"Expected validation/test length {HORIZON_VAL}+{HORIZON_TEST}, "
            f"got {y_val.shape[1]}+{y_test.shape[1]}"
        )

    if forecasts.shape[1] != HORIZON_TOTAL:
        raise RuntimeError(
            f"Forecast horizon {forecasts.shape[1]} does not match expected {HORIZON_TOTAL}"
        )

    f_val = forecasts[:, :HORIZON_VAL]
    f_test = forecasts[:, HORIZON_VAL:]

    # RMSSE per series (shared denominator for both windows)
    diff = np.diff(y_train, axis=1)
    scale = np.mean(diff ** 2, axis=1)
    scale = np.where(scale <= 0.0, 1.0, scale)
    denom = np.sqrt(scale)  # (B,)

    mse_val = np.mean((y_val - f_val) ** 2, axis=1)
    rmsse_val = np.sqrt(mse_val) / denom

    # Test labels are NaN when only sales_train_validation.csv is available
    # (M5 evaluation ground truth was never publicly released).
    test_labels_available = not np.all(np.isnan(y_test))
    if test_labels_available:
        mse_test = np.mean((y_test - f_test) ** 2, axis=1)
        rmsse_test = np.sqrt(mse_test) / denom
    else:
        logger.warning("Test-period sales are all NaN — test scores will be reported as NaN")
        rmsse_test = np.full_like(rmsse_val, np.nan)

    # Value weights from last 28 train days (approx competition weighting)
    train_indices = np.where(train_mask)[0]
    last_28_idx = train_indices[-HORIZON_VAL:]
    value = (sales[:, last_28_idx] * prices[:, last_28_idx]).sum(axis=1)
    value = np.where(value <= 0.0, 1.0, value)
    weights = value / value.sum()

    wrmsse_val = float(np.sum(weights * rmsse_val))
    wrmsse_test = float(np.sum(weights * rmsse_test))

    return {
        "wrmsse_val_bottom": wrmsse_val,
        "wrmsse_test_bottom": wrmsse_test,
        "mean_rmsse_val_bottom": float(rmsse_val.mean()),
        "mean_rmsse_test_bottom": float(rmsse_test.mean()),
    }


def main() -> None:
    args = _parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ds_path = args.data_dir / "m5_ds.nc"
    if not ds_path.exists():
        raise SystemExit(f"m5_ds.nc not found at {ds_path}. Run m5_preprocess.py first.")

    batch_params_path = args.run_dir / "batch_params.npz"
    if not batch_params_path.exists():
        raise SystemExit(f"batch_params.npz not found in {args.run_dir}")

    logger.info("Loading dataset from %s", ds_path)
    ds = xr.open_dataset(ds_path)

    logger.info("Reconstructing 56-step forecasts from %s", batch_params_path)
    forecasts, fit_idx = _reconstruct_forecasts(ds, batch_params_path, args.chunk_size)

    logger.info("Computing bottom-level WRMSSE over validation and test windows")
    metrics = _compute_bottom_wrmsse(ds, forecasts, fit_idx)

    # Save validation and test forecasts as separate files with F1-F28 columns each
    base_ids = ds["series_id"].values[fit_idx]
    f28_cols = [f"F{i}" for i in range(1, HORIZON_VAL + 1)]

    df_val = pd.DataFrame(forecasts[:, :HORIZON_VAL], columns=f28_cols)
    df_val.insert(0, "id", [f"{sid}_validation" for sid in base_ids])
    val_forecasts_path = args.run_dir / "offline_forecasts_val.csv"
    df_val.to_csv(val_forecasts_path, index=False)
    logger.info("Wrote validation forecasts to %s", val_forecasts_path)

    df_test = pd.DataFrame(forecasts[:, HORIZON_VAL:], columns=f28_cols)
    df_test.insert(0, "id", [f"{sid}_evaluation" for sid in base_ids])
    test_forecasts_path = args.run_dir / "offline_forecasts_test.csv"
    df_test.to_csv(test_forecasts_path, index=False)
    logger.info("Wrote test forecasts to %s", test_forecasts_path)

    # Save metric summary with one row per split for readability
    metrics_rows = [
        {
            "split": "validation",
            "steps": f"1-{HORIZON_VAL}",
            "wrmsse_bottom": metrics["wrmsse_val_bottom"],
            "mean_rmsse_bottom": metrics["mean_rmsse_val_bottom"],
        },
        {
            "split": "test",
            "steps": f"{HORIZON_VAL + 1}-{HORIZON_TOTAL}",
            "wrmsse_bottom": metrics["wrmsse_test_bottom"],
            "mean_rmsse_bottom": metrics["mean_rmsse_test_bottom"],
        },
    ]
    metrics_path = args.run_dir / "offline_wrmsse_summary.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    logger.info("Wrote WRMSSE summary to %s", metrics_path)

    logger.info(
        "Bottom-level WRMSSE  | validation (steps 1-%-2d): %.6f | test (steps %d-%d): %.6f",
        HORIZON_VAL,
        metrics["wrmsse_val_bottom"],
        HORIZON_VAL + 1,
        HORIZON_TOTAL,
        metrics["wrmsse_test_bottom"],
    )
    logger.info(
        "Bottom-level mean RMSSE | validation: %.6f | test: %.6f",
        metrics["mean_rmsse_val_bottom"],
        metrics["mean_rmsse_test_bottom"],
    )


if __name__ == "__main__":
    main()
