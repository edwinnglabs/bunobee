#!/usr/bin/env python3
"""Offline validation script for M5 SSP prototype.

Given a saved ``batch_params.npz`` from ``m5_run.py``, this script:

1. Reconstructs the filtered states via ``kalman_filter_1d_batch``.
2. Generates 28-day validation forecasts for the fitted series.
3. Computes the full 12-level hierarchical WRMSSE via
   ``m5_score_submission.compute_hierarchical_wrmsse``.
4. Writes validation forecasts and per-level WRMSSE CSVs into the run directory.

Test-window (d_1942-d_1969) scoring is intentionally omitted: the M5 evaluation
ground truth was never publicly released, so there are no actuals to score
against.
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

from m5_score_submission import compute_hierarchical_wrmsse


logger = logging.getLogger(__name__)

HORIZON_VAL = 28
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
    return dummies.iloc[:, 1:].values


def _reconstruct_val_forecasts(
    ds: xr.Dataset,
    batch_params_path: Path,
    chunk_size: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild filtered states from saved params and generate 28-step validation forecasts.

    Returns
    -------
    forecasts : np.ndarray
        Array of shape (B, HORIZON_VAL) with 28-day validation forecasts.
    fit_idx : np.ndarray
        Indices of the fitted series into ds["series_id"].
    """
    data = np.load(batch_params_path, allow_pickle=True)
    sigma_h = data["sigma_h"]
    sigma_q = data["sigma_q"]
    response_norm = data["response_norm"]
    fit_idx = data["fit_idx"].astype(int)

    split = ds["split"].values
    train_mask = split == "train"
    val_mask = split == "validation"

    sales = ds["sales"].values
    sales_train = sales[fit_idx][:, train_mask]

    if sales_train.shape[1] <= 1:
        raise RuntimeError("Training window too short to compute scale and states.")

    dow = ds["day_of_week"].values
    # Level column prepended to match the Z layout used in m5_run.py.
    dow_train = _dow_dummies(dow[train_mask])
    dow_val = _dow_dummies(dow[val_mask])
    ones_train = np.ones((dow_train.shape[0], 1), dtype=np.float32)
    ones_val = np.ones((dow_val.shape[0], 1), dtype=np.float32)
    Z_shared = jnp.asarray(np.concatenate([ones_train, dow_train], axis=1))
    Z_val = jnp.asarray(np.concatenate([ones_val, dow_val], axis=1))

    if Z_val.shape[0] != HORIZON_VAL:
        raise RuntimeError(f"Expected validation horizon {HORIZON_VAL}, got {Z_val.shape[0]}")

    sales_clipped = np.clip(sales_train, 1e-1, None).astype(np.float32)
    y = jnp.asarray(np.log(sales_clipped / response_norm[:, None]))

    n_states = Z_shared.shape[1]
    a0 = jnp.zeros(n_states)
    P0 = jnp.ones(n_states)

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

    at_np = np.asarray(at)
    Pt_np = np.asarray(Pt)
    a_last = at_np[:, -1, :]
    P_last = Pt_np[:, -1, :]

    Z_val_np = np.asarray(Z_val)
    mu_future = a_last @ Z_val_np.T

    # Predictive variance under random-walk states: P_{T+h} = P_T + h · σ_q²
    # Lognormal back-transform needs +0.5·Var for unbiased mean (Jensen correction).
    h_steps = np.arange(1, Z_val_np.shape[0] + 1, dtype=np.float64)
    sigma_q_sq = np.asarray(sigma_q) ** 2
    Z_sq = Z_val_np ** 2
    var_state = P_last @ Z_sq.T + sigma_q_sq @ Z_sq.T * h_steps[None, :]
    var_future = var_state + (np.asarray(sigma_h) ** 2)[:, None]

    forecasts = np.exp(mu_future + 0.5 * var_future) * response_norm[:, None]
    return forecasts, fit_idx


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

    logger.info("Reconstructing 28-step validation forecasts from %s", batch_params_path)
    forecasts, fit_idx = _reconstruct_val_forecasts(ds, batch_params_path, args.chunk_size)

    base_ids = ds["series_id"].values[fit_idx]
    f28_cols = [f"F{i}" for i in range(1, HORIZON_VAL + 1)]

    df_val = pd.DataFrame(forecasts, columns=f28_cols)
    df_val.insert(0, "id", [f"{sid}_validation" for sid in base_ids])
    val_forecasts_path = args.run_dir / "offline_forecasts_val.csv"
    df_val.to_csv(val_forecasts_path, index=False)
    logger.info("Wrote validation forecasts to %s", val_forecasts_path)

    logger.info("Computing 12-level hierarchical WRMSSE on validation window")
    per_level = compute_hierarchical_wrmsse(ds, forecasts.astype(np.float64), base_ids)
    final_wrmsse = float(per_level["wrmsse"].mean())

    per_level_path = args.run_dir / "offline_wrmsse_per_level.csv"
    per_level.to_csv(per_level_path, index=False)
    logger.info("Wrote per-level WRMSSE to %s", per_level_path)

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
