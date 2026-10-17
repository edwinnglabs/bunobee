#!/usr/bin/env python3
"""M5 forecasting script — fit a registered model, forecast, score and write the submission.

Usage examples
--------------
# Fit 10 random series (demo mode) with the baseline model, ssp_dow:
    python m5_run.py

# Fit 50 top-volume series:
    python m5_run.py --n-demo 50 --sample-mode top

# Fit all 30,490 series with a named model and generate a submission CSV:
    python m5_run.py --model ssp_dow --n-demo 0

``--model`` picks an entry of ``MODELS`` in ``m5_models``: a model plus its settings (for ``ssp_dow``, the
optimiser steps, learning rate and fit chunking). The runner never builds model inputs; adding a model
needs only a new entry there.

Every run fits the selected series twice, once per forecast origin (see ``WINDOWS`` in
``m5_score_submission``):

    validation : fit on d_1-d_1913, forecast d_1914-d_1941  (``_validation`` submission rows)
    evaluation : fit on d_1-d_1941, forecast d_1942-d_1969  (``_evaluation`` submission rows)

It scores each origin's forecasts with the 12-level hierarchical WRMSSE and writes the per-level tables
to ``wrmsse_per_level.csv`` in the run directory, one ``window`` per origin. The evaluation window is
scored only when the dataset carries its actuals (``m5_fetch.py`` + ``m5_preprocess.py``). A sampled
run aggregates and weights only its own series, so its score is a smoke test that the pipeline runs
end to end, not a number comparable to the full 30,490-series score.

Every run also writes the model name and settings, its wall-clock time and peak memory to ``run_stats.json``
in the run directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import resource
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from m5_common import DATA_DIR, load_m5_ds
from m5_models import DEFAULT_MODEL, MODELS
from m5_score_submission import WINDOWS, compute_hierarchical_wrmsse, log_per_level

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HORIZON = 28
_DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit a registered M5 model and generate forecasts.")
    p.add_argument(
        "--model",
        choices=tuple(MODELS),
        default=DEFAULT_MODEL,
        help=f"Registered model to fit (see MODELS in m5_models.py), by default {DEFAULT_MODEL}.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        metavar="DIR",
        help="Directory containing m5_ds.nc and sample_submission.csv.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help="Root output directory; a timestamped sub-dir is created inside.",
    )
    p.add_argument(
        "--n-demo", type=int, default=10, metavar="N", help="Number of series to fit. 0 = all 30,490 series."
    )
    p.add_argument(
        "--sample-mode",
        choices=["random", "top"],
        default="random",
        help="How to select demo series: random sample or top by total volume.",
    )
    p.add_argument("--seed", type=int, default=2026, help="RNG seed for reproducible random sampling.")
    p.add_argument(
        "--n-plot",
        type=int,
        default=50,
        metavar="N",
        help="Number of series to include in forecasts.pdf. 0 = all fitted series.",
    )
    p.add_argument(
        "--plots-per-page", type=int, default=5, metavar="N", help="Forecast subplots per PDF page, by default 5."
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_model_output(model_name: str, forecasts: np.ndarray, fitted: dict[str, np.ndarray], n_series: int) -> None:
    """Raise if a model's ``fit_forecast`` output breaks the contract of ``m5_models.Model``.

    Parameters
    ----------
    model_name : str
        Registered name of the model, for the error message.
    forecasts : np.ndarray
        The model's forecasts; must have shape ``(n_series, HORIZON)`` and no NaN or infinite values.
    fitted : dict of str to np.ndarray
        The model's per-series outputs; each must have leading dimension ``n_series``.
    n_series : int
        Number of series the model was asked to fit.

    Raises
    ------
    ValueError
        If any of the above does not hold.
    """
    if np.shape(forecasts) != (n_series, HORIZON):
        raise ValueError(f"{model_name}: forecasts have shape {np.shape(forecasts)}, expected {(n_series, HORIZON)}")
    n_bad = int((~np.isfinite(forecasts)).any(axis=1).sum())
    if n_bad:
        raise ValueError(f"{model_name}: forecasts of {n_bad} / {n_series} series have NaN or infinite values")
    bad_keys = [key for key, value in fitted.items() if np.shape(value)[:1] != (n_series,)]
    if bad_keys:
        raise ValueError(f"{model_name}: fitted arrays {bad_keys} do not have leading dimension {n_series}")


def _plot_fitted(fitted: dict[str, np.ndarray], title: str, output_path: Path) -> None:
    """Save one histogram per fitted array, with one series of bars per column of a 2-D array.

    Parameters
    ----------
    fitted : dict of str to np.ndarray
        Per-series outputs of a model, each with leading dimension ``n_series``.
    title : str
        Figure title.
    output_path : Path
        Where to save the figure.
    """
    if not fitted:
        logger.info("Skipping diagnostics.png — the model returned no fitted arrays.")
        return
    n_cols = min(len(fitted), 3)
    n_rows = math.ceil(len(fitted) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    for ax, (key, value) in zip(axes.flat, fitted.items()):
        columns = np.asarray(value).reshape(len(value), -1)
        for col in range(columns.shape[1]):
            label = f"[:, {col}]" if columns.shape[1] > 1 else None
            ax.hist(columns[:, col], bins=100, edgecolor="none", alpha=0.7, label=label)
        if columns.shape[1] > 1:
            ax.legend(fontsize=8)
        ax.set_title(key)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    for ax in axes.flat[len(fitted) :]:
        ax.set_visible(False)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def _peak_rss_gib() -> float:
    """Peak resident set size of this process so far, in GiB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in bytes on macOS and in kilobytes on Linux.
    return peak / 2**30 if sys.platform == "darwin" else peak / 2**20


def _make_submission(
    forecasts: dict[str, np.ndarray],
    base_ids: np.ndarray,
    sample_submission: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Build a competition-ready submission CSV.

    Parameters
    ----------
    forecasts : dict of str to np.ndarray
        Forecasts of shape ``(n_series, HORIZON)`` per window (``"validation"``, ``"evaluation"``); the
        window name is the ``id`` suffix of its rows.
    base_ids : np.ndarray
        ``series_id`` of each forecast row, without a window suffix.
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

    parts = []
    for window, window_forecasts in forecasts.items():
        part = pd.DataFrame(window_forecasts, columns=f_cols)
        part.insert(0, "id", [f"{sid}_{window}" for sid in base_ids])
        parts.append(part)
    full = pd.concat(parts, ignore_index=True)

    sub = sample_submission[["id"]].merge(full, on="id", how="left")

    assert (
        sub.shape[0] == sample_submission.shape[0]
    ), f"Row count mismatch: got {sub.shape[0]}, expected {sample_submission.shape[0]}"
    assert not sub[f_cols].isna().any().any(), "Missing forecasts detected — ensure all series are fitted."

    sub.to_csv(output_path, index=False)
    logger.info("Submission written to %s  (%d rows)", output_path, sub.shape[0])
    return sub


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t_start = time.time()
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
    ds = load_m5_ds(args.data_dir)

    split = ds["split"].values
    train_mask = split == "train"
    sales_matrix = ds["sales"].values[:, train_mask]  # (n_series, n_train_steps)
    series_ids = np.array([f"{sid}_validation" for sid in ds["series_id"].values])
    n_series_total = sales_matrix.shape[0]
    logger.info(
        "sales_matrix: %s  (n_series=%d, n_steps=%d)", sales_matrix.shape, sales_matrix.shape[0], sales_matrix.shape[1]
    )

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
    # 4. Fit + predict, once per forecast origin
    # ------------------------------------------------------------------
    model = MODELS[args.model]
    settings = asdict(model)
    fitted_by_window: dict[str, dict[str, np.ndarray]] = {}
    forecasts_by_window: dict[str, np.ndarray] = {}
    fit_seconds_by_window: dict[str, float] = {}
    for window, (fit_splits, forecast_split, days) in WINDOWS.items():
        fit_mask = np.isin(split, fit_splits)
        logger.info(
            "Origin %s: fit on %d days, forecast %s  (model %s: %s)",
            window,
            fit_mask.sum(),
            days,
            args.model,
            "  ".join(f"{key}={value}" for key, value in settings.items()),
        )
        t0 = time.time()
        forecasts_by_window[window], fitted_by_window[window] = model.fit_forecast(
            ds, fit_idx, fit_mask, split == forecast_split
        )
        fit_seconds_by_window[window] = time.time() - t0
        _check_model_output(args.model, forecasts_by_window[window], fitted_by_window[window], n_fit)
        logger.info(
            "Fitted %d series in %.1fs  (%.1f ms/series, peak RSS so far %.1f GiB)",
            n_fit,
            fit_seconds_by_window[window],
            fit_seconds_by_window[window] / n_fit * 1000,
            _peak_rss_gib(),
        )
    fit_seconds = sum(fit_seconds_by_window.values())
    # The diagnostics and the forecast PDF show the validation origin, whose window has actuals in sales_matrix.
    fitted, forecasts = fitted_by_window["validation"], forecasts_by_window["validation"]

    # ------------------------------------------------------------------
    # 5. Diagnostics plot
    # ------------------------------------------------------------------
    _plot_fitted(
        fitted, title=f"{args.model} fitted outputs — validation origin", output_path=run_dir / "diagnostics.png"
    )

    # ------------------------------------------------------------------
    # 6. Forecast PDF (paginated, --plots-per-page series per page)
    # ------------------------------------------------------------------
    n_plot = n_fit if args.n_plot == 0 else min(args.n_plot, n_fit)
    per_page = args.plots_per_page
    n_pages = math.ceil(n_plot / per_page)
    tail = 90
    pdf_path = run_dir / "forecasts.pdf"

    with PdfPages(pdf_path) as pdf:
        for page in range(n_pages):
            start = page * per_page
            end = min(start + per_page, n_plot)
            page_ranks = range(start, end)
            n_rows = end - start

            fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=True, squeeze=False)
            axes = axes[:, 0]

            for ax, rank in zip(axes, page_ranks):
                idx = fit_idx[rank]
                actual = sales_matrix[idx]
                forecast = forecasts[rank]
                ax.plot(range(tail), actual[-tail:], label="Actual", alpha=0.7)
                ax.plot(range(tail, tail + HORIZON), forecast, label="Forecast", linestyle="--", color="tomato")
                ax.axvline(tail, color="grey", linestyle=":", alpha=0.5)
                ax.set_title(series_ids[idx], fontsize=9, loc="left")
                ax.set_ylabel("Units")
                ax.legend(fontsize=8)

            axes[-1].set_xlabel("Day")
            fig.suptitle(
                f"{args.model} forecast — {label}  (page {page + 1}/{n_pages})",
                y=1.01,
            )
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    logger.info("Saved forecasts.pdf  (%d series, %d pages)", n_plot, n_pages)

    # ------------------------------------------------------------------
    # 7. Save the model's fitted outputs
    # ------------------------------------------------------------------
    np.savez(
        run_dir / "batch_params.npz",
        **fitted,
        forecasts=forecasts,
        **{f"{key}_evaluation": value for key, value in fitted_by_window["evaluation"].items()},
        forecasts_evaluation=forecasts_by_window["evaluation"],
        fit_idx=fit_idx,
    )
    logger.info("Saved batch_params.npz")

    # ------------------------------------------------------------------
    # 8. Submission CSV (only when all series are fitted)
    # ------------------------------------------------------------------
    base_ids = ds["series_id"].values[fit_idx]
    if n_fit == n_series_total:
        sample_sub = pd.read_csv(args.data_dir / "sample_submission.csv")
        _make_submission(forecasts_by_window, base_ids, sample_sub, output_path=run_dir / "submission.csv")
    else:
        logger.info(
            "Skipping submission — only %d / %d series fitted (use --n-demo 0 to fit all).", n_fit, n_series_total
        )

    # ------------------------------------------------------------------
    # 9. WRMSSE of the fitted series, per window with actuals
    # ------------------------------------------------------------------
    tables = []
    for window, (_, score_split, days) in WINDOWS.items():
        if not np.isfinite(ds["sales"].values[:, split == score_split]).all():
            logger.warning(
                "Skipping the %s score: the dataset has no actuals for %s (run m5_fetch.py, then m5_preprocess.py).",
                window,
                days,
            )
            continue
        per_level = compute_hierarchical_wrmsse(
            ds, np.asarray(forecasts_by_window[window], dtype=np.float64), base_ids, window=window
        )
        log_per_level(per_level, window=window)
        tables.append(per_level.assign(window=window)[["window", *per_level.columns]])
    pd.concat(tables, ignore_index=True).to_csv(run_dir / "wrmsse_per_level.csv", index=False)
    logger.info(
        "Saved wrmsse_per_level.csv  (%d series, windows: %s)", n_fit, ", ".join(t["window"][0] for t in tables)
    )
    if n_fit < n_series_total:
        logger.warning(
            "Sampled WRMSSE: only %d / %d series are aggregated and weighted, so this score is a smoke test, "
            "not comparable to the full-run score.",
            n_fit,
            n_series_total,
        )

    # ------------------------------------------------------------------
    # 10. Run stats (model, wall-clock and peak memory)
    # ------------------------------------------------------------------
    run_stats = {
        "model": args.model,
        "settings": settings,
        "n_series": n_fit,
        "fit_seconds": round(fit_seconds, 1),
        "fit_seconds_by_origin": {window: round(sec, 1) for window, sec in fit_seconds_by_window.items()},
        "total_seconds": round(time.time() - t_start, 1),
        "peak_rss_gib": round(_peak_rss_gib(), 2),
    }
    (run_dir / "run_stats.json").write_text(json.dumps(run_stats, indent=2) + "\n")
    logger.info(
        "Saved run_stats.json  (fit %.1fs, total %.1fs, peak RSS %.2f GiB)",
        run_stats["fit_seconds"],
        run_stats["total_seconds"],
        run_stats["peak_rss_gib"],
    )

    logger.info("Done. Outputs in %s", run_dir)


if __name__ == "__main__":
    main()
