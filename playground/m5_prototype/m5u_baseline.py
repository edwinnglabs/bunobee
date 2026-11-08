#!/usr/bin/env python3
"""A cheap M5 Uncertainty baseline: empirical quantiles of recent sales at every level (issue #157).

For each of the 42,840 series of the 12 levels, the forecast at every one of the 28 days is the empirical quantiles
of its own last :data:`HISTORY_DAYS` training days. It fits nothing, so it runs on the full hierarchy in seconds: its
point is to test the writer and the scorer at full size, and to be the bar a model has to beat.

    python m5u_baseline.py                 # full hierarchy, both windows: writes a submission, scores it locally
    python m5u_baseline.py --subset --windows prevalidation   # the 3-item benchmark subset: scores only

Both windows are forecast from the days before them (``validation`` from d_1913 back, ``evaluation`` from d_1941),
and each is scored when the dataset has its actuals. Nothing is submitted to Kaggle.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from m5_common import DATA_DIR, UNCERTAINTY_DIR, load_m5_ds
from m5_score_submission import N_STEPS, SUBMISSION_WINDOWS, WINDOWS, window_masks
from m5u_score import (
    QUANTILES,
    Hierarchy,
    aggregate,
    build_hierarchy,
    compute_hierarchical_wspl,
    log_per_level,
)
from m5u_submission import write_submission

logger = logging.getLogger(__name__)

#: Recent training days whose sales the quantiles are taken over: 13 weeks, a whole number of weeks.
HISTORY_DAYS = 91


def baseline_forecasts(
    ds: xr.Dataset,
    hierarchy: Hierarchy,
    window: str,
    history_days: int = HISTORY_DAYS,
    quantiles: tuple[float, ...] = QUANTILES,
) -> list[np.ndarray]:
    """Empirical quantiles of each group's last ``history_days`` training days, repeated over the 28 days.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from ``m5_preprocess.build_m5_dataset``.
    hierarchy : Hierarchy
        Groups of the bottom series, from ``m5u_score.build_hierarchy``.
    window : {"prevalidation", "validation", "evaluation"}
        Window to forecast, a key of ``WINDOWS``; the quantiles come from the days before it.
    history_days : int, optional
        Number of recent training days. Default :data:`HISTORY_DAYS`.
    quantiles : tuple of float, optional
        Quantile levels. Default :data:`QUANTILES`.

    Returns
    -------
    list of np.ndarray
        Per level, shape ``(n_groups, len(quantiles), N_STEPS)`` in ``hierarchy.ids`` order.
    """
    train_mask, _ = window_masks(ds["split"].values, window)
    recent = np.flatnonzero(train_mask)[-history_days:]
    sales = ds["sales"].values[hierarchy.rows][:, recent]
    forecasts = []
    for group_idx, n_groups in zip(hierarchy.group_idx, hierarchy.n_groups, strict=True):
        q = np.quantile(aggregate(sales, group_idx, n_groups), quantiles, axis=1)  # (n_quantiles, n_groups)
        forecasts.append(np.repeat(q.T[:, :, None], N_STEPS, axis=2))
    return forecasts


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=Path, default=DATA_DIR, metavar="DIR", help="Directory containing m5_ds.nc.")
    p.add_argument(
        "--sample",
        type=Path,
        default=UNCERTAINTY_DIR / "sample_submission.csv",
        metavar="CSV",
        help="The uncertainty competition's sample_submission.csv.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=UNCERTAINTY_DIR / "baseline_submission.csv",
        metavar="CSV",
        help="Where to write the submission (not written with --subset).",
    )
    p.add_argument("--history-days", type=int, default=HISTORY_DAYS, help="Recent training days to take quantiles of.")
    p.add_argument("--subset", action="store_true", help="Score the 3-item benchmark subset only; write nothing.")
    p.add_argument(
        "--windows",
        nargs="+",
        choices=sorted(WINDOWS),
        default=list(SUBMISSION_WINDOWS),
        help="Windows to score, by default the two a submission has rows for (the pre-validation one is the "
        "benchmark subset's). A full run always forecasts and writes both submission windows.",
    )
    return p.parse_args()


def main() -> None:
    """Forecast both windows, write the submission (full hierarchy) and log the local WSPL of each window."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ds = load_m5_ds(args.data_dir)
    if args.subset:
        from m5_subset import subset_series_idx  # here: it is only the subset run's dependency

        hierarchy = build_hierarchy(ds, subset_series_idx(ds))
    else:
        hierarchy = build_hierarchy(ds)
    logger.info("%d bottom series, %d series over 12 levels", len(hierarchy.rows), sum(hierarchy.n_groups))

    windows = list(args.windows) if args.subset else list(dict.fromkeys([*SUBMISSION_WINDOWS, *args.windows]))
    forecasts = {w: baseline_forecasts(ds, hierarchy, w, args.history_days) for w in windows}
    if not args.subset:
        write_submission(forecasts, hierarchy, args.sample, args.output)
        logger.info("Wrote %s (matches the ids and row order of %s); nothing was submitted", args.output, args.sample)

    scores: dict[str, pd.DataFrame] = {}
    for window in args.windows:
        try:
            scores[window] = compute_hierarchical_wspl(ds, forecasts[window], hierarchy, window=window)
        except RuntimeError as e:  # the evaluation actuals are missing from a dataset built from the Kaggle files
            logger.warning("Not scoring the %s window: %s", window, e)
            continue
        log_per_level(scores[window], window=window)
    for window, per_level in scores.items():
        print(f"{window} WSPL: {per_level['wspl'].mean():.5f}")


if __name__ == "__main__":
    main()
