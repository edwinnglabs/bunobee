#!/usr/bin/env python3
"""Compare the scores of two ``m5_run.py`` runs fitted on the same series.

Usage
-----
    python m5_compare.py output/<baseline run> output/<new run>

For each window both runs scored (``wrmsse_per_level.csv``), this prints the per-level WRMSSE and bias
(``bias_pct``) of both runs and their differences, then the hierarchical WRMSSE (the mean of the 12 levels)
of both and its difference. A difference is ``RUN_B - RUN_A``, so a negative WRMSSE difference means
``RUN_B`` scores better.

A sampled run aggregates and weights only its own series, so its score is comparable only with a run on
the same series. The script refuses two runs fitted on different series (``fit_idx`` in
``batch_params.npz``; their order does not matter) and two runs with no scored window in common.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from m5_score_submission import WINDOWS

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare the per-level WRMSSE and bias of two m5_run.py runs.")
    p.add_argument("run_a", type=Path, metavar="RUN_A", help="Baseline run directory, e.g. output/<timestamp>.")
    p.add_argument("run_b", type=Path, metavar="RUN_B", help="Run directory to compare with RUN_A.")
    return p.parse_args()


def load_run(run_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Read which series a run fitted and its per-level scores.

    Parameters
    ----------
    run_dir : Path
        An ``m5_run.py`` run directory.

    Returns
    -------
    fit_idx : np.ndarray
        Positions along ``series_id`` of the series the run fitted.
    per_level : pd.DataFrame
        The run's ``wrmsse_per_level.csv``: one row per window and level.

    Raises
    ------
    FileNotFoundError
        If ``batch_params.npz`` or ``wrmsse_per_level.csv`` is not in ``run_dir``.
    """
    for name in ("batch_params.npz", "wrmsse_per_level.csv"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"{run_dir / name} not found; is {run_dir} an m5_run.py run directory?")
    with np.load(run_dir / "batch_params.npz") as params:
        fit_idx = params["fit_idx"]
    return fit_idx, pd.read_csv(run_dir / "wrmsse_per_level.csv")


def check_same_series(fit_idx_a: np.ndarray, fit_idx_b: np.ndarray) -> None:
    """Raise unless two runs fitted the same series, in any order.

    Parameters
    ----------
    fit_idx_a, fit_idx_b : np.ndarray
        ``fit_idx`` of each run.

    Raises
    ------
    ValueError
        If the two runs did not fit the same set of series.
    """
    a, b = np.unique(fit_idx_a), np.unique(fit_idx_b)
    if not np.array_equal(a, b):
        raise ValueError(
            f"The runs were fitted on different series ({a.size} and {b.size} series, "
            f"{np.intersect1d(a, b).size} in common), so their scores are not comparable. Fit both with the same "
            "--n-demo, --sample-mode and --seed."
        )


def shared_windows(per_level_a: pd.DataFrame, per_level_b: pd.DataFrame) -> list[str]:
    """Windows scored by both runs, in ``WINDOWS`` order.

    Parameters
    ----------
    per_level_a, per_level_b : pd.DataFrame
        ``wrmsse_per_level.csv`` of each run.

    Returns
    -------
    list of str
        The shared windows.

    Raises
    ------
    ValueError
        If the runs share no scored window.
    """
    windows_a, windows_b = set(per_level_a["window"]), set(per_level_b["window"])
    windows = [window for window in WINDOWS if window in windows_a and window in windows_b]
    if not windows:
        raise ValueError(
            f"The runs share no scored window (RUN_A: {', '.join(sorted(windows_a)) or 'none'}; "
            f"RUN_B: {', '.join(sorted(windows_b)) or 'none'})."
        )
    return windows


def compare_runs(run_a: Path, run_b: Path) -> pd.DataFrame:
    """Per-level WRMSSE and bias of two runs and their differences, for each window both runs scored.

    Parameters
    ----------
    run_a, run_b : Path
        ``m5_run.py`` run directories; differences are ``run_b - run_a``.

    Returns
    -------
    pd.DataFrame
        One row per shared window and level: ``window``, ``level``, ``name``, then ``wrmsse_a``,
        ``wrmsse_b``, ``wrmsse_delta``, ``bias_pct_a``, ``bias_pct_b`` and ``bias_pct_delta``.

    Raises
    ------
    FileNotFoundError
        If a run directory lacks ``batch_params.npz`` or ``wrmsse_per_level.csv``.
    ValueError
        If the runs were fitted on different series or share no scored window.
    """
    fit_idx_a, per_level_a = load_run(run_a)
    fit_idx_b, per_level_b = load_run(run_b)
    check_same_series(fit_idx_a, fit_idx_b)
    shared_windows(per_level_a, per_level_b)

    # An inner merge keeps only the windows both runs scored.
    keys = ["window", "level", "name"]
    table = per_level_a[[*keys, "wrmsse", "bias_pct"]].merge(
        per_level_b[[*keys, "wrmsse", "bias_pct"]], on=keys, suffixes=("_a", "_b")
    )
    columns = list(keys)
    for metric in ("wrmsse", "bias_pct"):
        table[f"{metric}_delta"] = table[f"{metric}_b"] - table[f"{metric}_a"]
        columns += [f"{metric}_a", f"{metric}_b", f"{metric}_delta"]
    return table[columns].reset_index(drop=True)


def _describe(run_dir: Path) -> str:
    """The run directory, with its model and number of series when ``run_stats.json`` records them."""
    stats_path = run_dir / "run_stats.json"
    if not stats_path.is_file():
        return str(run_dir)
    stats = json.loads(stats_path.read_text())
    return f"{run_dir}  (model {stats.get('model', '?')}, {stats.get('n_series', '?')} series)"


def log_comparison(table: pd.DataFrame, run_a: Path, run_b: Path) -> None:
    """Log the output of :func:`compare_runs`, one table per window.

    Parameters
    ----------
    table : pd.DataFrame
        Output of :func:`compare_runs`.
    run_a, run_b : Path
        The compared run directories.
    """
    logger.info("A: %s", _describe(run_a))
    logger.info("B: %s", _describe(run_b))
    for window, rows in table.groupby("window", sort=False):
        logger.info("Per-level WRMSSE and bias_%% (%s %s), delta = B - A:", window, WINDOWS[window][2])
        logger.info(
            "  %-3s  %-22s  %-8s  %-8s  %-9s  %-8s  %-8s  %-8s",
            "lvl",
            "name",
            "wrmsse_A",
            "wrmsse_B",
            "delta",
            "bias_A",
            "bias_B",
            "delta",
        )
        for _, row in rows.iterrows():
            logger.info(
                "  L%-2d  %-22s  %.4f    %.4f    %+.4f    %+7.1f%%  %+7.1f%%  %+7.1f%%",
                int(row["level"]),
                row["name"],
                row["wrmsse_a"],
                row["wrmsse_b"],
                row["wrmsse_delta"],
                row["bias_pct_a"],
                row["bias_pct_b"],
                row["bias_pct_delta"],
            )
        mean_a, mean_b = rows["wrmsse_a"].mean(), rows["wrmsse_b"].mean()
        logger.info(
            "Hierarchical WRMSSE (mean of 12 levels): A %.6f  B %.6f  delta %+.6f", mean_a, mean_b, mean_b - mean_a
        )


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        table = compare_runs(args.run_a, args.run_b)
    except (FileNotFoundError, ValueError) as err:
        raise SystemExit(f"m5_compare.py: {err}") from None
    log_comparison(table, args.run_a, args.run_b)


if __name__ == "__main__":
    main()
