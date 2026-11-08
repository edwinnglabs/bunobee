#!/usr/bin/env python3
"""Local scorer for the M5 Uncertainty competition: the weighted scaled pinball loss (WSPL), issue #157.

The competition scores a forecast distribution of the same sales data as M5 Accuracy, as 9 quantiles of every one of
the 42,840 series of the 12 aggregation levels (not only the 30,490 bottom series). For a series ``i`` with forecast
quantile ``Q_t(u)`` and actual ``Y_t`` over the ``h = 28`` scored days, the scaled pinball loss is

    SPL_i(u) = 1/h * sum_t pinball_u(Y_t, Q_t(u)) / S_i,    S_i = mean |Y_t - Y_{t-1}| over the training days,

with ``pinball_u(y, q) = u * (y - q)`` if ``q <= y`` and ``(1 - u) * (q - y)`` otherwise. The series' WSPL contribution
averages ``SPL_i(u)`` over the 9 quantiles :data:`QUANTILES`; a level's WSPL is the weighted sum of its series'
contributions, and the overall score is the mean of the 12 levels' WSPL.

The scorer shares the accuracy scorer's details (``m5_score_submission``), so a local score is comparable with a
published one: the weights are the dollar sales over the last 28 training days, normalised within the level, and the
scale ``S_i`` is the accuracy metric's with absolute one-step differences instead of squared ones, counted from the
series' first non-zero sale on (:func:`spl_scale`).

Unlike the accuracy metric, the quantiles of a level cannot be summed from the bottom series', so a forecast is one
array of quantiles per level. :func:`build_hierarchy` fixes the groups of each level and their order; it takes the
full dataset or any set of bottom series, such as the 3-item subset of ``m5_subset``, and then scores that set's own
12 levels.

Usage
-----
    python m5u_score.py --submission baseline.csv --window validation
"""

from __future__ import annotations

import argparse
import logging
from functools import reduce
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse

from m5_common import DATA_DIR, load_m5_ds
from m5_score_submission import N_STEPS, SUBMISSION_WINDOWS, WINDOWS, dollar_sales, window_masks

logger = logging.getLogger(__name__)

#: The 9 quantiles the competition scores, in the order of the submission's rows.
QUANTILES: tuple[float, ...] = (0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995)

# (level_number, human_name, keys of the level's series id), in the competition's order. The id of a group joins its
# keys with "_", and a level of one key ends it with "_X" (the competition's name for "all"): "Total_X", "CA_X",
# "FOODS_1_001_X", "CA_FOODS_1", "WI_FOODS_1_001", "FOODS_1_001_CA_1". The order of the keys of a level is the
# competition's, which is not always the order the accuracy scorer groups them in (``WI_FOODS_1_001``, not
# ``FOODS_1_001_WI``).
LEVELS: list[tuple[int, str, tuple[str, ...]]] = [
    (1, "Total", ()),
    (2, "state_id", ("state_id",)),
    (3, "store_id", ("store_id",)),
    (4, "cat_id", ("cat_id",)),
    (5, "dept_id", ("dept_id",)),
    (6, "state_id x cat_id", ("state_id", "cat_id")),
    (7, "state_id x dept_id", ("state_id", "dept_id")),
    (8, "store_id x cat_id", ("store_id", "cat_id")),
    (9, "store_id x dept_id", ("store_id", "dept_id")),
    (10, "item_id", ("item_id",)),
    (11, "item_id x state_id", ("state_id", "item_id")),
    (12, "item_id x store_id", ("item_id", "store_id")),
]


class Hierarchy(NamedTuple):
    """The groups of the 12 levels over a set of bottom series.

    Attributes
    ----------
    rows : np.ndarray, shape (n_series,)
        Position along the dataset's ``series_id`` of each bottom series.
    ids : list[np.ndarray]
        Per level (``LEVELS`` order), the competition ids of its groups, sorted.
    group_idx : list[np.ndarray]
        Per level, the index into that level's ``ids`` of each bottom series' group.
    """

    rows: np.ndarray
    ids: list[np.ndarray]
    group_idx: list[np.ndarray]

    @property
    def n_groups(self) -> list[int]:
        """Number of groups of each level."""
        return [len(ids) for ids in self.ids]


def build_hierarchy(ds: xr.Dataset, rows: np.ndarray | None = None) -> Hierarchy:
    """Group a set of bottom series at the 12 levels.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from ``m5_preprocess.build_m5_dataset``.
    rows : np.ndarray, optional
        Positions along ``series_id`` of the bottom series to use, by default all of them. A subset gets its own
        groups, so its levels count only these series.

    Returns
    -------
    Hierarchy
        The groups of each level; for the full dataset, 42,840 in all.
    """
    rows = np.arange(ds.sizes["series_id"]) if rows is None else np.asarray(rows)
    meta = {key: ds[key].values[rows].astype(str) for key in ("state_id", "store_id", "cat_id", "dept_id", "item_id")}
    ids, group_idx = [], []
    for _, _, keys in LEVELS:
        if keys:
            labels = reduce(lambda a, b: np.char.add(np.char.add(a, "_"), b), [meta[k] for k in keys])
            if len(keys) == 1:
                labels = np.char.add(labels, "_X")
        else:
            labels = np.full(len(rows), "Total_X")
        level_ids, inverse = np.unique(labels, return_inverse=True)
        ids.append(level_ids)
        group_idx.append(inverse.ravel())
    return Hierarchy(rows, ids, group_idx)


def aggregate(matrix: np.ndarray, group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum the rows of ``matrix`` by group.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_series, ...)
        One row per bottom series; any trailing shape.
    group_idx : np.ndarray, shape (n_series,)
        Group of each row.
    n_groups : int
        Number of groups.

    Returns
    -------
    np.ndarray, shape (n_groups, ...)
        Sum of each group's rows, float64.
    """
    flat = matrix.reshape(matrix.shape[0], -1).astype(np.float64)
    n_series = len(group_idx)
    indicator = sparse.csr_matrix((np.ones(n_series), (group_idx, np.arange(n_series))), shape=(n_groups, n_series))
    return np.asarray(indicator @ flat).reshape((n_groups, *matrix.shape[1:]))


def pinball_loss(y: np.ndarray, q: np.ndarray, u: np.ndarray | float) -> np.ndarray:
    """Pinball loss of the quantile forecast ``q`` at level ``u`` against the actual ``y``.

    Parameters
    ----------
    y, q : np.ndarray
        Actual and forecast quantile, broadcastable to each other.
    u : np.ndarray or float
        Quantile level in (0, 1), broadcastable to ``y`` and ``q``.

    Returns
    -------
    np.ndarray
        ``u * (y - q)`` where ``q <= y``, ``(1 - u) * (q - y)`` where ``q > y``.
    """
    return np.where(q <= y, u * (y - q), (1.0 - u) * (q - y))


def spl_scale(y_train: np.ndarray) -> np.ndarray:
    """Per-series SPL scale: the mean absolute one-step difference from the first non-zero sale on.

    The accuracy metric's scale (``m5_score_submission.rmsse_scale``) with absolute instead of squared differences:
    the leading zeros of a late launch add neither zero-difference terms nor the launch jump.

    Parameters
    ----------
    y_train : np.ndarray, shape (n_series, T)
        Training history, one (possibly aggregated) series per row.

    Returns
    -------
    np.ndarray, shape (n_series,)
        Scale per series. A series with no one-step difference left after trimming, or a flat one, gets 1.0 so its
        loss stays finite.
    """
    started = np.logical_or.accumulate(y_train != 0.0, axis=1)
    # ``started`` is monotone, so a difference lies in the trimmed span exactly when its earlier day does.
    in_span = started[:, :-1]
    abs_diff = np.where(in_span, np.abs(np.diff(y_train, axis=1)), 0.0)
    scale = abs_diff.sum(axis=1) / np.maximum(in_span.sum(axis=1), 1)
    return np.where(scale <= 0.0, 1.0, scale)


def _check_forecasts(forecasts: list[np.ndarray], hierarchy: Hierarchy, n_quantiles: int) -> None:
    """Raise ``ValueError`` unless ``forecasts`` has one ``(n_groups, n_quantiles, N_STEPS)`` array per level."""
    if len(forecasts) != len(LEVELS):
        raise ValueError(f"Expected {len(LEVELS)} forecast arrays (one per level), got {len(forecasts)}")
    for (level_no, _, _), fq, n_groups in zip(LEVELS, forecasts, hierarchy.n_groups, strict=True):
        if np.shape(fq) != (n_groups, n_quantiles, N_STEPS):
            expected = (n_groups, n_quantiles, N_STEPS)
            raise ValueError(f"Level {level_no}: expected forecasts of shape {expected}, got {np.shape(fq)}")


def window_data(
    ds: xr.Dataset, hierarchy: Hierarchy, window: str
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Aggregate the training history, the scored days' actuals and the dollar sales of one window to all levels.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from ``m5_preprocess.build_m5_dataset``.
    hierarchy : Hierarchy
        Groups of the bottom series, from :func:`build_hierarchy`.
    window : {"prevalidation", "validation", "evaluation"}
        Scoring window, a key of ``WINDOWS``.

    Returns
    -------
    y_train, y_score, dollars : list of np.ndarray
        Per level (``LEVELS`` order): the groups' training history ``(n_groups, T_train)``, their actuals over the
        scored days ``(n_groups, N_STEPS)`` and their dollar sales ``(n_groups,)``.

    Raises
    ------
    ValueError
        If ``window`` is not a key of ``WINDOWS``.
    RuntimeError
        If the window is not ``N_STEPS`` days long or the dataset has no actuals for it.
    """
    train_mask, score_mask = window_masks(ds["split"].values, window)
    sales = ds["sales"].values[hierarchy.rows]
    prices = ds["sell_price"].values[hierarchy.rows]
    if int(score_mask.sum()) != N_STEPS:
        raise RuntimeError(f"Expected {window} length {N_STEPS}, got {int(score_mask.sum())}")
    if not np.isfinite(sales[:, score_mask]).all():
        raise RuntimeError(f"The dataset has no {window} actuals ({WINDOWS[window].days}); see m5_fetch.py")

    dollar_per_series = dollar_sales(sales, prices, train_mask)
    y_train, y_score, dollars = [], [], []
    for group_idx, n_groups in zip(hierarchy.group_idx, hierarchy.n_groups, strict=True):
        y_train.append(aggregate(sales[:, train_mask], group_idx, n_groups))
        y_score.append(aggregate(sales[:, score_mask], group_idx, n_groups))
        dollars.append(aggregate(dollar_per_series, group_idx, n_groups))
    return y_train, y_score, dollars


def level_wspl(
    y_train: np.ndarray,
    y_score: np.ndarray,
    forecast: np.ndarray,
    dollars: np.ndarray,
    quantiles: tuple[float, ...] = QUANTILES,
) -> float:
    """WSPL of one level: its groups' scaled pinball losses, weighted by their dollar sales.

    Parameters
    ----------
    y_train : np.ndarray, shape (n_groups, T_train)
        Training history of each group.
    y_score : np.ndarray, shape (n_groups, N_STEPS)
        Actuals of each group over the scored days.
    forecast : np.ndarray, shape (n_groups, n_quantiles, N_STEPS)
        Forecast quantiles of each group.
    dollars : np.ndarray, shape (n_groups,)
        Dollar sales of each group; the weights are these, normalised to sum to 1 (equal when all are 0).
    quantiles : tuple of float, optional
        Quantile levels of ``forecast``'s second axis. Default :data:`QUANTILES`.

    Returns
    -------
    float
        The level's WSPL.
    """
    u = np.asarray(quantiles, dtype=np.float64)[None, :, None]
    spl = pinball_loss(y_score[:, None, :], forecast, u).mean(axis=(1, 2)) / spl_scale(y_train)
    weights = dollars / dollars.sum() if dollars.sum() > 0 else np.full(len(dollars), 1.0 / len(dollars))
    return float(np.sum(weights * spl))


def compute_hierarchical_wspl(
    ds: xr.Dataset,
    forecasts: list[np.ndarray],
    hierarchy: Hierarchy,
    window: str = "validation",
    quantiles: tuple[float, ...] = QUANTILES,
) -> pd.DataFrame:
    """Compute the WSPL at all 12 M5 aggregation levels for one scoring window.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from ``m5_preprocess.build_m5_dataset``: ``sales`` and ``sell_price`` on ``(series_id, date)``,
        the hierarchy keys on ``series_id`` and ``split`` on ``date``.
    forecasts : list of np.ndarray
        Per level (``LEVELS`` order), the forecast quantiles of its groups in ``hierarchy.ids`` order, shape
        ``(n_groups, n_quantiles, N_STEPS)``.
    hierarchy : Hierarchy
        Groups of the scored bottom series, from :func:`build_hierarchy`: the full dataset or a subset.
    window : {"prevalidation", "validation", "evaluation"}, optional
        Scoring window, a key of ``WINDOWS``. Default ``"validation"``.
    quantiles : tuple of float, optional
        Quantile levels of the forecasts' second axis. Default :data:`QUANTILES`.

    Returns
    -------
    pd.DataFrame
        One row per level: ``level``, ``name``, ``n_series`` and ``wspl``. The overall score is the mean of ``wspl``.

    Raises
    ------
    ValueError
        If ``window`` is not a key of ``WINDOWS`` or ``forecasts`` has the wrong shapes.
    RuntimeError
        If the window is not ``N_STEPS`` days long or the dataset has no actuals for it.
    """
    _check_forecasts(forecasts, hierarchy, len(quantiles))
    y_train, y_score, dollars = window_data(ds, hierarchy, window)
    rows = [
        {"level": level_no, "name": name, "n_series": n_groups, "wspl": level_wspl(yt, ys, fq, d, quantiles)}
        for (level_no, name, _), n_groups, yt, ys, fq, d in zip(
            LEVELS, hierarchy.n_groups, y_train, y_score, forecasts, dollars, strict=True
        )
    ]
    return pd.DataFrame(rows)


def log_per_level(per_level: pd.DataFrame, window: str = "validation") -> None:
    """Log a per-level WSPL table and its 12-level mean.

    Parameters
    ----------
    per_level : pd.DataFrame
        Output of :func:`compute_hierarchical_wspl`.
    window : {"prevalidation", "validation", "evaluation"}, optional
        Scoring window ``per_level`` was computed on, named in the table header. Default ``"validation"``.
    """
    logger.info("Per-level WSPL (%s %s):", window, WINDOWS[window].days)
    logger.info("  %-3s  %-22s  %-6s  %-8s", "lvl", "name", "n", "wspl")
    for _, row in per_level.iterrows():
        logger.info("  L%-2d  %-22s  n=%-6d  %.5f", int(row["level"]), row["name"], int(row["n_series"]), row["wspl"])
    logger.info("Hierarchical WSPL (mean of 12 levels, %s): %.6f", window, float(per_level["wspl"].mean()))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score an M5 Uncertainty submission CSV against known labels.")
    p.add_argument("--submission", type=Path, required=True, metavar="CSV", help="Submission CSV to score.")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR, metavar="DIR", help="Directory containing m5_ds.nc.")
    p.add_argument("--window", choices=SUBMISSION_WINDOWS, default="validation", help="Scoring window.")
    return p.parse_args()


def main() -> None:
    """Score a submission CSV on one window and log the per-level WSPL."""
    from m5u_submission import read_forecasts  # here, not at the top: m5u_submission imports this module

    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.submission.exists():
        raise SystemExit(f"Submission not found: {args.submission}")

    ds = load_m5_ds(args.data_dir)
    hierarchy = build_hierarchy(ds)
    logger.info("Reading %s", args.submission)
    forecasts = read_forecasts(pd.read_csv(args.submission), args.window, hierarchy)
    log_per_level(compute_hierarchical_wspl(ds, forecasts, hierarchy, window=args.window), window=args.window)


if __name__ == "__main__":
    main()
