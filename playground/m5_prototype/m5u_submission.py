#!/usr/bin/env python3
"""Write and read an M5 Uncertainty submission (issue #157).

A submission has one row per ``<series_id>_<quantile>_<window>`` -- 42,840 series x 9 quantiles x 2 windows (the
``validation`` rows score on the public leaderboard, the ``evaluation`` rows on the private one) = 771,120 rows -- and
the columns ``F1`` ... ``F28``. Kaggle matches rows by id but the row order of its ``sample_submission.csv`` is the
reference, so the writer takes the ids and their order from that file and fails if the forecasts do not cover exactly
those rows.

The uncertainty competition's ``sample_submission.csv`` has the name of the accuracy competition's, so it lives in
``playground/resource/m5-forecasting-uncertainty/`` (``m5_fetch.py`` puts it there).

The forecasts are the per-level quantile arrays of ``m5u_score``: for each level, one ``(n_groups, 9, 28)`` array
in the order of ``Hierarchy.ids``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from m5_score_submission import N_STEPS, SUBMISSION_WINDOWS
from m5u_score import LEVELS, QUANTILES, Hierarchy

#: Rows of a full submission: 42,840 series x 9 quantiles x 2 windows.
N_ROWS = 42_840 * len(QUANTILES) * len(SUBMISSION_WINDOWS)

#: Forecast columns of a submission.
F_COLS = [f"F{i}" for i in range(1, N_STEPS + 1)]


class SubmissionError(ValueError):
    """A submission whose ids, row count or values do not match the sample submission."""


def read_sample_ids(path: Path | str) -> np.ndarray:
    """Read the row ids of the uncertainty competition's ``sample_submission.csv``.

    Parameters
    ----------
    path : Path | str
        The sample submission.

    Returns
    -------
    np.ndarray, shape (771120,)
        The ``id`` column, in file order; :data:`N_ROWS` of them in the competition's file.

    Raises
    ------
    SystemExit
        If the file is missing, with a message naming ``m5_fetch.py``.
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"{path} not found; run m5_fetch.py first.")
    return pd.read_csv(path, usecols=["id"])["id"].to_numpy(dtype=str)


def split_ids(ids: np.ndarray) -> pd.DataFrame:
    """Split submission ids ``<series_id>_<quantile>_<window>`` into their three parts.

    Parameters
    ----------
    ids : np.ndarray
        Submission ids.

    Returns
    -------
    pd.DataFrame
        One row per id: ``series_id``, ``quantile`` (a float) and ``window``.
    """
    parts = pd.Series(ids).str.rsplit("_", n=2, expand=True)
    parts.columns = ["series_id", "quantile", "window"]
    parts["quantile"] = parts["quantile"].astype(float)
    return parts


def _row_lookup(ids: np.ndarray, hierarchy: Hierarchy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map each submission id to its position in the stacked hierarchy, its quantile and its window.

    Returns
    -------
    series_pos, quantile_pos, window_pos : np.ndarray
        Per id: the position among all groups (levels stacked in ``LEVELS`` order), in :data:`QUANTILES` and in
        :data:`SUBMISSION_WINDOWS`. Each is -1 where the id has no match.
    """
    parts = split_ids(ids)
    series_pos = pd.Index(np.concatenate(hierarchy.ids)).get_indexer(parts["series_id"])
    quantile_pos = pd.Index(QUANTILES).get_indexer(parts["quantile"])
    window_pos = pd.Index(SUBMISSION_WINDOWS).get_indexer(parts["window"])
    return series_pos, quantile_pos, window_pos


def build_submission(
    forecasts: dict[str, list[np.ndarray]], hierarchy: Hierarchy, sample_ids: np.ndarray
) -> pd.DataFrame:
    """Lay the per-level quantile forecasts of both windows out in the sample submission's rows.

    Parameters
    ----------
    forecasts : dict
        For each window of :data:`SUBMISSION_WINDOWS` (``"validation"``, ``"evaluation"``), the forecast quantiles
        per level (``LEVELS`` order), each ``(n_groups, 9, N_STEPS)`` in ``hierarchy.ids`` order.
    hierarchy : Hierarchy
        Groups of all 30,490 bottom series, from ``m5u_score.build_hierarchy``.
    sample_ids : np.ndarray
        Row ids of the sample submission, from :func:`read_sample_ids`.

    Returns
    -------
    pd.DataFrame
        ``id`` and ``F1`` ... ``F28`` in the order of ``sample_ids``.

    Raises
    ------
    SubmissionError
        If a window is missing or a forecast has the wrong shape, an id of the sample has no forecast or occurs
        twice, or a forecast is not finite.
    """
    n_groups = sum(hierarchy.n_groups)
    stacked = []
    for window in SUBMISSION_WINDOWS:
        if window not in forecasts:
            raise SubmissionError(f"No forecasts for the {window} window")
        levels = forecasts[window]
        if len(levels) != len(LEVELS):
            raise SubmissionError(f"The {window} window has {len(levels)} forecast arrays; expected {len(LEVELS)}")
        stack = np.concatenate([np.asarray(f, dtype=np.float64) for f in levels])
        if stack.shape != (n_groups, len(QUANTILES), N_STEPS):
            raise SubmissionError(f"The {window} forecasts stack to {stack.shape}")
        if not np.isfinite(stack).all():
            raise SubmissionError(f"The {window} forecasts are not all finite")
        stacked.append(stack)

    series_pos, quantile_pos, window_pos = _row_lookup(sample_ids, hierarchy)
    unmatched = (series_pos < 0) | (quantile_pos < 0) | (window_pos < 0)
    if unmatched.any():
        raise SubmissionError(f"{int(unmatched.sum())} sample ids have no forecast, e.g. {sample_ids[unmatched][:3]}")
    if len(set(sample_ids)) != len(sample_ids):
        raise SubmissionError("The sample ids are not unique")
    if len(sample_ids) != n_groups * len(QUANTILES) * len(SUBMISSION_WINDOWS):
        raise SubmissionError(f"The sample has {len(sample_ids)} rows; the forecasts cover {n_groups * 18}")

    values = np.empty((len(sample_ids), N_STEPS))
    for w, stack in enumerate(stacked):
        rows = window_pos == w
        values[rows] = stack[series_pos[rows], quantile_pos[rows]]
    sub = pd.DataFrame(values, columns=F_COLS)
    sub.insert(0, "id", sample_ids)
    return sub


def check_submission(sub: pd.DataFrame, sample_ids: np.ndarray) -> None:
    """Check a submission against the sample's ids, row count and order.

    Parameters
    ----------
    sub : pd.DataFrame
        Submission with an ``id`` column and ``F1`` ... ``F28``.
    sample_ids : np.ndarray
        Row ids of the sample submission, from :func:`read_sample_ids`.

    Raises
    ------
    SubmissionError
        If the columns, the row count, the ids or their order differ from the sample's, or a value is not finite.
    """
    if list(sub.columns) != ["id", *F_COLS]:
        raise SubmissionError(f"Expected the columns id, F1 ... F{N_STEPS}; got {list(sub.columns)[:3]} ...")
    if len(sub) != len(sample_ids):
        raise SubmissionError(f"The submission has {len(sub)} rows; the sample has {len(sample_ids)}")
    ids = sub["id"].to_numpy(dtype=str)
    if not np.array_equal(ids, sample_ids):
        first = int(np.argmax(ids != sample_ids))
        raise SubmissionError(f"Row {first} is {ids[first]!r}; the sample has {sample_ids[first]!r}")
    if not np.isfinite(sub[F_COLS].to_numpy()).all():
        raise SubmissionError("The submission has values that are not finite")


def write_submission(
    forecasts: dict[str, list[np.ndarray]],
    hierarchy: Hierarchy,
    sample_path: Path | str,
    output: Path | str,
) -> pd.DataFrame:
    """Write a full submission CSV in the sample's row order, after checking it against the sample.

    Nothing is sent to Kaggle.

    Parameters
    ----------
    forecasts : dict
        Per-window, per-level forecast quantiles, as :func:`build_submission` takes them.
    hierarchy : Hierarchy
        Groups of all 30,490 bottom series.
    sample_path : Path | str
        The uncertainty competition's ``sample_submission.csv``.
    output : Path | str
        Where to write the CSV; its folder is created if missing.

    Returns
    -------
    pd.DataFrame
        The submission that was written.

    Raises
    ------
    SubmissionError
        As :func:`build_submission` and :func:`check_submission`. For the full hierarchy the row count is
        :data:`N_ROWS`, 771,120.
    """
    sample_ids = read_sample_ids(sample_path)
    sub = build_submission(forecasts, hierarchy, sample_ids)
    check_submission(sub, sample_ids)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output, index=False, float_format="%.6g")
    return sub


def read_forecasts(sub: pd.DataFrame, window: str, hierarchy: Hierarchy) -> list[np.ndarray]:
    """Read one window's rows of a submission back into per-level quantile arrays.

    Parameters
    ----------
    sub : pd.DataFrame
        Submission with an ``id`` column and ``F1`` ... ``F28``.
    window : {"validation", "evaluation"}
        Window whose rows to read.
    hierarchy : Hierarchy
        Groups of all 30,490 bottom series; the submission must cover each of them at every quantile.

    Returns
    -------
    list of np.ndarray
        Per level (``LEVELS`` order), shape ``(n_groups, 9, N_STEPS)`` in ``hierarchy.ids`` order.

    Raises
    ------
    SubmissionError
        If ``window`` is not a submission window, or a group has no row at some quantile of the window.
    """
    if window not in SUBMISSION_WINDOWS:
        raise SubmissionError(f"Unknown window {window!r}; expected one of {SUBMISSION_WINDOWS}")
    rows = sub[sub["id"].str.endswith(f"_{window}")]
    series_pos, quantile_pos, _ = _row_lookup(rows["id"].to_numpy(dtype=str), hierarchy)
    known = (series_pos >= 0) & (quantile_pos >= 0)
    stack = np.full((sum(hierarchy.n_groups), len(QUANTILES), N_STEPS), np.nan)
    stack[series_pos[known], quantile_pos[known]] = rows[F_COLS].to_numpy(dtype=np.float64)[known]
    if np.isnan(stack).any():
        n_missing = int(np.isnan(stack[..., 0]).sum())
        raise SubmissionError(f"The submission lacks {n_missing} series x quantile rows of the {window} window")
    return np.split(stack, np.cumsum(hierarchy.n_groups)[:-1])
