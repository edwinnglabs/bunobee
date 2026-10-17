#!/usr/bin/env python3
"""M5 forecasting script — fit a registered model, forecast, score and write the submission.

Usage examples
--------------
# Fit 10 random series (demo mode) with the baseline model, ssp_dow:
    python m5_run.py

# Fit 50 top-volume series:
    python m5_run.py --n-demo 50 --sample-mode top

# Quick loop: fit 3,000 random series at the validation origin only (about 20 s):
    python m5_run.py --model ssp_dow --n-demo 3000 --origins validation

# Fit all 30,490 series with a named model and generate a submission CSV:
    python m5_run.py --model ssp_dow --n-demo 0

# Review the 20 worst-forecast series of a quick loop in forecasts.pdf:
    python m5_run.py --n-demo 3000 --origins validation --plot-select worst --n-plot 20

``--model`` picks an entry of ``MODELS`` in ``m5_models``: a model plus its settings (for ``ssp_dow``, the
optimiser steps, learning rate and fit chunking). The runner never builds model inputs; adding a model
needs only a new entry there.

``--origins`` picks the forecast origins to fit (see ``WINDOWS`` in ``m5_score_submission``), by default
both:

    validation : fit on d_1-d_1913, forecast d_1914-d_1941  (``_validation`` submission rows)
    evaluation : fit on d_1-d_1941, forecast d_1942-d_1969  (``_evaluation`` submission rows)

``submission.csv`` needs every series at both origins, so only a full run with both origins writes it.

``forecasts.pdf`` reviews each fitted origin: first the total, store and department sums of the fitted
series, then ``--n-plot`` series picked by ``--plot-select`` (at random, by worst RMSSE or by top dollar-sales
weight). Every panel shows the last ``PLOT_HISTORY`` fit days, the actuals and the forecast over the forecast
window, and the model's in-sample fit when the model returns one (``m5_models.INSAMPLE_KEY``).

The run scores each origin's forecasts with the 12-level hierarchical WRMSSE and writes the per-level
tables to ``wrmsse_per_level.csv`` in the run directory, one ``window`` per origin. The evaluation window
is scored only when the dataset carries its actuals (``m5_fetch.py`` + ``m5_preprocess.py``). A sampled
run aggregates and weights only its own series, so its score is comparable with other runs on the same
sample (same ``--n-demo``, ``--sample-mode`` and ``--seed``; see ``m5_compare.py``), not with the full
30,490-series score.

Every run also writes the model name and settings, its wall-clock time and peak memory to ``run_stats.json``
in the run directory, and appends one row to the run log ``runs.csv`` in ``--output-dir`` (see
:func:`_append_run_log`).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import xarray as xr

from m5_common import DATA_DIR, load_m5_ds
from m5_models import DEFAULT_MODEL, INSAMPLE_KEY, MODELS
from m5_score_submission import WINDOWS, compute_hierarchical_wrmsse, dollar_sales, log_per_level, rmsse_scale

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HORIZON = 28
_DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"

#: ``--origins`` choice -> forecast origins to fit, in ``WINDOWS`` order.
ORIGINS: dict[str, tuple[str, ...]] = {
    "validation": ("validation",),
    "evaluation": ("evaluation",),
    "both": tuple(WINDOWS),
}

#: File name of the run log in ``--output-dir``, next to the run directories.
RUN_LOG = "runs.csv"

#: ``--plot-select`` choice -> how ``forecasts.pdf`` orders the fitted series before taking ``--n-plot``.
PLOT_SELECT: dict[str, str] = {"random": "picked at random", "worst": "worst RMSSE first", "top": "top weight first"}

#: Fit days ``forecasts.pdf`` shows before each forecast origin.
PLOT_HISTORY = 90

#: Sums of the fitted series ``forecasts.pdf`` draws after the total: panel name prefix -> the ``series_id``
#: coordinate they group by.
PLOT_SUMS: dict[str, str] = {"store": "store_id", "dept": "dept_id"}

#: Line colours of ``forecasts.pdf``: the first three slots of a colourblind-checked categorical palette.
_PLOT_COLORS = {"actual": "#2a78d6", "forecast": "#eb6834", "insample": "#1baf7a"}


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
        "--origins",
        choices=tuple(ORIGINS),
        default="both",
        help="Forecast origins to fit: validation (d_1914-d_1941), evaluation (d_1942-d_1969) or both, by default "
        "both. A quick loop fits validation only, so it never looks at the evaluation window.",
    )
    p.add_argument(
        "--n-plot",
        type=int,
        default=50,
        metavar="N",
        help="Number of series to include in forecasts.pdf per origin, after the total, store and department "
        "sums. 0 = all fitted series.",
    )
    p.add_argument(
        "--plot-select",
        choices=tuple(PLOT_SELECT),
        default="random",
        help="How to pick the --n-plot series at each origin: random, worst (highest RMSSE over the forecast "
        "window) or top (highest dollar-sales weight), by default random.",
    )
    p.add_argument(
        "--plots-per-page", type=int, default=5, metavar="N", help="Forecast subplots per PDF page, by default 5."
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_model_output(
    model_name: str, forecasts: np.ndarray, fitted: dict[str, np.ndarray], n_series: int, n_fit_days: int
) -> None:
    """Raise if a model's ``fit_forecast`` output breaks the contract of ``m5_models.Model``.

    Parameters
    ----------
    model_name : str
        Registered name of the model, for the error message.
    forecasts : np.ndarray
        The model's forecasts; must have shape ``(n_series, HORIZON)`` and no NaN or infinite values.
    fitted : dict of str to np.ndarray
        The model's per-series outputs; each must have leading dimension ``n_series``, and the in-sample
        fit (``INSAMPLE_KEY``), if there is one, shape ``(n_series, n_fit_days)``.
    n_series : int
        Number of series the model was asked to fit.
    n_fit_days : int
        Number of days the model was asked to fit on.

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
    if INSAMPLE_KEY in fitted and np.shape(fitted[INSAMPLE_KEY]) != (n_series, n_fit_days):
        raise ValueError(
            f"{model_name}: the in-sample fit {INSAMPLE_KEY!r} has shape {np.shape(fitted[INSAMPLE_KEY])}, expected "
            f"{(n_series, n_fit_days)}"
        )


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


@dataclass(frozen=True)
class _Panels:
    """The ``forecasts.pdf`` panels of one forecast origin: one row per series, or per sum of series.

    Attributes
    ----------
    names : list of str
        Panel name of each row: a ``series_id`` or the sum it is (``"Total"``, ``"store CA_1"``, ...).
    dates : np.ndarray
        The fit days, ``(n_fit_days,)``.
    forecast_dates : np.ndarray
        The forecast days, ``(HORIZON,)``.
    history : np.ndarray
        Sales over the fit days, ``(n_rows, n_fit_days)``. The panels show the last ``PLOT_HISTORY``; the
        RMSSE scale uses all of them.
    insample : np.ndarray or None
        The model's in-sample fit over the last ``PLOT_HISTORY`` fit days, ``(n_rows, PLOT_HISTORY)``, or
        None if the model returns none.
    actual : np.ndarray
        Sales over the forecast days, ``(n_rows, HORIZON)``; NaN where the dataset has no actuals.
    forecast : np.ndarray
        Forecasts over the forecast days, ``(n_rows, HORIZON)``.
    weight : np.ndarray
        Share of the fitted series' dollar sales over the last 28 fit days, ``(n_rows,)``: the WRMSSE
        weight of the row at its level, among the fitted series.
    """

    names: list[str]
    dates: np.ndarray
    forecast_dates: np.ndarray
    history: np.ndarray
    insample: np.ndarray | None
    actual: np.ndarray
    forecast: np.ndarray
    weight: np.ndarray

    def rmsse(self) -> np.ndarray:
        """RMSSE of each row's forecast over the forecast days, NaN for rows without actuals."""
        return np.sqrt(np.mean((self.forecast - self.actual) ** 2, axis=1) / rmsse_scale(self.history))

    def bias(self) -> np.ndarray:
        """Each row's forecast total over its actual total, minus one; NaN without actuals or sales."""
        actual = self.actual.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(actual > 0, self.forecast.sum(axis=1) / actual - 1.0, np.nan)

    def take(self, rows: np.ndarray) -> _Panels:
        """The given rows, in the given order."""
        return _Panels(
            names=[self.names[row] for row in rows],
            dates=self.dates,
            forecast_dates=self.forecast_dates,
            history=self.history[rows],
            insample=None if self.insample is None else self.insample[rows],
            actual=self.actual[rows],
            forecast=self.forecast[rows],
            weight=self.weight[rows],
        )

    def summed(self, names: list[str], members: np.ndarray) -> _Panels:
        """Sums of rows: row ``i`` of the result sums the rows where ``members[i]`` is True.

        Parameters
        ----------
        names : list of str
            Panel name of each sum.
        members : np.ndarray
            Boolean ``(len(names), n_rows)`` membership of each sum.
        """

        def total(values: np.ndarray) -> np.ndarray:
            return np.stack([values[row].sum(axis=0, dtype=np.float64) for row in members])

        return _Panels(
            names=names,
            dates=self.dates,
            forecast_dates=self.forecast_dates,
            history=total(self.history),
            insample=None if self.insample is None else total(self.insample),
            actual=total(self.actual),
            forecast=total(self.forecast),
            weight=total(self.weight),
        )


def _origin_panels(
    ds: xr.Dataset, fit_idx: np.ndarray, window: str, forecasts: np.ndarray, insample: np.ndarray | None
) -> _Panels:
    """Panels of the fitted series at one forecast origin, in fit order.

    Parameters
    ----------
    ds : xr.Dataset
        The M5 dataset.
    fit_idx : np.ndarray
        Positions along ``series_id`` of the fitted series.
    window : str
        Forecast origin, a key of ``WINDOWS``.
    forecasts : np.ndarray
        The origin's forecasts, ``(len(fit_idx), HORIZON)``.
    insample : np.ndarray or None
        The model's in-sample fit over the last ``PLOT_HISTORY`` fit days, or None.
    """
    split = ds["split"].values
    fit_splits, forecast_split, _ = WINDOWS[window]
    fit_mask, forecast_mask = np.isin(split, fit_splits), split == forecast_split
    sales = ds["sales"].values[fit_idx]
    dollars = dollar_sales(sales, ds["sell_price"].values[fit_idx], fit_mask)
    dates = ds["date"].values
    return _Panels(
        names=list(ds["series_id"].values[fit_idx]),
        dates=dates[fit_mask],
        forecast_dates=dates[forecast_mask],
        history=sales[:, fit_mask],
        insample=insample,
        actual=sales[:, forecast_mask],
        forecast=forecasts,
        weight=dollars / dollars.sum() if dollars.sum() > 0 else np.full(dollars.size, 1.0 / dollars.size),
    )


def _sum_panels(panels: _Panels, ds: xr.Dataset, fit_idx: np.ndarray) -> _Panels:
    """The total of the fitted series, then their sum per group of each ``PLOT_SUMS`` coordinate.

    Parameters
    ----------
    panels : _Panels
        Panels of the fitted series, in fit order (:func:`_origin_panels`).
    ds : xr.Dataset
        The M5 dataset.
    fit_idx : np.ndarray
        Positions along ``series_id`` of the fitted series.
    """
    names, members = ["Total"], [np.ones(fit_idx.size, dtype=bool)]
    for prefix, key in PLOT_SUMS.items():
        labels = ds[key].values[fit_idx]
        for group in np.unique(labels):
            names.append(f"{prefix} {group}")
            members.append(labels == group)
    return panels.summed(names, np.stack(members))


def _select_series(select: str, panels: _Panels, random_order: np.ndarray, n_plot: int) -> np.ndarray:
    """Rows of the series ``forecasts.pdf`` draws, in drawing order.

    Parameters
    ----------
    select : str
        A ``PLOT_SELECT`` choice: ``"random"`` takes ``random_order``, ``"worst"`` the highest RMSSE first
        (rows without actuals last) and ``"top"`` the highest weight first.
    panels : _Panels
        Panels of the fitted series.
    random_order : np.ndarray
        A random permutation of the rows of ``panels``.
    n_plot : int
        Number of rows to return.

    Returns
    -------
    np.ndarray
        ``n_plot`` row positions.
    """
    if select == "worst":
        order = np.argsort(-panels.rmsse(), kind="stable")  # -NaN is NaN, which argsort puts last
    elif select == "top":
        order = np.argsort(-panels.weight, kind="stable")
    else:
        order = random_order
    return order[:n_plot]


def _panel_title(name: str, rmsse: float, bias: float, weight: float) -> str:
    """Panel title: the name, then the RMSSE and bias when there are actuals, then the weight."""
    parts = [name]
    if np.isfinite(rmsse):
        parts.append(f"RMSSE {rmsse:.2f}")
    if np.isfinite(bias):
        parts.append(f"bias {bias:+.1%}")
    parts.append(f"weight {100 * weight:.3g}%")
    return "   ".join(parts)


def _draw_pages(pdf: PdfPages, panels: _Panels, title: str, per_page: int) -> int:
    """Draw one panel per row of ``panels``, ``per_page`` panels to a page.

    Each panel shows the last ``PLOT_HISTORY`` fit days and the forecast days: the sales (actuals included,
    when the dataset has them), the in-sample fit when there is one, and the forecast.

    Parameters
    ----------
    pdf : PdfPages
        The PDF to append the pages to.
    panels : _Panels
        Rows to draw, in order.
    title : str
        Title of every page, before its page number.
    per_page : int
        Panels per page.

    Returns
    -------
    int
        Number of pages drawn.
    """
    rmsse, bias = panels.rmsse(), panels.bias()
    dates = panels.dates[-PLOT_HISTORY:]
    sales_dates = np.concatenate([dates, panels.forecast_dates])
    sales = np.concatenate([panels.history[:, -PLOT_HISTORY:], panels.actual], axis=1)
    # Halfway between the last fit day and the first forecast day.
    origin = dates[-1] + (panels.forecast_dates[0] - dates[-1]) / 2
    n_rows = len(panels.names)
    n_pages = math.ceil(n_rows / per_page)
    for page in range(n_pages):
        rows = range(page * per_page, min((page + 1) * per_page, n_rows))
        fig, axes = plt.subplots(len(rows), 1, figsize=(12, 3 * len(rows)), sharex=True, squeeze=False)
        axes = axes[:, 0]
        for ax, row in zip(axes, rows):
            ax.plot(sales_dates, sales[row], color=_PLOT_COLORS["actual"], linewidth=1.2, label="Actual")
            if panels.insample is not None:
                ax.plot(
                    dates, panels.insample[row], color=_PLOT_COLORS["insample"], linewidth=1.2, label="In-sample fit"
                )
            ax.plot(
                panels.forecast_dates,
                panels.forecast[row],
                color=_PLOT_COLORS["forecast"],
                linewidth=1.5,
                linestyle="--",
                label="Forecast",
            )
            ax.axvline(origin, color="grey", linestyle=":", alpha=0.5)
            ax.set_title(
                _panel_title(panels.names[row], rmsse[row], bias[row], panels.weight[row]), fontsize=9, loc="left"
            )
            ax.set_ylabel("Units")
        axes[0].legend(fontsize=8, loc="upper left")
        fig.suptitle(f"{title}  (page {page + 1}/{n_pages})", y=1.01)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    return n_pages


def _peak_rss_gib() -> float:
    """Peak resident set size of this process so far, in GiB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in bytes on macOS and in kilobytes on Linux.
    return peak / 2**30 if sys.platform == "darwin" else peak / 2**20


def _git_state() -> tuple[str | None, bool | None]:
    """Commit and dirty flag of the checkout this script runs from.

    Returns
    -------
    commit : str or None
        Short hash of ``HEAD``, or None if git or the repository is unavailable.
    dirty : bool or None
        Whether tracked files have uncommitted changes (untracked files do not count, as in
        ``git describe --dirty``), or None if git or the repository is unavailable.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=Path(__file__).parent, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return git("rev-parse", "--short", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _append_run_log(path: Path, row: dict[str, object]) -> None:
    """Append one run's row to the run log, creating the log with a header if it does not exist yet.

    The log is read back as text and rewritten, so values typed into it by hand (the Kaggle scores of a
    submitted run) are kept verbatim, and a column new in ``row`` is added with blanks for earlier runs.

    Parameters
    ----------
    path : Path
        The run log CSV.
    row : dict of str to object
        Column name -> value for this run. None is written as a blank.
    """
    log = pd.DataFrame([{key: "" if value is None else str(value) for key, value in row.items()}])
    if path.exists():
        log = pd.concat([pd.read_csv(path, dtype=str, keep_default_na=False), log], ignore_index=True)
    log.to_csv(path, index=False)


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
    windows = ORIGINS[args.origins]
    # Read at the start: the run log should record the code this run fits, not edits made while it runs.
    git_commit, git_dirty = _git_state()

    # Create timestamped output directory
    started = datetime.now()
    run_dir: Path = args.output_dir / started.strftime("%Y%m%d_%H%M%S")
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
    # 4. Fit + predict, once per forecast origin in --origins
    # ------------------------------------------------------------------
    model = MODELS[args.model]
    settings = asdict(model)
    fitted_by_window: dict[str, dict[str, np.ndarray]] = {}
    forecasts_by_window: dict[str, np.ndarray] = {}
    fit_seconds_by_window: dict[str, float] = {}
    insample_by_window: dict[str, np.ndarray | None] = {}
    for window in windows:
        fit_splits, forecast_split, days = WINDOWS[window]
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
        _check_model_output(
            args.model, forecasts_by_window[window], fitted_by_window[window], n_fit, int(fit_mask.sum())
        )
        # The in-sample fit is as large as the sales it fits, so only the days forecasts.pdf shows are kept, and
        # it is neither histogrammed nor saved with the other fitted arrays.
        insample = fitted_by_window[window].pop(INSAMPLE_KEY, None)
        insample_by_window[window] = None if insample is None else np.asarray(insample)[:, -PLOT_HISTORY:]
        logger.info(
            "Fitted %d series in %.1fs  (%.1f ms/series, peak RSS so far %.1f GiB)",
            n_fit,
            fit_seconds_by_window[window],
            fit_seconds_by_window[window] / n_fit * 1000,
            _peak_rss_gib(),
        )
    fit_seconds = sum(fit_seconds_by_window.values())

    # ------------------------------------------------------------------
    # 5. Diagnostics plot of the first origin fitted: validation, unless only evaluation is
    # ------------------------------------------------------------------
    plot_window = windows[0]
    _plot_fitted(
        fitted_by_window[plot_window],
        title=f"{args.model} fitted outputs — {plot_window} origin",
        output_path=run_dir / "diagnostics.png",
    )

    # ------------------------------------------------------------------
    # 6. Forecast PDF: at each origin, the total, store and department sums of the fitted series, then
    #    --n-plot series picked by --plot-select (paginated, --plots-per-page panels per page)
    # ------------------------------------------------------------------
    n_plot = n_fit if args.n_plot == 0 else min(args.n_plot, n_fit)
    # One random order for every origin, so --plot-select random draws the same series at each.
    random_order = rng.permutation(n_fit)
    n_pages = 0
    with PdfPages(run_dir / "forecasts.pdf") as pdf:
        for window in windows:
            panels = _origin_panels(ds, fit_idx, window, forecasts_by_window[window], insample_by_window[window])
            heading = f"{args.model} forecast, {window} origin"
            n_pages += _draw_pages(
                pdf,
                _sum_panels(panels, ds, fit_idx),
                f"{heading} — total, store and department sums of the {n_fit} series [{label}]",
                args.plots_per_page,
            )
            select = args.plot_select
            if select == "worst" and not np.isfinite(panels.actual).any():
                logger.warning(
                    "forecasts.pdf picks the %s-origin series at random: the dataset has no actuals for %s, so "
                    "there is no RMSSE to pick the worst by.",
                    window,
                    WINDOWS[window][2],
                )
                select = "random"
            rows = _select_series(select, panels, random_order, n_plot)
            n_pages += _draw_pages(
                pdf,
                panels.take(rows),
                f"{heading} — {n_plot} of the {n_fit} series [{label}], {PLOT_SELECT[select]}",
                args.plots_per_page,
            )

    logger.info(
        "Saved forecasts.pdf  (%s; sums and %d series each, %s; %d pages)",
        " and ".join(windows),
        n_plot,
        PLOT_SELECT[args.plot_select],
        n_pages,
    )

    # ------------------------------------------------------------------
    # 7. Save the model's fitted outputs: validation-origin arrays under their own names, evaluation-origin
    #    arrays with an "_evaluation" suffix
    # ------------------------------------------------------------------
    arrays: dict[str, np.ndarray] = {}
    for window in windows:
        suffix = "" if window == "validation" else f"_{window}"
        arrays.update({f"{key}{suffix}": value for key, value in fitted_by_window[window].items()})
        arrays[f"forecasts{suffix}"] = forecasts_by_window[window]
    np.savez(run_dir / "batch_params.npz", **arrays, fit_idx=fit_idx)
    logger.info("Saved batch_params.npz")

    # ------------------------------------------------------------------
    # 8. Submission CSV (only when all series are fitted at both origins)
    # ------------------------------------------------------------------
    base_ids = ds["series_id"].values[fit_idx]
    if n_fit == n_series_total and windows == tuple(WINDOWS):
        sample_sub = pd.read_csv(args.data_dir / "sample_submission.csv")
        _make_submission(forecasts_by_window, base_ids, sample_sub, output_path=run_dir / "submission.csv")
    else:
        logger.info(
            "Skipping submission — it needs all %d series (--n-demo 0) at both origins (--origins both); this run "
            "fitted %d series at %s.",
            n_series_total,
            n_fit,
            " and ".join(windows),
        )

    # ------------------------------------------------------------------
    # 9. WRMSSE of the fitted series, per fitted window with actuals
    # ------------------------------------------------------------------
    tables = []
    for window in windows:
        _, score_split, days = WINDOWS[window]
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
    scores = {table["window"][0]: float(table["wrmsse"].mean()) for table in tables}
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(run_dir / "wrmsse_per_level.csv", index=False)
        logger.info("Saved wrmsse_per_level.csv  (%d series, windows: %s)", n_fit, ", ".join(scores))
    else:
        logger.warning("No window was scored, so there is no wrmsse_per_level.csv.")
    if tables and n_fit < n_series_total:
        logger.warning(
            "Sampled WRMSSE: only %d / %d series are aggregated and weighted, so this score is comparable with other "
            "runs on the same sample (same --n-demo, --sample-mode and --seed; see m5_compare.py), not with a "
            "full-run score.",
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
        "origins": list(windows),
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

    # ------------------------------------------------------------------
    # 11. Run log: one row per run, next to the run directories. The Kaggle columns are left blank, to be
    #     filled in by hand once the run's submission is scored.
    # ------------------------------------------------------------------
    run_log_path = args.output_dir / RUN_LOG
    _append_run_log(
        run_log_path,
        {
            "timestamp": started.isoformat(timespec="seconds"),
            "run_dir": run_dir.name,
            "model": args.model,
            "settings": json.dumps(settings),
            "n_series": n_fit,
            "sample_mode": args.sample_mode,
            "seed": args.seed,
            "origins": args.origins,
            **{f"wrmsse_{window}": round(scores[window], 6) if window in scores else None for window in WINDOWS},
            "fit_seconds": run_stats["fit_seconds"],
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "kaggle_public": None,
            "kaggle_private": None,
        },
    )
    logger.info("Appended this run to %s", run_log_path)

    logger.info("Done. Outputs in %s", run_dir)


if __name__ == "__main__":
    main()
