#!/usr/bin/env python3
"""Diagnose the forecast level bias of registered M5 models: which mean of the sales does the forecast track?

Usage
-----
    # The quick-loop sample (3,000 random series, seed 2026) at the validation origin, both SSP entries:
    python m5_bias.py --model ssp_dow ssp_level_dow

    # One model on a smaller sample:
    python m5_bias.py --model ssp_dow --n-demo 500

The SSP models fit ``log(max(sales, 0.1) / mean)``, so on intermittent series their level may follow the
geometric mean of the clipped sales rather than the arithmetic mean that WRMSSE rewards (issue #113). Run
outputs keep no per-series forecasts, so this script fits each ``--model`` (an entry of ``MODELS`` in
``m5_models.py``) at the validation origin, on the same series ``m5_run.py`` picks for the same ``--n-demo``
and ``--seed``, and compares each series' forecast with:

- the arithmetic mean of the actuals over the 28 forecast days,
- the arithmetic and geometric means of ``max(sales, 0.1)`` over two windows of fit days: all of them
  (``fit``, the days the model normalises by) and the last ``--recent-days`` (``last28`` by default, the
  local level the filtered states follow).

It then logs, per model and window, a table of ratios for the total, each department, each volume tercile
and each zero-share bin (see :func:`group_table`). Every ratio is a ratio of sums over the group's series, so
``fc/act`` is the group's forecast total over its actual total, the bias ``forecasts.pdf`` gives a department
panel plus one. Two columns say which mean the forecast tracks, both 0 for the geometric mean and 1 for the
arithmetic mean: ``position`` places the group's forecast between the two means on the log scale, and
``slope`` is how the forecast moves with the gap between them across the group's series (see
:func:`tracking_slope`). With two or more models it also logs each group's bias per model and the change from
the first model.

Read which mean the forecast tracks off the ``last28`` window. Many series have hundreds of zero days before
their item was first stocked, which pull the ``fit`` means, the geometric one most, far below the level the
forecast starts from, so on that window both columns sit near 1 whatever the forecast tracks.

Volume terciles split the series into thirds by their mean sales over the fit days. Zero-share bins group
them by the share of zero days from their first sale on, so days before an item was stocked do not count.

The per-series statistics and the group tables are written to ``series.csv`` and ``groups.csv`` in
``<--output-dir>/bias_<timestamp>/``. The evaluation window is never fitted or read.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from m5_common import DATA_DIR, load_m5_ds
from m5_models import DEFAULT_MODEL, INSAMPLE_KEY, MODELS
from m5_score_submission import WINDOWS

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"

#: Floor applied to the sales before the means, as the SSP models' ``LogTransform`` clips them before its log.
SALES_FLOOR = 0.1

#: Volume tercile labels, from the lowest mean sales over the fit days to the highest.
VOLUME_TERCILES = ("low", "mid", "high")

#: Upper edges of the zero-share bins, as shares of the days from a series' first sale on; each bin is
#: (previous edge, edge], the first starting at 0 inclusive.
ZERO_SHARE_EDGES = (0.25, 0.5, 0.75, 1.0)

#: Groupings of :func:`group_table`, in output order: name -> ``series.csv`` column of each series' group,
#: or None for one group of all series.
GROUPINGS: dict[str, str | None] = {"total": None, "dept": "dept_id", "volume": "volume", "zero_share": "zero_bin"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare M5 forecasts with the arithmetic and geometric means of sales.")
    p.add_argument(
        "--model",
        nargs="+",
        choices=tuple(MODELS),
        default=[DEFAULT_MODEL],
        metavar="NAME",
        help=f"Registered models to fit (see MODELS in m5_models.py), by default {DEFAULT_MODEL}. With two or more, "
        "the bias of each group is compared with the first model's.",
    )
    p.add_argument("--data-dir", type=Path, default=DATA_DIR, metavar="DIR", help="Directory containing m5_ds.nc.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help="Root output directory; a bias_<timestamp> sub-dir is created inside.",
    )
    p.add_argument(
        "--n-demo",
        type=int,
        default=3000,
        metavar="N",
        help="Number of random series to fit, by default 3000 (the quick-loop sample). 0 = all 30,490 series.",
    )
    p.add_argument("--seed", type=int, default=2026, help="RNG seed of the random sample, by default 2026.")
    p.add_argument(
        "--recent-days",
        type=int,
        default=28,
        metavar="N",
        help="Fit days at the end of the fit window for the local means, by default 28.",
    )
    return p.parse_args()


def sample_series(n_series_total: int, n_demo: int, seed: int) -> np.ndarray:
    """Positions of the series to fit: the random sample ``m5_run.py`` draws for the same arguments.

    Parameters
    ----------
    n_series_total : int
        Number of series in the dataset.
    n_demo : int
        Number of series to draw; 0 = all of them.
    seed : int
        RNG seed.

    Returns
    -------
    np.ndarray
        Positions along ``series_id``, in fit order.
    """
    n_fit = n_series_total if n_demo == 0 else n_demo
    return np.random.default_rng(seed).choice(n_series_total, size=n_fit, replace=False)


def window_names(recent_days: int = 28) -> tuple[str, str]:
    """Names of the two fit-day windows of :func:`series_stats`: all fit days, then the last ``recent_days``."""
    return "fit", f"last{recent_days}"


def _clipped_means(sales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Arithmetic and geometric mean of ``max(sales, SALES_FLOOR)`` along the last axis."""
    clipped = np.maximum(sales, SALES_FLOOR)
    return clipped.mean(axis=1), np.exp(np.log(clipped).mean(axis=1))


def series_stats(
    sales_fit: np.ndarray, actual: np.ndarray, forecasts: np.ndarray, recent_days: int = 28
) -> pd.DataFrame:
    """Per-series means of the forecast, the actuals and the clipped fit-day sales.

    Parameters
    ----------
    sales_fit : np.ndarray
        Sales over the fit days, ``(n_series, n_fit_days)``.
    actual : np.ndarray
        Sales over the forecast days, ``(n_series, horizon)``.
    forecasts : np.ndarray
        Forecasts over the forecast days, ``(n_series, horizon)``.
    recent_days : int
        Fit days at the end of the fit window for the ``last<recent_days>`` window, by default 28.

    Returns
    -------
    pd.DataFrame
        One row per series: ``forecast_mean`` and ``actual_mean`` over the forecast days; ``mean_fit``, the
        mean sales over the fit days; ``zero_share_active``, the share of zero days from the first sale on
        (all fit days if there is none); then for each window ``w`` of :func:`window_names`, ``amean_w`` and
        ``gmean_w``, the arithmetic and geometric means of ``max(sales, SALES_FLOOR)`` over its days, and
        ``zero_share_w``, its share of zero days.
    """
    sales_fit = np.asarray(sales_fit, dtype=np.float64)
    stats = {
        "forecast_mean": np.asarray(forecasts, dtype=np.float64).mean(axis=1),
        "actual_mean": np.asarray(actual, dtype=np.float64).mean(axis=1),
        "mean_fit": sales_fit.mean(axis=1),
    }
    first_sale = np.argmax(sales_fit > 0, axis=1)  # 0 for a series that never sells: all days count
    active = np.arange(sales_fit.shape[1]) >= first_sale[:, None]
    stats["zero_share_active"] = ((sales_fit == 0) & active).sum(axis=1) / active.sum(axis=1)
    for window, days in zip(window_names(recent_days), (sales_fit, sales_fit[:, -recent_days:])):
        stats[f"amean_{window}"], stats[f"gmean_{window}"] = _clipped_means(days)
        stats[f"zero_share_{window}"] = (days == 0).mean(axis=1)
    return pd.DataFrame(stats)


def assign_groups(stats: pd.DataFrame) -> pd.DataFrame:
    """Add each series' volume tercile and zero-share bin to :func:`series_stats` output.

    Parameters
    ----------
    stats : pd.DataFrame
        Output of :func:`series_stats`, one row per series.

    Returns
    -------
    pd.DataFrame
        ``stats`` with two ordered categorical columns: ``volume``, a :data:`VOLUME_TERCILES` label from
        the rank of ``mean_fit`` (thirds of equal size, ties broken by row order), and ``zero_bin``, the
        :data:`ZERO_SHARE_EDGES` bin of ``zero_share_active``, labelled like ``"25-50%"``.
    """
    n = len(stats)
    rank = np.argsort(np.argsort(stats["mean_fit"].to_numpy(), kind="stable"), kind="stable")
    tercile = np.asarray(VOLUME_TERCILES)[rank * len(VOLUME_TERCILES) // max(n, 1)]

    edges = (0.0, *ZERO_SHARE_EDGES)
    labels = [f"{100 * lo:.0f}-{100 * hi:.0f}%" for lo, hi in zip(edges[:-1], edges[1:])]
    # right=True puts a share equal to an edge in the bin it closes, so the last bin ends at 1 inclusive.
    zero_bin = np.asarray(labels)[np.digitize(stats["zero_share_active"], ZERO_SHARE_EDGES, right=True)]
    return stats.assign(
        volume=pd.Categorical(tercile, categories=VOLUME_TERCILES, ordered=True),
        zero_bin=pd.Categorical(zero_bin, categories=labels, ordered=True),
    )


def _ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, NaN unless the denominator is positive."""
    return numerator / denominator if denominator > 0 else float("nan")


def tracking_slope(members: pd.DataFrame, window: str) -> float:
    """Slope of ``log(forecast / gm)`` on ``log(am / gm)`` across series: which mean the forecast moves with.

    A forecast that is the geometric mean times a constant has slope 0; one that is the arithmetic mean
    times a constant has slope 1. Series with a non-positive forecast are left out.

    Parameters
    ----------
    members : pd.DataFrame
        :func:`series_stats` rows of the group's series.
    window : str
        A window of :func:`window_names`, whose means to use.

    Returns
    -------
    float
        The least-squares slope, NaN if fewer than two series are left or their mean gaps are all equal.
    """
    members = members[members["forecast_mean"] > 0]
    gm = members[f"gmean_{window}"].to_numpy()
    gap = np.log(members[f"amean_{window}"].to_numpy() / gm)
    lift = np.log(members["forecast_mean"].to_numpy() / gm)
    if gap.size < 2 or np.ptp(gap) == 0:
        return float("nan")
    return float(np.cov(gap, lift)[0, 1] / np.var(gap, ddof=1))


def _group_ratios(members: pd.DataFrame, window: str) -> dict[str, float]:
    """The ratio columns of :func:`group_table` for one group of series and one window."""
    fc, act = members["forecast_mean"].sum(), members["actual_mean"].sum()
    am, gm = members[f"amean_{window}"].sum(), members[f"gmean_{window}"].sum()
    # The arithmetic mean is never below the geometric mean; with no gap between them there is no position.
    gap = np.log(am / gm)
    return {
        "n": len(members),
        "zero_share": float(members[f"zero_share_{window}"].mean()),
        "fc_act": _ratio(fc, act),
        "act_am": _ratio(act, am),
        "fc_am": fc / am,
        "fc_gm": fc / gm,
        "am_gm": am / gm,
        "position": float(np.log(fc / gm) / gap) if gap > 1e-9 and fc > 0 else float("nan"),
        "slope": tracking_slope(members, window),
    }


def group_table(stats: pd.DataFrame, window: str) -> pd.DataFrame:
    """Ratios of the forecast to the actuals and to the means of one window, for every group of series.

    Parameters
    ----------
    stats : pd.DataFrame
        Output of :func:`assign_groups`, with a ``dept_id`` column.
    window : str
        A window of :func:`window_names`.

    Returns
    -------
    pd.DataFrame
        One row per group, in :data:`GROUPINGS` order: ``grouping``, ``group``, ``n`` (series), ``zero_share``
        (mean share of zero days over the window) and, as ratios of sums over the group's series,
        ``fc_act`` (forecast over actuals: bias + 1), ``act_am`` (actuals over the arithmetic mean),
        ``fc_am`` and ``fc_gm`` (forecast over the arithmetic and the geometric mean), ``am_gm`` (the gap
        between the two means) and ``position``, ``log(fc_gm) / log(am_gm)``: 0 when the forecast sits at
        the geometric mean, 1 at the arithmetic mean, NaN when the two means coincide. Last, ``slope``, the
        :func:`tracking_slope` across the group's series: 0 when the forecast moves with the geometric mean,
        1 with the arithmetic mean.
    """
    rows = []
    for grouping, column in GROUPINGS.items():
        groups = [("all", stats)] if column is None else stats.groupby(column, sort=True, observed=True)
        for group, members in groups:
            rows.append({"grouping": grouping, "group": str(group), **_group_ratios(members, window)})
    return pd.DataFrame(rows)


def log_group_table(table: pd.DataFrame, model: str, window: str, window_days: str) -> None:
    """Log one model's :func:`group_table` for one window.

    Parameters
    ----------
    table : pd.DataFrame
        Output of :func:`group_table`.
    model : str
        Registered model name.
    window : str
        The table's window.
    window_days : str
        What the window covers, for the heading.
    """
    logger.info(
        "%s, means over %s (%s): ratios of sums; position and slope 0 = geometric mean, 1 = arithmetic mean",
        model,
        window,
        window_days,
    )
    logger.info(
        "  %-11s %-12s %5s  %6s  %6s  %6s  %6s  %6s  %6s  %8s  %6s",
        *("grouping", "group", "n", "zero%", "fc/act", "act/am", "fc/am", "fc/gm", "am/gm", "position", "slope"),
    )
    for row in table.itertuples(index=False):
        logger.info(
            "  %-11s %-12s %5d  %5.1f%%  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %8.2f  %6.2f",
            row.grouping,
            row.group,
            row.n,
            100 * row.zero_share,
            row.fc_act,
            row.act_am,
            row.fc_am,
            row.fc_gm,
            row.am_gm,
            row.position,
            row.slope,
        )


def log_model_comparison(groups: pd.DataFrame, window: str) -> None:
    """Log each group's bias, position and slope per model, and the change of its bias from the first model.

    Parameters
    ----------
    groups : pd.DataFrame
        The :func:`group_table` of every model, stacked with ``model`` and ``window`` columns, the first
        model first.
    window : str
        The window whose ``position`` and ``slope`` to show; the bias does not depend on it.
    """
    rows = groups[groups["window"] == window]
    models = list(dict.fromkeys(rows["model"]))
    base = models[0]
    for model in models[1:]:
        pair = rows[rows["model"] == base].merge(
            rows[rows["model"] == model], on=["grouping", "group"], suffixes=("_a", "_b")
        )
        logger.info("Bias (fc/act - 1), position and slope over %s: A = %s, B = %s, delta = B - A", window, base, model)
        logger.info(
            "  %-11s %-12s %7s  %7s  %7s   %5s  %5s   %5s  %5s",
            *("grouping", "group", "bias_A", "bias_B", "delta", "pos_A", "pos_B", "slp_A", "slp_B"),
        )
        for row in pair.itertuples(index=False):
            bias_a, bias_b = 100 * (row.fc_act_a - 1), 100 * (row.fc_act_b - 1)
            logger.info(
                "  %-11s %-12s %+6.1f%%  %+6.1f%%  %+6.1f%%   %5.2f  %5.2f   %5.2f  %5.2f",
                row.grouping,
                row.group,
                bias_a,
                bias_b,
                bias_b - bias_a,
                row.position_a,
                row.position_b,
                row.slope_a,
                row.slope_b,
            )


def main() -> None:
    t_start = time.time()
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    models = list(dict.fromkeys(args.model))

    ds: xr.Dataset = load_m5_ds(args.data_dir)
    split = ds["split"].values
    fit_splits, forecast_split, forecast_days = WINDOWS["validation"]
    fit_mask, forecast_mask = np.isin(split, fit_splits), split == forecast_split
    if args.recent_days < 1 or args.recent_days > fit_mask.sum():
        raise SystemExit(f"m5_bias.py: --recent-days must be between 1 and {fit_mask.sum()}, got {args.recent_days}")

    fit_idx = sample_series(ds.sizes["series_id"], args.n_demo, args.seed)
    sales = ds["sales"].values[fit_idx]
    sales_fit, actual = sales[:, fit_mask], sales[:, forecast_mask]
    if not np.isfinite(actual).all():
        raise SystemExit(
            f"m5_bias.py: the dataset has no actuals for {forecast_days}; run m5_fetch.py, then m5_preprocess.py."
        )
    logger.info(
        "Validation origin: fit on %d days, compare with %s; %d series (random sample, seed=%d)",
        fit_mask.sum(),
        forecast_days,
        fit_idx.size,
        args.seed,
    )

    coords = pd.DataFrame({key: ds[key].values[fit_idx] for key in ("series_id", "dept_id", "store_id")})
    windows = window_names(args.recent_days)
    window_days = {windows[0]: f"all {fit_mask.sum()} fit days", windows[1]: f"the last {args.recent_days} fit days"}
    series_tables, group_tables = [], []
    for model in models:
        t0 = time.time()
        forecasts, fitted = MODELS[model].fit_forecast(ds, fit_idx, fit_mask, forecast_mask)
        logger.info("Fitted %s on %d series in %.1fs", model, fit_idx.size, time.time() - t0)
        fitted.pop(INSAMPLE_KEY, None)
        # Per-series scalars the model returns (for ssp_dow: sigma_h, response_norm, final_loss), for follow-up.
        scalars = {key: np.asarray(value) for key, value in fitted.items() if np.ndim(value) == 1}
        stats = assign_groups(pd.concat([coords, series_stats(sales_fit, actual, forecasts, args.recent_days)], axis=1))
        series_tables.append(stats.assign(model=model, **scalars))
        for window in windows:
            table = group_table(stats, window)
            log_group_table(table, model, window, window_days[window])
            group_tables.append(table.assign(model=model, window=window))

    groups = pd.concat(group_tables, ignore_index=True)
    groups = groups[["model", "window", *groups.columns[:-2]]]
    if len(models) > 1:
        log_model_comparison(groups, windows[1])

    out_dir = args.output_dir / f"bias_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    series = pd.concat(series_tables, ignore_index=True)
    series = series[["model", *series.columns.drop("model")]]
    series.to_csv(out_dir / "series.csv", index=False)
    groups.to_csv(out_dir / "groups.csv", index=False)
    logger.info("Wrote series.csv and groups.csv to %s  (total %.1fs)", out_dir, time.time() - t_start)


if __name__ == "__main__":
    main()
