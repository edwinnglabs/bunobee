#!/usr/bin/env python3
"""Score an SSP model's predictive quantiles on the M5 Uncertainty subset by WSPL (issue #158).

The SSP models give a Gaussian forecast on the transformed scale. This script turns it into the 9 quantiles of the
M5 Uncertainty competition at the 12 aggregation levels of the 3-item x 10-store benchmark subset (``m5_subset.py``),
fits the registered model at one origin (a few seconds for the default) and scores the quantiles by WSPL against the
cheap baseline of ``m5u_baseline.py``. It runs on the subset only: there is no full run and nothing is submitted.

**Bottom series (level 12).** The transform is monotone, so the ``u`` quantile of a series' forecast in units is the
transform's inverse of the Gaussian's ``u`` quantile, ``Transform.quantile``: the exact quantiles of the distribution
whose mean the point forecast is, floored at 0 (Yeo-Johnson of sales is never negative).

**Aggregate levels (1-11).** The quantiles of a sum are not the sum of the quantiles, and the series are fitted
independently, so the models give no joint distribution. The quantiles of a group are taken from sample paths of its
bottom series, summed, and the paths differ only in how the series depend on each other (``--dependence``):

``independent``
    Draws of the series are independent. The sum's spread is the root of the sum of the variances: too narrow when
    the series move together, as they do with the calendar and the weather.
``copula``
    A Gaussian copula: the series' Gaussian forecast errors are correlated, with the correlation of their in-sample
    one-step-ahead errors (the rank correlation of the last :data:`CORR_DAYS` fit days, as normal scores), the same
    on every forecast day. The marginals stay the exact ones of the bottom level. This is the method chosen.
``comonotone``
    Perfect dependence: every series at the same quantile, so a group's quantile is the sum of its series'. The
    widest the spread can be, and the upper bound the other two sit under.

Only the cross-series dependence matters, not the dependence over days, because the pinball loss scores each day on
its own.

    python m5u_ssp.py                       # the default model, pre-validation window, the three dependences
    python m5u_ssp.py --model ssp_dow       # another registered SspDow entry
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import special, stats

from m5_common import DATA_DIR, load_m5_ds
from m5_models import DEFAULT_MODEL, MODELS, GaussianForecast, SspDow
from m5_score_submission import WINDOWS, window_masks
from m5u_baseline import baseline_forecasts
from m5u_score import LEVELS, QUANTILES, Hierarchy, aggregate, build_hierarchy, compute_hierarchical_wspl, window_data

logger = logging.getLogger(__name__)

#: Dependences between the series for the aggregate levels; see the module docstring.
DEPENDENCES = ("independent", "copula", "comonotone")

#: Recent fit days whose one-step-ahead errors the copula's correlation is estimated from: 52 weeks.
CORR_DAYS = 364

#: Sample paths drawn for the aggregate levels. 0.005 of 10,000 is 50 paths on the tail.
N_DRAWS = 10_000

#: The level of the bottom series, whose quantiles are exact.
BOTTOM_LEVEL = len(LEVELS) - 1


def fit_gaussian(
    model: SspDow, ds: xr.Dataset, rows: np.ndarray, window: str
) -> tuple[GaussianForecast, np.ndarray, np.ndarray]:
    """Fit ``model`` on the bottom series ``rows`` at the origin of ``window``.

    Parameters
    ----------
    model : SspDow
        A registered MAP-fitted SSP entry.
    ds : xr.Dataset
        Dataset from ``m5_preprocess.build_m5_dataset``.
    rows : np.ndarray
        Positions along ``series_id`` of the bottom series.
    window : {"prevalidation", "validation", "evaluation"}
        Scoring window, a key of ``WINDOWS``; the model is fitted on the days before it.

    Returns
    -------
    gaussian : GaussianForecast
        The Gaussian forecast of each series.
    sales : np.ndarray
        The series' fit-day sales, ``(len(rows), n_fit_steps)``.
    fit_mask : np.ndarray
        The fit days along ``date``.
    """
    fit_mask, forecast_mask = window_masks(ds["split"].values, window)
    gaussian = model.fit_forecast_gaussian(ds, rows, fit_mask, forecast_mask)
    return gaussian, ds["sales"].values[rows][:, fit_mask], fit_mask


def copula_correlation(residuals: np.ndarray, n_days: int = CORR_DAYS) -> np.ndarray:
    """Correlation of the series' one-step-ahead errors, as the correlation of their normal scores.

    The errors of an SSP fitted to sales have heavy tails and a spike on the days a series does not sell, so each
    series' last ``n_days`` errors are ranked and mapped to normal scores before they are correlated, the rank
    correlation of a Gaussian copula.

    Parameters
    ----------
    residuals : np.ndarray, shape (n_series, n_fit_steps)
        Standardised one-step-ahead errors, e.g. ``GaussianForecast.residuals``.
    n_days : int, optional
        Number of last days to use. Default :data:`CORR_DAYS`.

    Returns
    -------
    np.ndarray, shape (n_series, n_series)
        A correlation matrix: symmetric, unit diagonal, positive semi-definite.
    """
    recent = np.asarray(residuals, dtype=np.float64)[:, -n_days:]
    scores = special.ndtri((stats.rankdata(recent, axis=1) - 0.5) / recent.shape[1])
    corr = np.corrcoef(scores)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return _nearest_psd_correlation(corr)


def _nearest_psd_correlation(corr: np.ndarray) -> np.ndarray:
    """Clip a symmetric matrix's negative eigenvalues to 0 and rescale it to unit diagonal."""
    values, vectors = np.linalg.eigh((corr + corr.T) / 2)
    clipped = (vectors * np.maximum(values, 0.0)) @ vectors.T
    scale = np.sqrt(np.diag(clipped))
    return clipped / np.outer(scale, scale)


def dependence_matrix(dependence: str, n_series: int, residuals: np.ndarray | None = None) -> np.ndarray:
    """The correlation of the series' Gaussian forecast errors under one of :data:`DEPENDENCES`.

    Parameters
    ----------
    dependence : {"independent", "copula", "comonotone"}
        The dependence.
    n_series : int
        Number of bottom series.
    residuals : np.ndarray, optional
        In-sample one-step-ahead errors, needed for ``"copula"``.

    Returns
    -------
    np.ndarray, shape (n_series, n_series)
        Identity for ``independent``, all ones for ``comonotone``, :func:`copula_correlation` for ``copula``.
    """
    if dependence == "independent":
        return np.eye(n_series)
    if dependence == "comonotone":
        return np.ones((n_series, n_series))
    if dependence == "copula":
        if residuals is None:
            raise ValueError("the copula dependence needs the in-sample residuals")
        return copula_correlation(residuals)
    raise ValueError(f"dependence must be one of {DEPENDENCES}, got {dependence!r}")


def joint_quantiles(
    gaussian: GaussianForecast,
    hierarchy: Hierarchy,
    corr: np.ndarray,
    quantiles: tuple[float, ...] = QUANTILES,
    n_draws: int = N_DRAWS,
    seed: int = 2026,
) -> list[np.ndarray]:
    """Forecast quantiles of every group of the 12 levels from sample paths of the bottom series.

    A path draws each series' Gaussian forecast error with correlation ``corr`` between the series, maps it to units
    with the transform, and sums the series of each group. A group's quantiles are the empirical quantiles of its
    paths. The bottom level is not sampled: its quantiles are the exact ones of ``gaussian.quantile``.

    Parameters
    ----------
    gaussian : GaussianForecast
        Gaussian forecast of the hierarchy's bottom series, in ``hierarchy.rows`` order.
    hierarchy : Hierarchy
        Groups of the bottom series, from ``m5u_score.build_hierarchy``.
    corr : np.ndarray, shape (n_series, n_series)
        Correlation of the series' forecast errors, from :func:`dependence_matrix`.
    quantiles : tuple of float, optional
        Quantile levels. Default :data:`QUANTILES`.
    n_draws : int, optional
        Sample paths. Default :data:`N_DRAWS`.
    seed : int, optional
        Seed of the draws. Default 2026.

    Returns
    -------
    list of np.ndarray
        Per level, shape ``(n_groups, len(quantiles), N_STEPS)`` in ``hierarchy.ids`` order, as
        ``compute_hierarchical_wspl`` takes them.

    Raises
    ------
    ValueError
        If a group of the bottom level has not exactly one series.
    """
    n_series, horizon = gaussian.mu.shape
    values, vectors = np.linalg.eigh(corr)
    factor = vectors * np.sqrt(np.maximum(values, 0.0))  # factor @ factor.T == corr
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_draws, horizon, n_series))
    error = np.einsum("bk,shk->sbh", factor, noise)  # (n_draws, n_series, horizon), correlation ``corr``
    z = gaussian.mu + np.sqrt(gaussian.var) * error
    paths = gaussian.transform.to_units(z, gaussian.params).astype(np.float64)
    paths = paths.transpose(1, 0, 2)  # (n_series, n_draws, horizon)

    bottom = gaussian.quantile(np.asarray(quantiles))
    forecasts = []
    for level, (group_idx, n_groups) in enumerate(zip(hierarchy.group_idx, hierarchy.n_groups, strict=True)):
        if level == BOTTOM_LEVEL:
            if n_groups != n_series:
                raise ValueError(f"the bottom level has {n_groups} groups for {n_series} series")
            ordered = np.empty_like(bottom, dtype=np.float64)
            ordered[group_idx] = bottom
            forecasts.append(ordered)
            continue
        q = np.quantile(aggregate(paths, group_idx, n_groups), quantiles, axis=1)  # (n_quantiles, n_groups, horizon)
        forecasts.append(np.moveaxis(q, 0, 1))
    return forecasts


def coverage(
    ds: xr.Dataset,
    hierarchy: Hierarchy,
    forecasts: list[np.ndarray],
    window: str,
    quantiles: tuple[float, ...] = QUANTILES,
) -> pd.DataFrame:
    """Share of the actuals at or below each forecast quantile, over the groups and days of a level.

    Parameters
    ----------
    ds, hierarchy, window
        As ``m5u_score.window_data``.
    forecasts : list of np.ndarray
        Per level, ``(n_groups, len(quantiles), N_STEPS)``.
    quantiles : tuple of float, optional
        Quantile levels. Default :data:`QUANTILES`.

    Returns
    -------
    pd.DataFrame
        One row per level, one column per quantile level; calibrated forecasts have the level in its column.
    """
    _, y_score, _ = window_data(ds, hierarchy, window)
    rows = {
        level_no: (y[:, None, :] <= fq).mean(axis=(0, 2)) for (level_no, _, _), y, fq in zip(LEVELS, y_score, forecasts)
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=[f"{u:g}" for u in quantiles]).rename_axis("level")


def compare_methods(
    ds: xr.Dataset,
    hierarchy: Hierarchy,
    forecasts: dict[str, list[np.ndarray]],
    window: str,
) -> pd.DataFrame:
    """Per-level WSPL of each set of forecasts, with the 12-level mean as the last row.

    Parameters
    ----------
    ds, hierarchy, window
        As ``m5u_score.compute_hierarchical_wspl``.
    forecasts : dict of str to list of np.ndarray
        Per method name, its per-level quantile forecasts.

    Returns
    -------
    pd.DataFrame
        Indexed by level number and name, one column per method; the last row, ``overall``, is the mean.
    """
    columns = {
        name: compute_hierarchical_wspl(ds, fq, hierarchy, window=window)["wspl"] for name, fq in forecasts.items()
    }
    table = pd.DataFrame(columns)
    table.index = [f"L{n} {name}" for n, name, _ in LEVELS]
    table.loc["overall"] = table.mean()
    return table


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS), help="Registered SspDow entry to fit.")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR, metavar="DIR", help="Directory containing m5_ds.nc.")
    p.add_argument("--windows", nargs="+", choices=sorted(WINDOWS), default=["prevalidation"], help="Windows to score.")
    p.add_argument("--dependence", nargs="+", choices=DEPENDENCES, default=list(DEPENDENCES), help="Dependences.")
    p.add_argument("--n-draws", type=int, default=N_DRAWS, help="Sample paths for the aggregate levels.")
    p.add_argument("--seed", type=int, default=2026, help="Seed of the sample paths.")
    p.add_argument("--output", type=Path, default=None, metavar="CSV", help="Also write the WSPL table of each window.")
    return p.parse_args()


def main() -> None:
    """Fit the model on the subset and log the per-level WSPL of each dependence against the baseline."""
    from m5_subset import subset_series_idx  # here: it is only this run's dependency

    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model = MODELS[args.model]
    if not isinstance(model, SspDow):
        raise SystemExit(f"{args.model} is not an SspDow entry: only the MAP-fitted ones have a Gaussian forecast.")

    ds = load_m5_ds(args.data_dir)
    hierarchy = build_hierarchy(ds, subset_series_idx(ds))
    logger.info("%d bottom series, %d series over 12 levels", len(hierarchy.rows), sum(hierarchy.n_groups))
    tables = []
    for window in args.windows:
        gaussian, sales, _ = fit_gaussian(model, ds, hierarchy.rows, window)
        residuals = gaussian.residuals(sales)
        forecasts = {"baseline": baseline_forecasts(ds, hierarchy, window)}
        for dependence in args.dependence:
            corr = dependence_matrix(dependence, len(hierarchy.rows), residuals)
            if dependence == "copula":
                off_diagonal = corr[np.triu_indices_from(corr, k=1)]
                logger.info(
                    "Copula correlation of the %d series: mean %.3f, min %.3f, max %.3f",
                    len(corr),
                    off_diagonal.mean(),
                    off_diagonal.min(),
                    off_diagonal.max(),
                )
            forecasts[dependence] = joint_quantiles(gaussian, hierarchy, corr, n_draws=args.n_draws, seed=args.seed)
        table = compare_methods(ds, hierarchy, forecasts, window)
        logger.info("WSPL on the %s window (%s):\n%s", window, WINDOWS[window].days, table.round(5).to_string())
        for dependence in args.dependence:
            cover = coverage(ds, hierarchy, forecasts[dependence], window)
            logger.info("Coverage, %s:\n%s", dependence, cover.loc[[1, 12]].round(3).to_string())
        tables.append(table.assign(window=window))
    if args.output is not None:
        pd.concat(tables).rename_axis("level").to_csv(args.output)
        logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
