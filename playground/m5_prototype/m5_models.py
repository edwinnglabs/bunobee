"""Model registry for ``m5_run.py``: each entry is a model plus its settings.

``m5_run.py --model NAME`` fits ``MODELS[NAME]`` once per forecast origin. An entry builds its own inputs
from the dataset and calls the model functions in ``bunobee.models.m5``, so the model math stays in
``src/``. Data loading, series sampling, the submission and scoring stay in the runner.

To add a model, write a frozen dataclass with a ``fit_forecast`` method (see :class:`Model`) and register
an instance of it in :data:`MODELS`. Its fields are its settings: ``m5_run.py`` records them in
``run_stats.json``, so two entries of one class with different settings are two models. An SSP entry picks
its target transform (a ``bunobee.models.m5.transforms.Transform``) with its ``transform`` field, so trying a
transform needs only a new entry, not an edit of the fitter.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Protocol

import jax.numpy as jnp
import numpy as np
import xarray as xr

from bunobee.models.m5.m5_ssp_optim import fit_batch_series_opt, predict_batch_series_opt
from bunobee.models.m5.transforms import LogTransform, Transform
from bunobee.regression import make_periodic_dummies

logger = logging.getLogger(__name__)

#: Optional key of a model's ``fitted`` output: its in-sample fit of every fit day, in units, of shape
#: ``(len(series_idx), fit_mask.sum())``. ``m5_run.py`` draws it in ``forecasts.pdf`` instead of saving it.
INSAMPLE_KEY = "insample"


class Model(Protocol):
    """What ``m5_run.py`` needs from a registered model.

    Entries are dataclasses; their fields are the settings ``m5_run.py`` records in ``run_stats.json``.
    """

    def fit_forecast(
        self,
        ds: xr.Dataset,
        series_idx: np.ndarray,
        fit_mask: np.ndarray,
        forecast_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Fit the selected series on the fit days and forecast the forecast days.

        Parameters
        ----------
        ds : xr.Dataset
            The preprocessed M5 dataset (``m5_common.load_m5_ds``): every series over every day.
        series_idx : np.ndarray
            Positions along ``series_id`` of the series to fit, in output order.
        fit_mask : np.ndarray
            Boolean mask along ``date`` of the days to fit on.
        forecast_mask : np.ndarray
            Boolean mask along ``date`` of the days to forecast. They follow the fit days.

        Returns
        -------
        forecasts : np.ndarray
            Point forecasts of shape ``(len(series_idx), forecast_mask.sum())``, in ``series_idx`` order,
            with no NaNs.
        fitted : dict of str to np.ndarray
            Per-series outputs to save, each with leading dimension ``len(series_idx)``. ``m5_run.py``
            writes them to ``batch_params.npz`` and plots their distributions in ``diagnostics.png``, except
            the optional :data:`INSAMPLE_KEY` entry, the in-sample fit, which it draws in ``forecasts.pdf``.
        """
        ...


def _one_step_ahead_fit(result: dict, Z: np.ndarray) -> np.ndarray:
    """In-sample fit of a ``fit_batch_series_opt`` result: each fit day predicted from the days before it.

    Day ``t`` is predicted from the filtered state of day ``t - 1`` (``a0`` and ``P0`` on the first day), the
    states being random walks, and back-transformed as ``predict_batch_series_opt`` back-transforms a
    forecast: by the fit's ``transform.inverse`` of ``mu`` and ``var``, the predicted mean and variance of the
    transformed sales. For the log transform that is ``exp(mu + var / 2) * response_norm``. The forecast of
    the first day after the fit days is this prediction one day on.

    Parameters
    ----------
    result : dict
        Output of ``fit_batch_series_opt`` for B series.
    Z : np.ndarray
        ``(n_steps, n_states)`` design matrix the series were fitted with.

    Returns
    -------
    np.ndarray
        In-sample fit of shape ``(B, n_steps)``, in units.
    """
    Z = np.asarray(Z)
    at, Pt = result["at"], result["Pt"]  # (B, n_steps, n_states) each
    mu = np.empty(at.shape[:2], dtype=at.dtype)
    var = np.empty_like(mu)
    mu[:, 0] = np.asarray(result["a0"]) @ Z[0]
    var[:, 0] = np.asarray(result["P0"]) @ Z[0] ** 2
    # einsum, not a broadcast product, so no (B, n_steps, n_states) temporary is built.
    mu[:, 1:] = np.einsum("bts,ts->bt", at[:, :-1], Z[1:])
    var[:, 1:] = np.einsum("bts,ts->bt", Pt[:, :-1], Z[1:] ** 2)
    var += np.asarray(result["sigma_h"])[:, None] ** 2
    return result["transform"].inverse(mu, var, result)


@dataclass(frozen=True)
class SspDow:
    """Weekday-dummy SSP fitted by MAP: ``fit_batch_series_opt`` with ``predict_batch_series_opt``.

    Each series is fitted to ``transform`` of its sales, by default the log of its sales over their mean.

    The design matrix has one column per weekday from Tuesday to Sunday, with Monday dropped. Without
    ``level`` there is no column of ones, so Monday rows are all zero and every Monday forecast is the
    series' fit-window mean (see ``WORKFLOW.md``, "Known problems"). With ``level`` a column of ones leads
    the matrix: its state is the local level, which ``fit_batch_series_opt`` gives its own drift, and the
    weekday states are offsets from Monday that share the other drift.

    Parameters
    ----------
    level : bool
        Lead the design matrix with a column of ones, the level state, by default False.
    transform : Transform
        Target transform of ``fit_batch_series_opt``, by default ``LogTransform()``: each series is fitted to
        ``log(max(sales, 0.1) / mean)`` and its forecasts are inverted as lognormal means.
    n_iter : int
        Adam optimisation steps per fit run, by default 100.
    lr : float
        Adam learning rate, by default 3e-2.
    chunk_size : int | None
        Max series per Kalman-filter ``vmap`` call within a run. ``None`` = no chunking. All chunks share
        one gradient pass, so this does not cap peak memory while fitting; ``fit_chunk`` does. Default 4096.
    fit_chunk : int | None
        Max series per fit run, so peak memory scales with ``fit_chunk`` rather than with the number of
        series. ``None`` = one run over all series. Default None.
    """

    level: bool = False
    transform: Transform = LogTransform()
    n_iter: int = 100
    lr: float = 3e-2
    chunk_size: int | None = 4096
    fit_chunk: int | None = None

    #: Outputs of ``fit_batch_series_opt`` not returned as ``fitted``: the filtered states ``at`` and ``Pt``
    #: (n_series x n_steps x n_states each), only needed to forecast, and the inputs shared by every series.
    #: The rest are per series (``sigma_h``, ``sigma_q``, ``final_loss`` and the transform's parameters, e.g.
    #: ``response_norm``), so the parameters of any transform are returned.
    DROPPED_KEYS = ("at", "Pt", "Z", "a0", "P0", "transform")

    def fit_forecast(
        self,
        ds: xr.Dataset,
        series_idx: np.ndarray,
        fit_mask: np.ndarray,
        forecast_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Fit and forecast the series in independent runs of at most ``fit_chunk`` series.

        ``fit_batch_series_opt`` chunks only its Kalman filter, inside one jitted ``value_and_grad``, so the
        backward pass keeps every chunk's intermediates alive at once. Calling it once per ``fit_chunk``
        series bounds that memory instead. The series are independent — the loss is a sum over series and
        Adam updates each parameter on its own — so ``fit_chunk`` changes the forecasts only by float32
        rounding: a different batch shape vectorises differently, and 100 Adam steps can amplify that to a
        relative difference of up to ~4e-4 on a few series.

        See :meth:`Model.fit_forecast` for the parameters and returns. ``fitted`` holds the outputs of
        ``fit_batch_series_opt`` but the :attr:`DROPPED_KEYS` and, under :data:`INSAMPLE_KEY`, the
        one-step-ahead fit of every fit day (see :func:`_one_step_ahead_fit`).
        """
        # The calendar is contiguous daily dates, so phasing step 0 at the first day's weekday (Mon=0) makes
        # each dummy column a weekday from Tuesday to Sunday on every day, with Monday dropped.
        dow = ds["day_of_week"].values
        weekly = make_periodic_dummies(dow.size, period=7, phase=int(dow[0]), intercept=self.level)
        Z = jnp.asarray(weekly[fit_mask])  # (n_fit_steps, n_states): 6 states, or 7 with the level
        Z_future = jnp.asarray(weekly[forecast_mask])  # (horizon, n_states)
        sales = ds["sales"].values[series_idx][:, fit_mask]  # (n_series, n_fit_steps)

        n_series = sales.shape[0]
        step = self.fit_chunk or n_series
        n_runs = math.ceil(n_series / step)
        fitted: dict[str, list[np.ndarray]] = {}
        forecasts = []
        for run, start in enumerate(range(0, n_series, step), start=1):
            end = min(start + step, n_series)
            if n_runs > 1:
                logger.info("Fit run %d/%d: series %d-%d", run, n_runs, start, end - 1)
            result = fit_batch_series_opt(
                sales[start:end],
                n_iter=self.n_iter,
                lr=self.lr,
                Z=Z,
                chunk_size=self.chunk_size,
                log_every=100,
                show_progress=True,
                transform=self.transform,
            )
            forecasts.append(predict_batch_series_opt(result, Z_future=Z_future))
            for key, value in result.items():
                if key not in self.DROPPED_KEYS:
                    fitted.setdefault(key, []).append(np.asarray(value))
            fitted.setdefault(INSAMPLE_KEY, []).append(_one_step_ahead_fit(result, Z))
        return np.concatenate(forecasts), {key: np.concatenate(parts) for key, parts in fitted.items()}


#: Registered models by name, the choices of ``m5_run.py --model``.
MODELS: dict[str, Model] = {
    "ssp_dow": SspDow(),
    "ssp_level_dow": SspDow(level=True),
}

#: The baseline entry, ``m5_run.py``'s default ``--model``.
DEFAULT_MODEL = "ssp_dow"
