"""Model registry for ``m5_run.py``: each entry is a model plus its settings.

``m5_run.py --model NAME`` fits ``MODELS[NAME]`` once per forecast origin. An entry builds its own inputs
from the dataset and calls the model functions in ``bunobee.models.m5``, so the model math stays in
``src/``. Data loading, series sampling, the submission and scoring stay in the runner.

To add a model, write a frozen dataclass with a ``fit_forecast`` method (see :class:`Model`) and register
an instance of it in :data:`MODELS`. Its fields are its settings: ``m5_run.py`` records them in
``run_stats.json``, so two entries of one class with different settings are two models. An SSP entry picks
its target transform (a ``bunobee.models.m5.transforms.Transform``) with its ``transform`` field, so trying a
transform needs only a new entry, not an edit of the fitter: the ``*_boxcox_*`` and ``*_yeojohnson`` entries are
``ssp_dow`` and ``ssp_level_dow`` with a power transform fitted per series (issue #115). It picks the priors of its
noise and drifts the same way, with its ``sigma_h_prior`` and ``sigma_q_prior`` fields (a
``bunobee.models.m5.m5_ssp_optim.SigmaPrior`` each): the ``*_wide``, ``*_halfnormal`` and ``*_lognormal`` entries
swap the default uniform priors, set for the log target, for wider or soft ones (issue #116). It turns the weekday
dummies on or off with its ``dow`` field: ``ssp_level_yeojohnson_halfnormal`` is the default entry without them
(issue #145). It adds annual Fourier terms, events, month and day of month with its ``covariates`` field (a
:class:`CalendarCovariates`): the ``*_calendar`` entries are the default entry and the level-only entry with them
(issue #146). :class:`SspReg` is the other way to use them: the covariates are a static regression ``X beta`` outside
the Kalman filter, with no states of their own, fitted by NUTS, so a design of any width costs the filter the same 7
states.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import jax.numpy as jnp
import numpy as np
import xarray as xr

from bunobee.models.m5.m5_ssp_optim import (
    SIGMA_H_PRIOR,
    SIGMA_Q_PRIOR,
    SigmaPrior,
    fit_batch_series_opt,
    forecast_gaussian_batch_series_opt,
    predict_batch_series_opt,
)
from bunobee.models.m5.m5_ssp_reg import fit_batch_series_reg, predict_batch_series_reg
from bunobee.models.m5.transforms import LogTransform, PowerTransform, Transform
from bunobee.regression import make_fourier_series, make_periodic_dummies

logger = logging.getLogger(__name__)

#: Optional key of a model's ``fitted`` output: its in-sample fit of every fit day, in units, of shape
#: ``(len(series_idx), fit_mask.sum())``. ``m5_run.py`` draws it in ``forecasts.pdf`` instead of saving it.
INSAMPLE_KEY = "insample"

#: Period of the annual Fourier terms, in days.
ANNUAL_PERIOD = 365.25


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


@dataclass(frozen=True)
class GaussianForecast:
    """A fitted SSP's Gaussian forecast and one-step-ahead fit on the transformed scale (issue #158).

    Attributes
    ----------
    mu, var : np.ndarray
        Forecast mean and variance of each series, each of shape ``(n_series, horizon)``.
    fit_mu, fit_var : np.ndarray
        Mean and variance of each fit day predicted from the days before it, each of shape ``(n_series, n_fit_steps)``.
    transform : Transform
        The transform the series were fitted on.
    params : dict of str to np.ndarray
        The fit's per-series outputs, i.e. the transform's parameters among them; what ``transform`` reads to map
        the Gaussian to units.
    """

    mu: np.ndarray
    var: np.ndarray
    fit_mu: np.ndarray
    fit_var: np.ndarray
    transform: Transform
    params: dict[str, Any]

    def quantile(self, u: np.ndarray) -> np.ndarray:
        """Quantiles in units of the forecast, ``(n_series, len(u), horizon)``; see ``Transform.quantile``."""
        return self.transform.quantile(self.mu, self.var, u, self.params)

    def residuals(self, sales: np.ndarray) -> np.ndarray:
        """Standardised one-step-ahead forecast errors on the transformed scale.

        Parameters
        ----------
        sales : np.ndarray
            The fit days' sales, ``(n_series, n_fit_steps)``.

        Returns
        -------
        np.ndarray
            ``(transform.forward(sales) - fit_mu) / sqrt(fit_var)``, same shape.
        """
        z = np.asarray(self.transform.forward(sales, self.params), dtype=np.float64)
        return (z - self.fit_mu) / np.sqrt(self.fit_var)


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
    mu, var = _one_step_ahead_gaussian(result, Z)
    return result["transform"].inverse(mu, var, result)


def _one_step_ahead_gaussian(result: dict, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance, on the transformed scale, of each fit day predicted from the days before it.

    The Gaussian that :func:`_one_step_ahead_fit` maps to units. Its standardised residuals
    ``(transform.forward(sales) - mu) / sqrt(var)`` are the one-step-ahead forecast errors.

    Parameters
    ----------
    result : dict
        Output of ``fit_batch_series_opt`` for B series.
    Z : np.ndarray
        ``(n_steps, n_states)`` design matrix the series were fitted with.

    Returns
    -------
    mu, var : np.ndarray
        Predicted mean and variance of each fit day, each of shape ``(B, n_steps)``.
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
    return mu, var


@dataclass(frozen=True)
class CalendarCovariates:
    """Calendar columns of an SSP design: annual Fourier terms, events, month and day of month.

    Every series shares these columns, so the design matrix ``Z`` holds them for all series, and each series gets its
    own coefficients. :class:`SspDow` fixes those coefficients: they do not drift, and their prior is the filter's
    ``N(0, 1)``, zero-centered. All columns are known in advance over the forecast days, so nothing here looks ahead:
    the same columns are built from the whole calendar at every forecast origin.

    Parameters
    ----------
    annual_order : int
        Harmonics of the annual Fourier terms, which take ``2 * annual_order`` columns. 0 leaves them out.
    annual_period : float
        Period of the annual Fourier terms in days, by default :data:`ANNUAL_PERIOD`.
    events : bool
        Include one 0/1 column per distinct name in ``event_name_1`` and ``event_name_2``, by default True. A day with
        two events is 1 in both columns.
    month : bool
        Include 0/1 columns for February to December, with January dropped, by default True.
    day_of_month : bool
        Include 0/1 columns for days 2 to 31 of the month, with day 1 dropped, by default True.

    Raises
    ------
    ValueError
        If ``annual_order`` is negative or no column is left.
    """

    annual_order: int
    annual_period: float = ANNUAL_PERIOD
    events: bool = True
    month: bool = True
    day_of_month: bool = True

    def __post_init__(self) -> None:
        if self.annual_order < 0:
            raise ValueError(f"annual_order must be non-negative, got {self.annual_order}")
        if not (self.annual_order or self.events or self.month or self.day_of_month):
            raise ValueError("CalendarCovariates needs at least one group of columns.")

    def design(self, ds: xr.Dataset) -> np.ndarray:
        """Build the columns over every day of the dataset, in the order of the fields.

        Parameters
        ----------
        ds : xr.Dataset
            The preprocessed M5 dataset.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(n_dates, n_covariates)``.
        """
        n_dates = ds.sizes["date"]
        blocks = []
        if self.annual_order:
            # Step 0 is the first day of the calendar, so the terms carry on over the forecast days of every origin.
            blocks.append(make_fourier_series(n_dates, period=self.annual_period, order=self.annual_order))
        if self.events:
            blocks.append(_event_columns(ds))
        if self.month:
            blocks.append(np.eye(12)[ds["month_of_year"].values.astype(int) - 1][:, 1:])
        if self.day_of_month:
            blocks.append(np.eye(31)[ds["day_of_month"].values.astype(int) - 1][:, 1:])
        return np.concatenate(blocks, axis=1).astype(np.float32)


def _event_columns(ds: xr.Dataset) -> np.ndarray:
    """One 0/1 column per distinct event name of the calendar, in sorted order of the names.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with ``event_name_1`` and ``event_name_2`` along ``date``. A day without an event holds ``""``, as
        ``m5_ds.nc`` does, or NaN, as a dataset fresh from ``m5_preprocess.build_m5_dataset`` does.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(n_dates, n_events)``.
    """
    first, second = (ds[name].values.astype(str) for name in ("event_name_1", "event_name_2"))
    names = np.setdiff1d(np.concatenate([first, second]), ["", "nan"])
    return np.stack([(first == name) | (second == name) for name in names], axis=1)


@dataclass(frozen=True)
class SspDow:
    """Weekday-dummy SSP fitted by MAP: ``fit_batch_series_opt`` with ``predict_batch_series_opt``.

    Each series is fitted to ``transform`` of its sales, by default the log of its sales over their mean.

    The design matrix has one column per weekday from Tuesday to Sunday, with Monday dropped. Without
    ``level`` there is no column of ones, so Monday rows are all zero and every Monday forecast is the
    series' fit-window mean (see ``WORKFLOW.md``, "Known problems"). With ``level`` a column of ones leads
    the matrix: its state is the local level, which ``fit_batch_series_opt`` gives its own drift, and the
    weekday states are offsets from Monday that share the other drift. Without ``dow`` the weekday columns are
    left out, so the level is the only state and a series' forecast is the same on every forecast day. With
    ``covariates`` their columns close the matrix, each a state with no drift, so ``fit_batch_series_opt`` updates
    it by the filter alone. The series' forecast then moves with them over the forecast days, with or without ``dow``.

    Parameters
    ----------
    level : bool
        Lead the design matrix with a column of ones, the level state, by default False.
    dow : bool
        Include the weekday columns, by default True. ``dow=False`` needs ``level=True``, as there would be no
        states otherwise.
    transform : Transform
        Target transform of ``fit_batch_series_opt``, by default ``LogTransform()``: each series is fitted to
        ``log(max(sales, 0.1) / mean)`` and its forecasts are inverted as lognormal means.
    sigma_h_prior : SigmaPrior
        Prior of the noise ``sigma_h``, by default ``SIGMA_H_PRIOR``, U(0.1, 0.5), set for the log target.
    sigma_q_prior : SigmaPrior
        Prior of each drift, by default ``SIGMA_Q_PRIOR``, U(0.01, 0.1), set for the log target.
    covariates : CalendarCovariates | None
        Calendar columns to add to the design matrix, by default None.
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

    Raises
    ------
    ValueError
        If both ``level`` and ``dow`` are False.
    """

    level: bool = False
    dow: bool = True
    transform: Transform = LogTransform()
    sigma_h_prior: SigmaPrior = SIGMA_H_PRIOR
    sigma_q_prior: SigmaPrior = SIGMA_Q_PRIOR
    covariates: CalendarCovariates | None = None
    n_iter: int = 100
    lr: float = 3e-2
    chunk_size: int | None = 4096
    fit_chunk: int | None = None

    #: Outputs of ``fit_batch_series_opt`` not returned as ``fitted``: the filtered states ``at`` and ``Pt``
    #: (n_series x n_steps x n_states each), only needed to forecast, and the inputs shared by every series.
    #: The rest are per series (``sigma_h``, ``sigma_q``, ``final_loss`` and the transform's parameters, e.g.
    #: ``response_norm``), so the parameters of any transform are returned.
    DROPPED_KEYS = ("at", "Pt", "Z", "a0", "P0", "transform")

    def __post_init__(self) -> None:
        if not (self.level or self.dow):
            raise ValueError("SspDow needs level=True or dow=True: with neither, the design matrix has no columns.")

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
        one-step-ahead fit of every fit day (see :func:`_one_step_ahead_fit`). Its ``sigma_q`` holds the drifting
        states only: the level and the weekdays, whatever the ``covariates``.
        """
        forecasts, fitted = [], {}
        for result, Z, Z_future in self._fit_runs(ds, series_idx, fit_mask, forecast_mask):
            forecasts.append(predict_batch_series_opt(result, Z_future=Z_future))
            for key, value in result.items():
                if key not in self.DROPPED_KEYS:
                    fitted.setdefault(key, []).append(np.asarray(value))
            fitted.setdefault(INSAMPLE_KEY, []).append(_one_step_ahead_fit(result, Z))
        return np.concatenate(forecasts), {key: np.concatenate(parts) for key, parts in fitted.items()}

    def fit_forecast_gaussian(
        self,
        ds: xr.Dataset,
        series_idx: np.ndarray,
        fit_mask: np.ndarray,
        forecast_mask: np.ndarray,
    ) -> GaussianForecast:
        """Fit as :meth:`fit_forecast` does, but return the Gaussian forecast on the transformed scale.

        The forecast quantiles in units are ``transform.quantile`` of ``mu`` and ``var`` (issue #158); the same
        fit and the same Gaussian give the point forecast of :meth:`fit_forecast`.

        See :meth:`Model.fit_forecast` for the parameters.

        Returns
        -------
        GaussianForecast
            The forecast and one-step-ahead fit of every series, in ``series_idx`` order.
        """
        parts: dict[str, list[np.ndarray]] = {key: [] for key in ("mu", "var", "fit_mu", "fit_var")}
        params: dict[str, list[np.ndarray]] = {}
        transform = self.transform
        for result, Z, Z_future in self._fit_runs(ds, series_idx, fit_mask, forecast_mask):
            mu, var = forecast_gaussian_batch_series_opt(result, Z_future)
            fit_mu, fit_var = _one_step_ahead_gaussian(result, Z)
            for key, value in zip(parts, (mu, var, fit_mu, fit_var)):
                parts[key].append(np.asarray(value))
            transform = result["transform"]
            for key, value in result.items():
                if key not in self.DROPPED_KEYS:
                    params.setdefault(key, []).append(np.asarray(value))
        return GaussianForecast(
            **{key: np.concatenate(value) for key, value in parts.items()},
            transform=transform,
            params={key: np.concatenate(value) for key, value in params.items()},
        )

    def _fit_runs(
        self,
        ds: xr.Dataset,
        series_idx: np.ndarray,
        fit_mask: np.ndarray,
        forecast_mask: np.ndarray,
    ) -> Iterator[tuple[dict, jnp.ndarray, jnp.ndarray]]:
        """Fit the series in runs of at most ``fit_chunk`` and yield each run's result and design matrices.

        Yields
        ------
        result : dict
            Output of ``fit_batch_series_opt`` for the run's series, its ``sigma_q`` cut to the drifting states.
        Z, Z_future : jnp.ndarray
            ``(n_fit_steps, n_states)`` and ``(horizon, n_states)`` design matrices, shared by every run.
        """
        weekday = ds["day_of_week"].values
        if self.dow:
            # The calendar is contiguous daily dates, so phasing step 0 at the first day's weekday (Mon=0) makes
            # each dummy column a weekday from Tuesday to Sunday on every day, with Monday dropped.
            design = make_periodic_dummies(weekday.size, period=7, phase=int(weekday[0]), intercept=self.level)
        else:
            design = np.ones((weekday.size, 1), dtype=np.float32)  # __post_init__ ensures this is the level
        fixed_states = None
        n_drifting = design.shape[1]  # 6 states, 7 with the level, or the level alone
        if self.covariates is not None:
            columns = self.covariates.design(ds)
            design = np.concatenate([design, columns], axis=1)
            fixed_states = np.arange(design.shape[1]) >= n_drifting
        Z = jnp.asarray(design[fit_mask])  # (n_fit_steps, n_states)
        Z_future = jnp.asarray(design[forecast_mask])  # (horizon, n_states)
        sales = ds["sales"].values[series_idx][:, fit_mask]  # (n_series, n_fit_steps)

        n_series = sales.shape[0]
        step = self.fit_chunk or n_series
        n_runs = math.ceil(n_series / step)
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
                sigma_h_prior=self.sigma_h_prior,
                sigma_q_prior=self.sigma_q_prior,
                fixed_states=fixed_states,
            )
            result["sigma_q"] = result["sigma_q"][:, :n_drifting]
            yield result, Z, Z_future


#: Sigma priors for the standardised power scale (issue #116), where the default priors, set for log sales, hold
#: ``sigma_h`` at 0.5 and the drifts at 0.1. Fitted with no bound, the noise sits at 0.16-1.0 there, the level's
#: drift at 0.004-0.33 and the weekdays' drift at 0.0005-0.04. ``WIDE_PRIORS`` are uniform priors wide enough for all
#: of them, and ``HALF_NORMAL_PRIORS`` and ``LOG_NORMAL_PRIORS`` soft priors, with no bound, centred near them.
WIDE_PRIORS = {
    "sigma_h_prior": SigmaPrior("uniform", (0.05, 1.5)),
    "sigma_q_prior": SigmaPrior("uniform", (0.0005, 0.5)),
}
HALF_NORMAL_PRIORS = {
    "sigma_h_prior": SigmaPrior("half_normal", (1.0,)),
    "sigma_q_prior": SigmaPrior("half_normal", (0.05,)),
}
LOG_NORMAL_PRIORS = {
    "sigma_h_prior": SigmaPrior("log_normal", (math.log(0.8), 0.5)),
    "sigma_q_prior": SigmaPrior("log_normal", (math.log(0.02), 1.0)),
}

#: The stage 1 covariates of #143 (issue #146). ``annual_order`` is the Fourier order chosen on the 3,000-series sample
#: (see ``WORKFLOW.md``, "Annual and calendar covariates (#146)"). A design this wide needs ``fit_chunk`` to bound the
#: memory of the backward pass, which grows with the number of states.
CALENDAR_COVARIATES = CalendarCovariates(annual_order=1)
CALENDAR_FIT_CHUNK = 1024


@dataclass(frozen=True)
class SspReg:
    """Level + weekday SSP with a static regression on shared columns, fitted by NUTS.

    The model of a series, on the scale of ``transform``, is ``level_t + dow_t · alpha_t + x_t · beta``. The level and
    the six weekday offsets are the filter's seven drifting states, as in ``SspDow(level=True)``. The covariates are
    not states: ``x_t`` is a row of ``covariates.design(ds)``, the same for every series, and ``beta``, with the prior
    ``N(0, tau^2)`` on each coefficient, is a parameter of the series with no time dimension (see
    :mod:`bunobee.models.m5.m5_ssp_reg`). The forecast is the mean over posterior draws of the back-transformed
    forecast, so it is a posterior mean where ``SspDow`` forecasts from one MAP point.

    Parameters
    ----------
    transform : Transform
        Target transform, by default Yeo-Johnson, fitted once per series and held fixed.
    sigma_h_prior : SigmaPrior
        Prior of the noise ``sigma_h``, by default the half-normal prior of the #118 default entry.
    sigma_q_prior : SigmaPrior
        Prior of each of the two drifts, by default the half-normal prior of the #118 default entry.
    covariates : CalendarCovariates | None
        Calendar columns of the regression, by default None: no regression.
    tau : float
        Prior scale of each coefficient, by default 1.0 (loose). 0.1 is tight.
    num_warmup, num_samples, num_chains : int
        NUTS warmup and draws per chain, and chains, by default 200, 200 and 4.
    n_forecast_draws : int
        Posterior draws the forecast averages, by default 100.
    fit_chunk : int | None
        Max series per NUTS run, by default 10. The series of one run share a step size, so a run adapts to the
        hardest of them, and a smaller chunk fits each series more closely at the cost of more runs. ``None`` = one
        run over all series.
    seed : int
        PRNG seed of every run, by default 0.
    """

    transform: Transform = PowerTransform("yeo-johnson")
    sigma_h_prior: SigmaPrior = HALF_NORMAL_PRIORS["sigma_h_prior"]
    sigma_q_prior: SigmaPrior = HALF_NORMAL_PRIORS["sigma_q_prior"]
    covariates: CalendarCovariates | None = None
    tau: float = 1.0
    num_warmup: int = 200
    num_samples: int = 200
    num_chains: int = 4
    n_forecast_draws: int = 100
    fit_chunk: int | None = 10
    seed: int = 0

    def fit_forecast(
        self,
        ds: xr.Dataset,
        series_idx: np.ndarray,
        fit_mask: np.ndarray,
        forecast_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Fit and forecast the series in independent NUTS runs of at most ``fit_chunk`` series.

        See :meth:`Model.fit_forecast` for the parameters and returns. ``fitted`` holds, per series, the posterior
        means of ``sigma_h``, ``sigma_q_level``, ``sigma_q_dow`` and, with covariates, ``beta`` (series × covariates),
        the transform's parameters, and the worst R-hat (``r_hat_max``, NaN with one chain) and the smallest effective
        sample size (``n_eff_min``) over the series' parameters. It has no in-sample fit, so ``forecasts.pdf`` draws
        none.
        """
        weekday = ds["day_of_week"].values
        # The calendar is contiguous daily dates, so phasing step 0 at the first day's weekday (Mon=0) makes each
        # dummy column a weekday from Tuesday to Sunday on every day, with Monday dropped.
        Z = make_periodic_dummies(weekday.size, period=7, phase=int(weekday[0]), intercept=True)
        X = None if self.covariates is None else self.covariates.design(ds)
        sales = ds["sales"].values[series_idx][:, fit_mask]  # (n_series, n_fit_steps)

        n_series = sales.shape[0]
        step = self.fit_chunk or n_series
        n_runs = math.ceil(n_series / step)
        forecasts = []
        fitted: dict[str, list[np.ndarray]] = {}
        for run, start in enumerate(range(0, n_series, step), start=1):
            end = min(start + step, n_series)
            if n_runs > 1:
                logger.info("Fit run %d/%d: series %d-%d", run, n_runs, start, end - 1)
            result = fit_batch_series_reg(
                sales[start:end],
                Z[fit_mask],
                None if X is None else X[fit_mask],
                tau=self.tau,
                transform=self.transform,
                sigma_h_prior=self.sigma_h_prior,
                sigma_q_prior=self.sigma_q_prior,
                num_warmup=self.num_warmup,
                num_samples=self.num_samples,
                num_chains=self.num_chains,
                seed=self.seed,
            )
            forecasts.append(
                predict_batch_series_reg(
                    result,
                    Z[forecast_mask],
                    None if X is None else X[forecast_mask],
                    n_draws=self.n_forecast_draws,
                )
            )
            samples = result["samples"]
            outputs = {site: samples[site].mean(axis=0) for site in samples}  # (B,) or (B, n_covariates)
            outputs |= result["transform_params"]
            outputs |= {"r_hat_max": result["r_hat_max"], "n_eff_min": result["n_eff_min"]}
            for key, value in outputs.items():
                fitted.setdefault(key, []).append(np.asarray(value))
        return np.concatenate(forecasts), {key: np.concatenate(parts) for key, parts in fitted.items()}


#: The regression entries' covariates (see ``WORKFLOW.md``): the event columns alone, and the annual Fourier pair with
#: the events. Both leave out month and day of month, which the stage 1 covariates of #146 add.
EVENT_COVARIATES = CalendarCovariates(annual_order=0, month=False, day_of_month=False)
ANNUAL_EVENT_COVARIATES = CalendarCovariates(annual_order=1, month=False, day_of_month=False)

#: Registered models by name, the choices of ``m5_run.py --model``. The power-transform entries fit Box-Cox of
#: ``sales + 0.5`` (``boxcox_shift``), Box-Cox of ``max(sales, 0.1)`` (``boxcox_floor``) and Yeo-Johnson, which on
#: sales is Box-Cox of ``sales + 1`` (``yeojohnson``), each without and with the level state. The ``_wide``,
#: ``_halfnormal`` and ``_lognormal`` entries swap the default sigma priors for those above.
#: ``ssp_level_yeojohnson_halfnormal`` is the default entry without its weekday dummies: the day-of-week ablation.
#: The ``_calendar`` entries are the default and level-only entries with ``CALENDAR_COVARIATES``.
MODELS: dict[str, Model] = {
    "ssp_dow": SspDow(),
    "ssp_level_dow": SspDow(level=True),
    "ssp_dow_boxcox_shift": SspDow(transform=PowerTransform("box-cox", shift=0.5)),
    "ssp_dow_boxcox_floor": SspDow(transform=PowerTransform("box-cox", floor=0.1)),
    "ssp_dow_yeojohnson": SspDow(transform=PowerTransform("yeo-johnson")),
    "ssp_level_dow_boxcox_shift": SspDow(level=True, transform=PowerTransform("box-cox", shift=0.5)),
    "ssp_level_dow_boxcox_floor": SspDow(level=True, transform=PowerTransform("box-cox", floor=0.1)),
    "ssp_level_dow_yeojohnson": SspDow(level=True, transform=PowerTransform("yeo-johnson")),
    "ssp_level_dow_yeojohnson_wide": SspDow(level=True, transform=PowerTransform("yeo-johnson"), **WIDE_PRIORS),
    "ssp_level_dow_yeojohnson_halfnormal": SspDow(
        level=True, transform=PowerTransform("yeo-johnson"), **HALF_NORMAL_PRIORS
    ),
    "ssp_level_dow_yeojohnson_lognormal": SspDow(
        level=True, transform=PowerTransform("yeo-johnson"), **LOG_NORMAL_PRIORS
    ),
    "ssp_level_dow_halfnormal": SspDow(level=True, **HALF_NORMAL_PRIORS),
    "ssp_level_yeojohnson_halfnormal": SspDow(
        level=True, dow=False, transform=PowerTransform("yeo-johnson"), **HALF_NORMAL_PRIORS
    ),
    "ssp_level_dow_yeojohnson_halfnormal_calendar": SspDow(
        level=True,
        transform=PowerTransform("yeo-johnson"),
        covariates=CALENDAR_COVARIATES,
        fit_chunk=CALENDAR_FIT_CHUNK,
        **HALF_NORMAL_PRIORS,
    ),
    "ssp_level_yeojohnson_halfnormal_calendar": SspDow(
        level=True,
        dow=False,
        transform=PowerTransform("yeo-johnson"),
        covariates=CALENDAR_COVARIATES,
        fit_chunk=CALENDAR_FIT_CHUNK,
        **HALF_NORMAL_PRIORS,
    ),
    # The NUTS regression entries: the default model's states and priors, a static ``X beta`` outside the filter.
    "ssp_reg": SspReg(),
    "ssp_reg_events": SspReg(covariates=EVENT_COVARIATES, tau=1.0),
    "ssp_reg_annual_events_tight": SspReg(covariates=ANNUAL_EVENT_COVARIATES, tau=0.1),
}

#: ``m5_run.py``'s default ``--model``: the best entry on the full run at all three origins and on Kaggle (issue
#: #118). ``ssp_dow`` stays registered as the baseline.
DEFAULT_MODEL = "ssp_level_dow_yeojohnson_halfnormal"
