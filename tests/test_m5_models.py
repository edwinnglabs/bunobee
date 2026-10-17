"""Unit tests for the M5 model registry, ``playground/m5_prototype/m5_models.py``.

``m5_run.py --model NAME`` fits ``MODELS[NAME]``, so adding a model needs only a new entry there (issue #103).
These tests hold every entry to the contract the runner relies on, and check that the baseline entry
``ssp_dow`` keeps the model of the #94 submission: weekday dummies from Tuesday to Sunday with no column of
ones, fitted in independent runs of at most ``fit_chunk`` series. ``ssp_level_dow`` (issue #106) adds the
column of ones, so its Monday forecast is the fitted level rather than the series mean. Each model's
in-sample fit, drawn in ``forecasts.pdf`` (issue #105), is checked against the Kalman filter's own
one-step-ahead prediction. An entry picks its target transform with its ``transform`` field (issue #114),
so a transform the fitter has never seen is tried without editing it.

Every test fits a tiny synthetic dataset built in this module, so none needs the raw M5 data. The registry
is a playground script, not part of the package, so it is loaded by path as ``tests/test_m5_score.py``
loads the scorer.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from types import ModuleType

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

pytest.importorskip("optax", reason="needs the optional 'm5' extra: pip install 'bunobee[m5]'")

M5_DIR = Path(__file__).resolve().parents[1] / "playground" / "m5_prototype"


def _load_script(name: str) -> ModuleType:
    """Import a playground M5 script by path and register it under its own name.

    Parameters
    ----------
    name : str
        Module name, i.e. the file stem under ``playground/m5_prototype/``.

    Returns
    -------
    ModuleType
        The imported module.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, M5_DIR / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover - only on a broken checkout
        raise ImportError(f"cannot load {name} from {M5_DIR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


m5_models = _load_script("m5_models")

from bunobee.models.m5.transforms import LogTransform  # noqa: E402
from bunobee.models.ssp.kalman_1d import kalman_filter_1d  # noqa: E402
from bunobee.regression import make_periodic_dummies  # noqa: E402

# Calendar: 8 fit weeks, then one 28-day forecast window. Day 0 is a Saturday (Mon=0), as d_1 is in M5.
N_SERIES = 4
N_FIT = 56
HORIZON = 28
N_DATES = N_FIT + HORIZON
FIRST_DOW = 5
FIT_MASK = np.arange(N_DATES) < N_FIT
FORECAST_MASK = ~FIT_MASK


@dataclass(frozen=True)
class CentredLog1p:
    """``log1p(sales)`` less its mean over the fit days: a transform the fitter has never seen, with its own parameter.

    The inverse is the mean of ``expm1`` of a Gaussian, ``expm1(mu + centre + var / 2)``.
    """

    def fit(self, sales: np.ndarray) -> dict[str, np.ndarray]:
        return {"centre": np.log1p(sales).mean(axis=-1).astype(np.float32)}

    def forward(self, sales: np.ndarray, params) -> np.ndarray:
        return (np.log1p(sales) - np.asarray(params["centre"])[..., None]).astype(np.float32)

    def inverse(self, mu: np.ndarray, var: np.ndarray, params) -> np.ndarray:
        return np.expm1(mu + np.asarray(params["centre"])[..., None] + 0.5 * var)


@pytest.fixture(scope="module")
def ds() -> xr.Dataset:
    """Synthetic M5-shaped dataset: Poisson sales with a weekly pattern, one level per series."""
    rng = np.random.default_rng(0)
    dow = (FIRST_DOW + np.arange(N_DATES)) % 7
    weekday_factor = np.array([0.8, 0.9, 1.0, 1.0, 1.1, 1.4, 1.3])[dow]
    level = np.array([0.5, 2.0, 5.0, 12.0])[:, None]
    sales = rng.poisson(level * weekday_factor).astype(np.float32)
    return xr.Dataset(
        {
            "sales": (("series_id", "date"), sales),
            "day_of_week": ("date", dow.astype(np.int8)),
        },
        coords={"series_id": [f"S_{i}" for i in range(N_SERIES)]},
    )


@pytest.fixture(scope="module")
def ssp_dow_fit(ds) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """``SspDow`` forecasts and fitted arrays of every synthetic series, in one fit run."""
    return m5_models.SspDow(n_iter=20).fit_forecast(ds, np.arange(N_SERIES), FIT_MASK, FORECAST_MASK)


@pytest.fixture(scope="module")
def ssp_level_dow_fit(ds) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The registered ``ssp_level_dow`` entry's forecasts and fitted arrays of every synthetic series."""
    model = replace(m5_models.MODELS["ssp_level_dow"], n_iter=20)
    return model.fit_forecast(ds, np.arange(N_SERIES), FIT_MASK, FORECAST_MASK)


@pytest.fixture(scope="module")
def centred_log1p_fit(ds) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The ``ssp_dow`` entry with the ``CentredLog1p`` transform: forecasts and fitted arrays of every series."""
    model = replace(m5_models.MODELS["ssp_dow"], n_iter=20, transform=CentredLog1p())
    return model.fit_forecast(ds, np.arange(N_SERIES), FIT_MASK, FORECAST_MASK)


def _refilter(ds: xr.Dataset, fitted: dict[str, np.ndarray], level: bool, transform=LogTransform()) -> list[tuple]:
    """Rerun each synthetic series' Kalman filter on its fit days with the series' fitted parameters.

    Parameters
    ----------
    ds : xr.Dataset
        The synthetic dataset.
    fitted : dict of str to np.ndarray
        ``fitted`` output of an ``SspDow`` fit of every synthetic series, in series order.
    level : bool
        Whether the fit's design matrix was led by a column of ones (``SspDow.level``).
    transform : Transform
        The fit's target transform (``SspDow.transform``), by default the log transform.

    Returns
    -------
    list of tuple
        Per series, its transformed sales ``y`` followed by the six outputs of ``kalman_filter_1d``:
        ``(y, log_p, at, Pt, vt, Ft, Kt)``.
    """
    Z = jnp.asarray(make_periodic_dummies(N_DATES, period=7, phase=FIRST_DOW, intercept=level)[FIT_MASK])
    sales = ds["sales"].values[:, FIT_MASK]
    runs = []
    for i in range(N_SERIES):
        y = transform.forward(sales[i], {key: value[i] for key, value in fitted.items()})
        outputs = kalman_filter_1d(
            a0=jnp.zeros(Z.shape[1]),
            P0=jnp.ones(Z.shape[1]),
            Z=Z,
            sigma_h=fitted["sigma_h"][i],
            sigma_q=fitted["sigma_q"][i],
            y=jnp.asarray(y),
        )
        runs.append((y, *outputs))
    return runs


class TestRegistry:
    """The registry names the baseline, and every entry's settings can be written to ``run_stats.json``."""

    def test_default_model_is_registered(self):
        assert m5_models.DEFAULT_MODEL == "ssp_dow"
        assert m5_models.DEFAULT_MODEL in m5_models.MODELS

    @pytest.mark.parametrize("name", ["ssp_dow", "ssp_level_dow"])
    def test_ssp_entries_keep_the_log_transform(self, name):
        assert m5_models.MODELS[name].transform == LogTransform()

    @pytest.mark.parametrize("name", list(m5_models.MODELS))
    def test_settings_are_json_serialisable(self, name):
        model = m5_models.MODELS[name]
        assert is_dataclass(model)
        json.dumps(asdict(model))


class TestContract:
    """Every entry returns forecasts and fitted arrays of the shapes ``m5_run.py`` checks."""

    @pytest.mark.parametrize("name", list(m5_models.MODELS))
    def test_fit_forecast_shapes(self, ds, name):
        series_idx = np.array([3, 0, 2])
        forecasts, fitted = m5_models.MODELS[name].fit_forecast(ds, series_idx, FIT_MASK, FORECAST_MASK)
        assert forecasts.shape == (series_idx.size, HORIZON)
        assert np.isfinite(forecasts).all()
        for key, value in fitted.items():
            assert np.shape(value)[0] == series_idx.size, key
        if m5_models.INSAMPLE_KEY in fitted:
            assert fitted[m5_models.INSAMPLE_KEY].shape == (series_idx.size, N_FIT)
            assert np.isfinite(fitted[m5_models.INSAMPLE_KEY]).all()


class TestSspDow:
    """``ssp_dow`` is the #94 model: Monday is the dropped weekday, there is no level, and series are independent."""

    def test_monday_forecast_is_the_series_mean(self, ds, ssp_dow_fit):
        # With no column of ones, Monday rows of the design are all zero, so a Monday forecast is the series'
        # fit-window mean times the lognormal correction exp(sigma_h^2 / 2), whatever the filtered states are.
        forecasts, fitted = ssp_dow_fit
        monday = ds["day_of_week"].values[FORECAST_MASK] == 0
        expected = fitted["response_norm"][:, None] * np.exp(fitted["sigma_h"][:, None] ** 2 / 2)
        np.testing.assert_allclose(forecasts[:, monday], np.broadcast_to(expected, (N_SERIES, monday.sum())), rtol=1e-6)
        assert not np.allclose(forecasts[:, ~monday], expected)

    def test_fit_chunk_and_series_order_change_only_rounding(self, ds, ssp_dow_fit):
        forecasts, _ = ssp_dow_fit
        order = np.array([2, 3, 0, 1])
        chunked, fitted = m5_models.SspDow(n_iter=20, fit_chunk=3).fit_forecast(ds, order, FIT_MASK, FORECAST_MASK)
        np.testing.assert_allclose(chunked, forecasts[order], rtol=1e-3)
        assert {key: value.shape[0] for key, value in fitted.items()} == dict.fromkeys(
            ("sigma_h", "sigma_q", "response_norm", "final_loss", m5_models.INSAMPLE_KEY), N_SERIES
        )
        np.testing.assert_allclose(
            fitted[m5_models.INSAMPLE_KEY], ssp_dow_fit[1][m5_models.INSAMPLE_KEY][order], rtol=1e-3
        )


class TestSspLevelDow:
    """``ssp_level_dow`` leads the weekday dummies with a level state, so Monday sales update the states."""

    def test_level_state_has_its_own_drift(self, ssp_level_dow_fit):
        # Seven states, the level first: fit_batch_series_opt gives the first state its own drift and the six
        # weekday states a shared one.
        _, fitted = ssp_level_dow_fit
        assert m5_models.MODELS["ssp_level_dow"].level
        assert fitted["sigma_q"].shape == (N_SERIES, 7)
        np.testing.assert_array_equal(fitted["sigma_q"][:, 1:], fitted["sigma_q"][:, [1]].repeat(6, axis=1))

    def test_monday_forecast_is_the_fitted_level(self, ds, ssp_level_dow_fit):
        # A Monday row of the design is the column of ones alone, so a Monday forecast is the last filtered
        # level back-transformed, exp(a + (P + sigma_h^2) / 2) * mean, not ssp_dow's mean * exp(sigma_h^2 / 2).
        forecasts, fitted = ssp_level_dow_fit
        monday = ds["day_of_week"].values[FORECAST_MASK] == 0
        runs = _refilter(ds, fitted, level=True)
        a_last = np.array([np.asarray(at)[-1, 0] for _, _, at, _, _, _, _ in runs])
        P_last = np.array([np.asarray(Pt)[-1, 0] for _, _, _, Pt, _, _, _ in runs])
        level = np.exp(a_last + 0.5 * (P_last + fitted["sigma_h"] ** 2)) * fitted["response_norm"]
        np.testing.assert_allclose(
            forecasts[:, monday], np.broadcast_to(level[:, None], (N_SERIES, monday.sum())), rtol=1e-4
        )
        series_mean = fitted["response_norm"] * np.exp(fitted["sigma_h"] ** 2 / 2)
        assert not np.allclose(forecasts[:, monday], series_mean[:, None], rtol=1e-2)


class TestInsample:
    """Each SSP entry's in-sample fit is the Kalman filter's one-step-ahead prediction of every fit day."""

    @pytest.mark.parametrize(("fit", "level"), [("ssp_dow_fit", False), ("ssp_level_dow_fit", True)])
    def test_insample_fit_is_the_filters_one_step_ahead_prediction(self, request, ds, fit, level):
        # Rerun the filter with each series' fitted parameters: its innovation v_t = y_t - yhat_t and variance
        # F_t give the one-step-ahead prediction of the log sales, back-transformed as the forecast is.
        _, fitted = request.getfixturevalue(fit)
        for i, (y, _, _, _, vt, Ft, _) in enumerate(_refilter(ds, fitted, level)):
            expected = np.exp(y - np.asarray(vt)[:, 0] + 0.5 * np.asarray(Ft)[:, 0]) * fitted["response_norm"][i]
            np.testing.assert_allclose(fitted[m5_models.INSAMPLE_KEY][i], expected, rtol=1e-4)


class TestTransform:
    """A registry entry picks its target transform with ``SspDow.transform``; the fitter is not edited."""

    def test_settings_record_the_transform(self):
        model = replace(m5_models.MODELS["ssp_dow"], transform=LogTransform(floor=1.0))
        assert json.loads(json.dumps(asdict(model)))["transform"] == {"floor": 1.0}

    def test_fitted_holds_the_transforms_parameters(self, centred_log1p_fit):
        _, fitted = centred_log1p_fit
        assert "centre" in fitted and "response_norm" not in fitted
        assert fitted["centre"].shape == (N_SERIES,)

    def test_monday_forecast_is_the_transforms_inverse(self, ds, centred_log1p_fit):
        # ssp_dow's Monday rows of the design are all zero, so mu = 0 and var = sigma_h^2: the Monday forecast
        # is CentredLog1p's inverse of them, expm1(centre + sigma_h^2 / 2), not the log transform's.
        forecasts, fitted = centred_log1p_fit
        monday = ds["day_of_week"].values[FORECAST_MASK] == 0
        expected = np.expm1(fitted["centre"] + 0.5 * fitted["sigma_h"] ** 2)
        np.testing.assert_allclose(
            forecasts[:, monday], np.broadcast_to(expected[:, None], (N_SERIES, monday.sum())), rtol=1e-6
        )

    def test_insample_fit_is_inverted_by_the_transform(self, ds, centred_log1p_fit):
        _, fitted = centred_log1p_fit
        for i, (y, _, _, _, vt, Ft, _) in enumerate(_refilter(ds, fitted, level=False, transform=CentredLog1p())):
            expected = np.expm1(y - np.asarray(vt)[:, 0] + fitted["centre"][i] + 0.5 * np.asarray(Ft)[:, 0])
            np.testing.assert_allclose(fitted[m5_models.INSAMPLE_KEY][i], expected, rtol=1e-4)
