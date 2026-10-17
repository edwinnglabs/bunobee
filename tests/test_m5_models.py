"""Unit tests for the M5 model registry, ``playground/m5_prototype/m5_models.py``.

``m5_run.py --model NAME`` fits ``MODELS[NAME]``, so adding a model needs only a new entry there (issue #103).
These tests hold every entry to the contract the runner relies on, and check that the baseline entry
``ssp_dow`` keeps the model of the #94 submission: weekday dummies from Tuesday to Sunday with no column of
ones, fitted in independent runs of at most ``fit_chunk`` series.

Every test fits a tiny synthetic dataset built in this module, so none needs the raw M5 data. The registry
is a playground script, not part of the package, so it is loaded by path as ``tests/test_m5_score.py``
loads the scorer.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import ModuleType

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

# Calendar: 8 fit weeks, then one 28-day forecast window. Day 0 is a Saturday (Mon=0), as d_1 is in M5.
N_SERIES = 4
N_FIT = 56
HORIZON = 28
N_DATES = N_FIT + HORIZON
FIRST_DOW = 5
FIT_MASK = np.arange(N_DATES) < N_FIT
FORECAST_MASK = ~FIT_MASK


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


class TestRegistry:
    """The registry names the baseline, and every entry's settings can be written to ``run_stats.json``."""

    def test_default_model_is_registered(self):
        assert m5_models.DEFAULT_MODEL == "ssp_dow"
        assert m5_models.DEFAULT_MODEL in m5_models.MODELS

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
            m5_models.SspDow.FITTED_KEYS, N_SERIES
        )
