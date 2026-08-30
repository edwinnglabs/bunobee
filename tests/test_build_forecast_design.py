"""Unit tests for :func:`bunobee.models.ssp.forecast.build_forecast_design`.

One test (class) per acceptance criterion of issue #27:

1. Weekly-dummy continuation: the seasonal block equals
   ``make_peridoic_dummies(n_steps + horizon, 7)[n_steps:]``.
2. The intercept column stays all-ones over the horizon.
3. ``covariates_future`` of the wrong length raises ``ValueError``.
4. The output round-trips through ``forecast_ssp`` (shape-compatible).
5. Exported from ``ssp/__init__.py``; NumPy-style docstring; lines <= 120 chars.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import bunobee.models.ssp as ssp
from bunobee.models.ssp import build_forecast_design, forecast_ssp
from bunobee.models.ssp import forecast as forecast_mod
from bunobee.regression import make_peridoic_dummies

N_STEPS = 30
HORIZON = 14
PERIOD = 7
WEEKLY = {"columns": slice(1, PERIOD), "period": PERIOD, "drop_first": True}


def _weekly_z_train(n_steps: int = N_STEPS, n_series: int | None = None) -> np.ndarray:
    """Intercept + weekly-dummy training design, optionally broadcast over series."""
    dummies = make_peridoic_dummies(n_steps, period=PERIOD, drop_first=True)
    Z = np.concatenate([np.ones((n_steps, 1)), dummies], axis=1).astype(float)
    if n_series is None:
        return Z
    return np.broadcast_to(Z[:, None, :], (n_steps, n_series, Z.shape[1])).copy()


# ---------------------------------------------------------------------------
# Criterion 1 — weekly-dummy continuation matches the long-array slice
# ---------------------------------------------------------------------------


class TestPeriodicContinuation:
    def test_weekly_block_matches_long_array_slice(self):
        Z_train = _weekly_z_train()
        Z_future = build_forecast_design(Z_train, HORIZON, periodic=WEEKLY)

        expected = make_peridoic_dummies(N_STEPS + HORIZON, period=PERIOD, drop_first=True)[N_STEPS:]
        np.testing.assert_allclose(Z_future[:, 0, 1:], expected)

    def test_phase_continues_for_every_offset(self):
        # The dummy phase is anchored at step 0, so every training length must line up.
        for n_steps in range(PERIOD, PERIOD * 4):
            Z_future = build_forecast_design(_weekly_z_train(n_steps), PERIOD, periodic=WEEKLY)
            expected = make_peridoic_dummies(n_steps + PERIOD, period=PERIOD, drop_first=True)[n_steps:]
            np.testing.assert_allclose(Z_future[:, 0, 1:], expected)

    def test_multi_series_shares_the_seasonal_block(self):
        Z_future = build_forecast_design(_weekly_z_train(n_series=3), HORIZON, periodic=WEEKLY)
        assert Z_future.shape == (HORIZON, 3, PERIOD)
        expected = make_peridoic_dummies(N_STEPS + HORIZON, period=PERIOD, drop_first=True)[N_STEPS:]
        for j in range(3):
            np.testing.assert_allclose(Z_future[:, j, 1:], expected)

    def test_drop_first_false_keeps_full_block(self):
        dummies = make_peridoic_dummies(N_STEPS, period=PERIOD, drop_first=False)
        Z_train = np.concatenate([np.ones((N_STEPS, 1)), dummies], axis=1)
        spec = {"columns": slice(1, None), "period": PERIOD, "drop_first": False}
        Z_future = build_forecast_design(Z_train, HORIZON, periodic=spec)

        expected = make_peridoic_dummies(N_STEPS + HORIZON, period=PERIOD, drop_first=False)[N_STEPS:]
        np.testing.assert_allclose(Z_future[:, 0, 1:], expected)

    def test_block_width_mismatch_raises(self):
        Z_train = _weekly_z_train()
        with pytest.raises(ValueError, match="needs 5 columns"):
            build_forecast_design(Z_train, HORIZON, periodic={"columns": slice(1, None), "period": 6})

    def test_wrong_columns_raise(self):
        # Column 0 is the intercept, not part of the seasonal block.
        Z_train = _weekly_z_train()
        with pytest.raises(ValueError, match="do not match periodic dummies"):
            build_forecast_design(Z_train, HORIZON, periodic={"columns": slice(0, PERIOD - 1), "period": PERIOD})

    def test_column_selection_accepts_list_and_mask(self):
        Z_train = _weekly_z_train()
        expected = build_forecast_design(Z_train, HORIZON, periodic=WEEKLY)

        by_list = build_forecast_design(
            Z_train, HORIZON, periodic={"columns": list(range(1, PERIOD)), "period": PERIOD}
        )
        mask = np.zeros(PERIOD, dtype=bool)
        mask[1:] = True
        by_mask = build_forecast_design(Z_train, HORIZON, periodic={"columns": mask, "period": PERIOD})

        np.testing.assert_allclose(by_list, expected)
        np.testing.assert_allclose(by_mask, expected)

    def test_multiple_periodic_blocks(self):
        weekly = make_peridoic_dummies(N_STEPS, period=7, drop_first=True)
        triad = make_peridoic_dummies(N_STEPS, period=3, drop_first=True)
        Z_train = np.concatenate([np.ones((N_STEPS, 1)), weekly, triad], axis=1)
        specs = [
            {"columns": slice(1, 7), "period": 7},
            {"columns": slice(7, 9), "period": 3},
        ]
        Z_future = build_forecast_design(Z_train, HORIZON, periodic=specs)

        np.testing.assert_allclose(
            Z_future[:, 0, 1:7], make_peridoic_dummies(N_STEPS + HORIZON, period=7, drop_first=True)[N_STEPS:]
        )
        np.testing.assert_allclose(
            Z_future[:, 0, 7:9], make_peridoic_dummies(N_STEPS + HORIZON, period=3, drop_first=True)[N_STEPS:]
        )


# ---------------------------------------------------------------------------
# Criterion 2 — constant / intercept columns carry forward
# ---------------------------------------------------------------------------


class TestConstantColumns:
    def test_intercept_stays_all_ones(self):
        Z_future = build_forecast_design(_weekly_z_train(), HORIZON, periodic=WEEKLY)
        np.testing.assert_allclose(Z_future[:, :, 0], np.ones((HORIZON, 1)))

    def test_per_series_constant_carries_its_own_value(self):
        Z_train = _weekly_z_train(n_series=2)
        Z_train[:, 0, 0] = 2.0
        Z_train[:, 1, 0] = 5.0
        Z_future = build_forecast_design(Z_train, HORIZON, periodic=WEEKLY)

        np.testing.assert_allclose(Z_future[:, 0, 0], 2.0)
        np.testing.assert_allclose(Z_future[:, 1, 0], 5.0)

    def test_unresolved_time_varying_column_raises(self):
        Z_train = _weekly_z_train()
        Z_train[:, 0] = np.linspace(0.0, 1.0, N_STEPS)  # a trend, not a constant
        with pytest.raises(ValueError, match="varies over time"):
            build_forecast_design(Z_train, HORIZON, periodic=WEEKLY)


# ---------------------------------------------------------------------------
# Criterion 3 — covariate handling and its ValueError
# ---------------------------------------------------------------------------


class TestCovariatesFuture:
    def _z_train_with_covariate(self, n_series: int | None = None) -> np.ndarray:
        base = _weekly_z_train(n_series=n_series)
        rng = np.random.default_rng(0)
        if n_series is None:
            cov = rng.normal(size=(N_STEPS, 1))
            return np.concatenate([base, cov], axis=1)
        cov = rng.normal(size=(N_STEPS, n_series, 1))
        return np.concatenate([base, cov], axis=2)

    def test_mapping_fills_the_named_column(self):
        Z_train = self._z_train_with_covariate()
        future = np.arange(HORIZON, dtype=float)
        Z_future = build_forecast_design(Z_train, HORIZON, periodic=WEEKLY, covariates_future={PERIOD: future})
        np.testing.assert_allclose(Z_future[:, 0, PERIOD], future)

    def test_trailing_array_fills_the_last_columns(self):
        Z_train = self._z_train_with_covariate()
        future = np.arange(HORIZON, dtype=float)[:, None]
        Z_future = build_forecast_design(Z_train, HORIZON, periodic=WEEKLY, covariates_future=future)
        np.testing.assert_allclose(Z_future[:, 0, PERIOD], future[:, 0])

    def test_per_series_array_is_kept_per_series(self):
        Z_train = self._z_train_with_covariate(n_series=2)
        rng = np.random.default_rng(1)
        future = rng.normal(size=(HORIZON, 2, 1))
        Z_future = build_forecast_design(Z_train, HORIZON, periodic=WEEKLY, covariates_future=future)
        np.testing.assert_allclose(Z_future[:, :, PERIOD], future[:, :, 0])

    def test_wrong_length_mapping_raises(self):
        Z_train = self._z_train_with_covariate()
        bad = np.arange(HORIZON - 1, dtype=float)
        with pytest.raises(ValueError, match="leading dimension"):
            build_forecast_design(Z_train, HORIZON, periodic=WEEKLY, covariates_future={PERIOD: bad})

    def test_wrong_length_array_raises(self):
        Z_train = self._z_train_with_covariate()
        bad = np.arange(HORIZON + 3, dtype=float)[:, None]
        with pytest.raises(ValueError, match="leading dimension"):
            build_forecast_design(Z_train, HORIZON, periodic=WEEKLY, covariates_future=bad)

    def test_wrong_series_width_raises(self):
        Z_train = self._z_train_with_covariate(n_series=2)
        bad = np.zeros((HORIZON, 3, 1))
        with pytest.raises(ValueError, match="must broadcast to"):
            build_forecast_design(Z_train, HORIZON, periodic=WEEKLY, covariates_future=bad)

    def test_scalar_covariate_raises(self):
        Z_train = self._z_train_with_covariate()
        with pytest.raises(ValueError, match="leading dimension"):
            build_forecast_design(Z_train, HORIZON, periodic=WEEKLY, covariates_future={PERIOD: 1.0})

    def test_too_many_covariate_columns_raises(self):
        Z_train = self._z_train_with_covariate()
        bad = np.zeros((HORIZON, Z_train.shape[-1] + 1))
        with pytest.raises(ValueError, match="only"):
            build_forecast_design(Z_train, HORIZON, periodic=WEEKLY, covariates_future=bad)


# ---------------------------------------------------------------------------
# Criterion 4 — round-trips through forecast_ssp
# ---------------------------------------------------------------------------


class TestForecastSspRoundTrip:
    def test_output_feeds_forecast_ssp(self):
        n_series, n_chains, n_draws = 2, 2, 5
        Z_train = _weekly_z_train(n_series=n_series)
        Z_future = build_forecast_design(Z_train, HORIZON, periodic=WEEKLY)

        n_states = Z_train.shape[-1]
        rng = np.random.default_rng(3)
        idata = xr.Dataset(
            data_vars={
                "at": (["chain", "draw", "step", "state"], rng.normal(size=(n_chains, n_draws, N_STEPS, n_states))),
                "sigma_q": (["chain", "draw", "state"], np.full((n_chains, n_draws, n_states), 0.05)),
                "sigma_h": (["chain", "draw", "series"], np.full((n_chains, n_draws, n_series), 0.1)),
            },
            coords={"chain": np.arange(n_chains), "draw": np.arange(n_draws)},
        )

        out = forecast_ssp(idata, Z_future, seed=0)
        assert out["forecast_samples"].dims == ("sample", "time", "series")
        assert out.sizes["time"] == HORIZON
        assert out.sizes["series"] == n_series
        assert out.sizes["sample"] == n_chains * n_draws
        assert np.isfinite(out["forecast_samples"].values).all()

    def test_two_d_input_gains_a_singleton_series_axis(self):
        Z_future = build_forecast_design(_weekly_z_train(), HORIZON, periodic=WEEKLY)
        assert Z_future.shape == (HORIZON, 1, PERIOD)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("horizon", [0, -3])
    def test_non_positive_horizon_raises(self, horizon):
        with pytest.raises(ValueError, match="positive integer"):
            build_forecast_design(_weekly_z_train(), horizon, periodic=WEEKLY)

    def test_bad_ndim_raises(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            build_forecast_design(np.ones((2, 2, 2, 2)), HORIZON)

    def test_bad_periodic_type_raises(self):
        with pytest.raises(TypeError, match="periodic must be"):
            build_forecast_design(_weekly_z_train(), HORIZON, periodic=7)

    def test_no_periodic_carries_constants(self):
        Z_train = np.ones((N_STEPS, 2)) * np.array([1.0, 3.0])
        Z_future = build_forecast_design(Z_train, 4)
        np.testing.assert_allclose(Z_future, np.broadcast_to(np.array([1.0, 3.0]), (4, 1, 2)))


# ---------------------------------------------------------------------------
# Criterion 5 — export, docstring, and line length
# ---------------------------------------------------------------------------


class TestPackaging:
    def test_exported_from_ssp_namespace(self):
        assert ssp.build_forecast_design is build_forecast_design
        assert "build_forecast_design" in ssp.__all__

    def test_numpy_style_docstring(self):
        doc = build_forecast_design.__doc__
        assert doc is not None
        for section in ("Parameters", "Returns", "Z_train :", "horizon :", "periodic :", "covariates_future :"):
            assert section in doc

    def test_source_lines_within_120_chars(self):
        path = Path(forecast_mod.__file__)
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            assert len(line) <= 120, f"{path.name}:{i} is {len(line)} chars"
