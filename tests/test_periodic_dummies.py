"""Unit tests for :func:`bunobee.regression.make_periodic_dummies`.

One test (class) per acceptance criterion of issue #102 that the helper owns:

1. The default output encodes cycle position ``t % period`` with step 0 at position 0.
2. ``phase`` sets the cycle position of step 0.
3. ``intercept=True`` prepends a column of ones.
4. The deprecated ``make_peridoic_dummies`` alias returns the same output and warns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bunobee.regression import make_peridoic_dummies, make_periodic_dummies

N_STEPS = 30
PERIOD = 7


# ---------------------------------------------------------------------------
# Criterion 1 — default layout
# ---------------------------------------------------------------------------


class TestDefaultLayout:
    @pytest.mark.parametrize("drop_first", [True, False])
    def test_shape_and_dtype(self, drop_first):
        dummies = make_periodic_dummies(N_STEPS, PERIOD, drop_first=drop_first)
        assert dummies.shape == (N_STEPS, PERIOD - int(drop_first))
        assert dummies.dtype == np.float32

    def test_full_block_is_one_hot_of_step_mod_period(self):
        dummies = make_periodic_dummies(N_STEPS, PERIOD, drop_first=False)
        np.testing.assert_array_equal(dummies, np.eye(PERIOD)[np.arange(N_STEPS) % PERIOD])

    def test_drop_first_zeroes_position_zero(self):
        dummies = make_periodic_dummies(N_STEPS, PERIOD, drop_first=True)
        positions = np.arange(N_STEPS) % PERIOD
        np.testing.assert_array_equal(dummies.sum(axis=1), (positions != 0).astype(np.float32))
        np.testing.assert_array_equal(dummies, make_periodic_dummies(N_STEPS, PERIOD, drop_first=False)[:, 1:])

    def test_zero_steps(self):
        assert make_periodic_dummies(0, PERIOD).shape == (0, PERIOD - 1)


# ---------------------------------------------------------------------------
# Criterion 2 — phase
# ---------------------------------------------------------------------------


class TestPhase:
    @pytest.mark.parametrize("drop_first", [True, False])
    @pytest.mark.parametrize("phase", range(PERIOD))
    def test_phase_shifts_the_cycle(self, phase, drop_first):
        shifted = make_periodic_dummies(N_STEPS, PERIOD, drop_first=drop_first, phase=phase)
        expected = make_periodic_dummies(N_STEPS + phase, PERIOD, drop_first=drop_first)[phase:]
        np.testing.assert_array_equal(shifted, expected)

    @pytest.mark.parametrize("phase", [-1, PERIOD + 2, 3 * PERIOD])
    def test_phase_wraps_modulo_period(self, phase):
        np.testing.assert_array_equal(
            make_periodic_dummies(N_STEPS, PERIOD, phase=phase),
            make_periodic_dummies(N_STEPS, PERIOD, phase=phase % PERIOD),
        )

    def test_long_array_slice_keeps_the_phase(self):
        # The future block of a phased training window is the tail of one longer phased array.
        n_train, horizon, phase = 23, 9, 4
        long = make_periodic_dummies(n_train + horizon, PERIOD, phase=phase)
        np.testing.assert_array_equal(long[n_train:], make_periodic_dummies(horizon, PERIOD, phase=phase + n_train))

    def test_first_weekday_phase_reproduces_calendar_dummies(self):
        # M5 starts on a Saturday: phasing by the first day's weekday gives Tuesday..Sunday dummies.
        dates = pd.date_range("2011-01-29", periods=60, freq="D")
        weekday = dates.dayofweek.to_numpy()  # Mon=0, Sun=6
        calendar = pd.get_dummies(weekday, dtype=np.float32).reindex(columns=range(PERIOD), fill_value=0.0)
        dummies = make_periodic_dummies(len(dates), PERIOD, phase=int(weekday[0]))
        np.testing.assert_array_equal(dummies, calendar.iloc[:, 1:].to_numpy())


# ---------------------------------------------------------------------------
# Criterion 3 — intercept
# ---------------------------------------------------------------------------


class TestIntercept:
    @pytest.mark.parametrize("drop_first", [True, False])
    def test_intercept_prepends_ones(self, drop_first):
        with_intercept = make_periodic_dummies(N_STEPS, PERIOD, drop_first=drop_first, phase=3, intercept=True)
        without = make_periodic_dummies(N_STEPS, PERIOD, drop_first=drop_first, phase=3)

        assert with_intercept.shape == (N_STEPS, 1 + PERIOD - int(drop_first))
        assert with_intercept.dtype == np.float32
        np.testing.assert_array_equal(with_intercept[:, 0], np.ones(N_STEPS, dtype=np.float32))
        np.testing.assert_array_equal(with_intercept[:, 1:], without)

    def test_intercept_defaults_off(self):
        assert make_periodic_dummies(N_STEPS, PERIOD).shape == (N_STEPS, PERIOD - 1)


# ---------------------------------------------------------------------------
# Criterion 4 — deprecated alias
# ---------------------------------------------------------------------------


class TestDeprecatedAlias:
    @pytest.mark.parametrize("drop_first", [True, False])
    @pytest.mark.parametrize("period", [3, PERIOD])
    def test_alias_matches_and_warns(self, period, drop_first):
        with pytest.warns(DeprecationWarning, match="make_periodic_dummies"):
            legacy = make_peridoic_dummies(N_STEPS, period, drop_first=drop_first)

        expected = make_periodic_dummies(N_STEPS, period, drop_first=drop_first)
        assert legacy.dtype == expected.dtype
        np.testing.assert_array_equal(legacy, expected)

    def test_alias_docstring_names_its_replacement(self):
        doc = make_peridoic_dummies.__doc__
        assert ".. deprecated::" in doc
        assert "make_periodic_dummies" in doc
