"""Unit tests for the local M5 WRMSSE scorer, ``playground/m5_prototype/m5_score_submission.py``.

One test class per acceptance criterion of issue #91:

1. The RMSSE scale ignores each series' leading zeros (the M5 Competitors Guide starts it at the first
   non-zero sale).
2. A series with partially missing prices keeps the dollar sales of its priced days as its weight.
3. The scorer accepts a validation (d_1914-d_1941) or an evaluation (d_1942-d_1969) window.

Criterion 4 — running in CI without the raw M5 data — holds because every test here scores a tiny
synthetic hierarchy built in this module. The scorer is a playground script, not part of the package,
so it is loaded by path as ``tests/test_m5_accuracy.py`` loads the fixture builder.

The synthetic hierarchy is one state, one category and one department, with two items in two stores:
four bottom series that each sell nothing before day :data:`LAUNCH` and then alternate 1, 0, 1, 0, ...
Every aggregate of ``n`` of them is ``n`` times that pattern, so its trimmed scale is ``n**2`` and its
RMSSE is ``|sum of its children's errors| / n``. Only series 0 is forecast off (by 1 on every day), so
each level's WRMSSE is ``1 / n`` times the dollar weight of the size-``n`` group holding series 0 — which
is what makes every expected value below computable by hand.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import xarray as xr

M5_DIR = Path(__file__).resolve().parents[1] / "playground" / "m5_prototype"


def _load_script(name: str) -> ModuleType:
    """Import a playground M5 script by path and register it under its own name.

    Registering matters for ``m5_common``: the scorer imports it by name, which only resolves from
    ``tests/`` once it is in ``sys.modules``.

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


_load_script("m5_common")
scorer = _load_script("m5_score_submission")

# Calendar: 40 train days, then one 28-day validation and one 28-day test window, as in the M5 splits.
N_TRAIN = 40
N_STEPS = 28
N_DATES = N_TRAIN + 2 * N_STEPS
VALIDATION_DAYS = slice(N_TRAIN, N_TRAIN + N_STEPS)
EVALUATION_DAYS = slice(N_TRAIN + N_STEPS, N_DATES)

#: First day with a sale (odd, so the 1, 0 pattern starts on a 1). Over the 40 train days the full
#: history has 39 one-step differences of which 33 are non-zero, so a scale that kept the leading zeros
#: would be 33/39 of the trimmed one.
LAUNCH = 7

#: Added to the test-window actuals, so scoring the wrong window's actuals cannot pass by accident.
TEST_SHIFT = 5.0

SERIES_IDS = ["FOODS_1_001_CA_1", "FOODS_1_001_CA_2", "FOODS_1_002_CA_1", "FOODS_1_002_CA_2"]
N_SERIES = len(SERIES_IDS)
HIERARCHY = {
    "item_id": ["FOODS_1_001", "FOODS_1_001", "FOODS_1_002", "FOODS_1_002"],
    "dept_id": ["FOODS_1"] * N_SERIES,
    "cat_id": ["FOODS"] * N_SERIES,
    "store_id": ["CA_1", "CA_2", "CA_1", "CA_2"],
    "state_id": ["CA"] * N_SERIES,
}

#: Forecast error per bottom series, constant over the window: only series 0 is off.
OFFSETS = np.array([1.0, 0.0, 0.0, 0.0])

#: Series 0 sells at 4.0 but has no listed price over the first half of the validation weighting window
#: (train days 12-39). The other three sell at 1.0 throughout.
PRICE_0 = 4.0
PRICE_GAP = slice(N_TRAIN - N_STEPS, N_TRAIN - N_STEPS // 2)

#: Bottom series in the group holding series 0, per level. With one state, category and department,
#: levels 1, 2 and 4-7 are a single group of all four; store CA_1 (levels 3, 8, 9) and item 001 (levels
#: 10, 11) pair series 0 with one other series; level 12 is series 0 alone.
GROUP_SIZE_OF_SERIES_0 = {1: 4, 2: 4, 3: 2, 4: 4, 5: 4, 6: 4, 7: 4, 8: 2, 9: 2, 10: 2, 11: 2, 12: 1}

# Validation: weights come from train days 12-39, on 14 of which every series sells one unit. Series 0
# is priced on only the 7 of them after the gap -> 7 x 4.0 = $28; the others $14 each; $70 in all.
#   group of 4 : RMSSE 1/4 x weight 70/70 = 0.25
#   pair       : RMSSE 1/2 x weight 42/70 = 0.3
#   alone      : RMSSE 1   x weight 28/70 = 0.4
EXPECTED_VALIDATION = {level: {4: 0.25, 2: 0.3, 1: 0.4}[n] for level, n in GROUP_SIZE_OF_SERIES_0.items()}
#: Mean of the 12 levels: (6 x 0.25 + 5 x 0.3 + 0.4) / 12.
EXPECTED_VALIDATION_WRMSSE = 17 / 60

# Evaluation: weights come from the 28 validation days, all priced -> series 0 14 x 4.0 = $56, the others
# $14 each, $98 in all. The scale now runs over train + validation, still 1 per unit of the pattern.
#   group of 4 : RMSSE 1/4 x weight 98/98 = 0.25
#   pair       : RMSSE 1/2 x weight 70/98 = 5/14
#   alone      : RMSSE 1   x weight 56/98 = 4/7
EXPECTED_EVALUATION = {level: {4: 0.25, 2: 5 / 14, 1: 4 / 7}[n] for level, n in GROUP_SIZE_OF_SERIES_0.items()}
#: Mean of the 12 levels: (6 x 0.25 + 5 x 5/14 + 4/7) / 12.
EXPECTED_EVALUATION_WRMSSE = 9 / 28


def _make_ds(test_actuals: bool = True) -> xr.Dataset:
    """Build the synthetic four-series hierarchy in the layout of ``m5_preprocess.build_m5_dataset``.

    Parameters
    ----------
    test_actuals : bool, optional
        If False, the test window's sales are NaN, as in a dataset built from the Kaggle files alone.
        Default True.

    Returns
    -------
    xr.Dataset
        ``sales`` and ``sell_price`` on ``(series_id, date)``, the hierarchy keys on ``series_id`` and
        ``split`` on ``date``.
    """
    t = np.arange(N_DATES)
    sales = np.tile(np.where(t >= LAUNCH, t % 2, 0).astype(float), (N_SERIES, 1))
    if test_actuals:
        sales[:, EVALUATION_DAYS] += TEST_SHIFT
    else:
        sales[:, EVALUATION_DAYS] = np.nan

    prices = np.ones((N_SERIES, N_DATES))
    prices[0] = PRICE_0
    prices[0, PRICE_GAP] = np.nan

    split = np.array(["train"] * N_TRAIN + ["validation"] * N_STEPS + ["test"] * N_STEPS)
    return xr.Dataset(
        {"sales": (("series_id", "date"), sales), "sell_price": (("series_id", "date"), prices)},
        coords={
            "series_id": SERIES_IDS,
            "date": t,
            **{key: ("series_id", values) for key, values in HIERARCHY.items()},
            "split": ("date", split),
        },
    )


def _score(ds: xr.Dataset, days: slice, series_order: list[str] = SERIES_IDS, **kwargs) -> dict[int, float]:
    """Score a forecast that is the actuals over ``days`` plus :data:`OFFSETS`.

    Parameters
    ----------
    ds : xr.Dataset
        Output of :func:`_make_ds`.
    days : slice
        Day positions of the window the forecast covers.
    series_order : list of str, optional
        Order of the forecast rows. Default :data:`SERIES_IDS`.
    **kwargs
        Passed to ``compute_hierarchical_wrmsse`` (e.g. ``window``).

    Returns
    -------
    dict of int to float
        WRMSSE per level number.
    """
    order = [SERIES_IDS.index(sid) for sid in series_order]
    forecasts = ds["sales"].values[order, days] + OFFSETS[order, None]
    per_level = scorer.compute_hierarchical_wrmsse(ds, forecasts, np.array(series_order), **kwargs)
    assert list(per_level["level"]) == list(range(1, 13))
    return dict(zip(per_level["level"], per_level["wrmsse"]))


class TestRmsseScale:
    def test_scale_starts_at_the_first_sale(self):
        y = np.array(
            [
                [0.0, 0.0, 1.0, 3.0, 2.0],  # late launch: differences (2, -1) -> 5/2; full history gives 6/4
                [2.0, 0.0, 4.0, 4.0, 1.0],  # sells from day 0, so its interior zero counts: 29/4
                [0.0, 0.0, 0.0, 0.0, 5.0],  # first sale on the last day: no difference left, falls back
                [0.0, 0.0, 0.0, 0.0, 0.0],  # never sold: falls back
            ]
        )

        np.testing.assert_allclose(scorer.rmsse_scale(y), [2.5, 7.25, 1.0, 1.0])

    def test_launch_zeros_do_not_shrink_the_hierarchy_scale(self):
        # Every series, and so every aggregate, launches on day 7. The trimmed level-1 scale is 16 (4^2),
        # giving RMSSE 1/4; keeping the six leading zero-differences would scale it by 33/39 and inflate
        # the score to 0.25 x sqrt(39/33) ~ 0.272.
        levels = _score(_make_ds(), VALIDATION_DAYS)

        assert levels[1] == pytest.approx(0.25)


class TestDollarWeights:
    def test_price_gap_keeps_the_priced_dollar_sales(self):
        # Level 12 is series 0's own RMSSE (1) times its weight. Summing straight through the gap made
        # its dollar sales NaN, which was then zeroed, scoring 0; skipping the gap keeps $28 of $70.
        levels = _score(_make_ds(), VALIDATION_DAYS)

        assert levels[12] == pytest.approx(28 / 70)

    def test_a_fully_priced_window_weighs_every_day(self):
        # The evaluation weights come from the validation days, which have no gap: 14 x 4.0 of $98.
        levels = _score(_make_ds(), EVALUATION_DAYS, window="evaluation")

        assert levels[12] == pytest.approx(56 / 98)


class TestScoringWindow:
    def test_validation_window_matches_the_hand_computed_levels(self):
        levels = _score(_make_ds(), VALIDATION_DAYS, window="validation")

        assert levels == pytest.approx(EXPECTED_VALIDATION)
        assert np.mean(list(levels.values())) == pytest.approx(EXPECTED_VALIDATION_WRMSSE)

    def test_evaluation_window_matches_the_hand_computed_levels(self):
        # The test-window actuals sit TEST_SHIFT above the validation ones, so a scorer reading the
        # wrong window would see an error of 5 on every series and miss these values by far.
        levels = _score(_make_ds(), EVALUATION_DAYS, window="evaluation")

        assert levels == pytest.approx(EXPECTED_EVALUATION)
        assert np.mean(list(levels.values())) == pytest.approx(EXPECTED_EVALUATION_WRMSSE)

    def test_validation_is_the_default_window(self):
        assert _score(_make_ds(), VALIDATION_DAYS) == pytest.approx(EXPECTED_VALIDATION)

    def test_forecast_rows_are_matched_by_series_id(self):
        levels = _score(_make_ds(), VALIDATION_DAYS, series_order=SERIES_IDS[::-1])

        assert levels == pytest.approx(EXPECTED_VALIDATION)

    def test_evaluation_without_actuals_raises(self):
        forecasts = np.zeros((N_SERIES, N_STEPS))

        with pytest.raises(RuntimeError, match="no evaluation actuals"):
            scorer.compute_hierarchical_wrmsse(
                _make_ds(test_actuals=False), forecasts, np.array(SERIES_IDS), window="evaluation"
            )

    def test_unknown_window_raises(self):
        forecasts = np.zeros((N_SERIES, N_STEPS))

        with pytest.raises(ValueError, match="Unknown window"):
            scorer.compute_hierarchical_wrmsse(_make_ds(), forecasts, np.array(SERIES_IDS), window="test")

    @pytest.mark.parametrize(("window", "value"), [("validation", 1.0), ("evaluation", 2.0)])
    def test_submission_rows_follow_the_window(self, window: str, value: float):
        # Validation rows forecast 1.0 everywhere, evaluation rows 2.0, as a two-window submission would.
        sub = pd.DataFrame({"id": [f"{sid}_{w}" for w in ("validation", "evaluation") for sid in SERIES_IDS]})
        for i in range(1, N_STEPS + 1):
            sub[f"F{i}"] = [1.0] * N_SERIES + [2.0] * N_SERIES

        forecasts, series_order = scorer._submission_rows(sub, window)

        assert list(series_order) == SERIES_IDS
        assert forecasts.shape == (N_SERIES, N_STEPS)
        np.testing.assert_array_equal(forecasts, value)
