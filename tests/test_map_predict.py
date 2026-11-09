"""Unit tests for the predictors of the MAP fits in ``bunobee.models.ssp.map_fit``.

1. The deprecated MAP predictors keep their closed-form output bit-for-bit and carry a replacement note pointing at
   ``forecast_ssp``.
2. A MAP fit with fixed annual Fourier states carries the annual shape into the forecast days (issue #146).
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from bunobee import target_transforms
from bunobee.models.ssp import build_forecast_design
from bunobee.regression import make_fourier_series, make_periodic_dummies

try:
    from bunobee.models.ssp import map_fit
    from bunobee.models.ssp.map_fit import fit_map_batch, predict_map, predict_map_batch
except ModuleNotFoundError as exc:  # the optional 'optim' extra (optax, tqdm) is not installed
    if exc.name not in {"optax", "tqdm"}:
        raise
    map_fit = None
    fit_map_batch = None
    predict_map = None
    predict_map_batch = None

requires_optim_extra = pytest.mark.skipif(
    map_fit is None,
    reason="needs the optional 'optim' extra: pip install 'bunobee[optim]'",
)

HORIZON = 14
N_STEPS = 28
N_STATES = 7


def _make_z_future() -> np.ndarray:
    """A ``(horizon, n_states)`` future design matrix — intercept plus weekly dummies."""
    return np.asarray(make_periodic_dummies(HORIZON, period=7, intercept=True), dtype=float)


@requires_optim_extra
class TestDeprecatedMapPredictors:
    """The MAP predictors are untouched numerically and clearly deprecated."""

    def test_single_series_closed_form_unchanged(self):
        """``predict_map`` still returns ``exp(mu + 0.5 var) * response_norm``."""
        rng = np.random.default_rng(3)
        at = rng.normal(size=(N_STEPS, N_STATES))
        Pt = rng.uniform(0.01, 0.2, size=(N_STEPS, N_STATES))
        fit_result = {"at": at, "Pt": Pt, "sigma_h": 0.25, "response_norm": 2.0}
        Z_future = _make_z_future()

        out = predict_map(fit_result, Z_future=Z_future)
        expected = np.exp(Z_future @ at[-1] + 0.5 * (Z_future**2 @ Pt[-1] + 0.25**2)) * 2.0
        assert np.allclose(out, expected, atol=0, rtol=0)

    def test_batch_matches_single_series(self):
        """``predict_map_batch`` agrees with the per-series function row by row."""
        rng = np.random.default_rng(4)
        B = 3
        at = rng.normal(size=(B, N_STEPS, N_STATES))
        Pt = rng.uniform(0.01, 0.2, size=(B, N_STEPS, N_STATES))
        sigma_h = rng.uniform(0.1, 0.4, size=B)
        response_norm = rng.uniform(1.0, 5.0, size=B)
        Z_future = _make_z_future()

        batch = predict_map_batch(
            {"at": at, "Pt": Pt, "sigma_h": sigma_h, "response_norm": response_norm},
            Z_future=Z_future,
        )
        for b in range(B):
            single = predict_map(
                {
                    "at": at[b],
                    "Pt": Pt[b],
                    "sigma_h": sigma_h[b],
                    "response_norm": response_norm[b],
                },
                Z_future=Z_future,
            )
            assert np.allclose(batch[b], single)

    @pytest.mark.parametrize("fn", [predict_map, predict_map_batch])
    def test_deprecation_note_points_at_forecast_ssp(self, fn):
        """Each deprecated predictor names its replacement in the docstring."""
        doc = fn.__doc__ or ""
        assert ".. deprecated::" in doc
        assert "forecast_ssp" in doc


@requires_optim_extra
class TestAnnualTermsInTheForecast:
    """Issue #146: fixed annual Fourier states carry the annual shape past the end of the fit days."""

    N_FIT = 3 * 365
    HORIZON = 90

    def _forecasts(self, annual: bool) -> tuple[np.ndarray, np.ndarray]:
        """Forecasts of series with an annual cycle whose fit days end at its peak, and their true means."""
        days = np.arange(self.N_FIT + self.HORIZON)
        cycle = 0.6 * np.cos(2 * np.pi * (days - self.N_FIT) / 365.25)  # The peak falls on the first forecast day.
        mean = 20.0 * np.exp(cycle)
        values = np.random.default_rng(0).poisson(mean[: self.N_FIT], size=(4, self.N_FIT)).astype(np.float32)

        Z = np.ones((self.N_FIT, 1), dtype=np.float32)
        fixed = None
        Z_future = np.ones((self.HORIZON, 1), dtype=np.float32)
        if annual:
            terms = make_fourier_series(self.N_FIT, period=365.25, order=1).astype(np.float32)
            Z = np.concatenate([Z, terms], axis=1)
            fixed = np.array([False, True, True])
            Z_future = build_forecast_design(
                Z, self.HORIZON, fourier={"columns": [1, 2], "period": 365.25, "order": 1}
            )[:, 0]
        result = fit_map_batch(values, n_iter=50, Z=jnp.asarray(Z), fixed_states=fixed)
        return predict_map_batch(result, Z_future=Z_future), mean[self.N_FIT :]

    def test_annual_terms_track_the_fall_a_flat_forecast_misses(self):
        flat, truth = self._forecasts(annual=False)
        annual, _ = self._forecasts(annual=True)

        np.testing.assert_allclose(flat, np.broadcast_to(flat[:, [0]], flat.shape), rtol=1e-6)
        last_week, first_week = annual[:, -7:].mean(axis=1), annual[:, :7].mean(axis=1)
        assert np.all(last_week < 0.8 * first_week)
        assert np.mean((annual - truth) ** 2) < 0.5 * np.mean((flat - truth) ** 2)


class TestStyle:
    """Line-length budget for the modules."""

    @pytest.mark.parametrize(
        "module",
        [target_transforms, pytest.param(map_fit, marks=requires_optim_extra)],
    )
    def test_lines_within_120_chars(self, module):
        """Project style caps source lines at 120 characters."""
        for i, line in enumerate(Path(module.__file__).read_text().splitlines(), start=1):
            assert len(line) <= 120, f"{module.__name__}:{i} is {len(line)} chars"
