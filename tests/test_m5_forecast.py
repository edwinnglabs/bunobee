"""Unit tests for the m5 forecast wrappers.

One test (class) per acceptance criterion of issue #31:

1. ``predict_one_series`` delegates to ``forecast_ssp`` — it carries no propagation
   or contraction math of its own.
2. Its output is numerically unchanged versus the pre-refactor implementation
   (``median(exp(a_last @ Z_future.T + eps) * response_norm)``).
3. The deprecated MAP predictors keep their closed-form output bit-for-bit and carry
   a replacement note pointing at ``forecast_ssp``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bunobee.models.m5 import m5_ssp_mcmc
from bunobee.models.m5.m5_ssp_mcmc import predict_one_series

try:
    from bunobee.models.m5 import m5_ssp_optim
    from bunobee.models.m5.m5_ssp_optim import predict_batch_series_opt, predict_one_series_opt
except ModuleNotFoundError:  # the optional 'm5' extra (optax, tqdm) is not installed
    m5_ssp_optim = None
    predict_one_series_opt = None
    predict_batch_series_opt = None

requires_m5_extra = pytest.mark.skipif(
    m5_ssp_optim is None,
    reason="needs the optional 'm5' extra: pip install 'bunobee[m5]'",
)

HORIZON = 14
N_SAMPLES = 4000
N_STEPS = 28
N_STATES = 7


def _make_fit_result(seed: int = 0) -> dict:
    """Build a synthetic ``fit_one_series`` result with the sites the predictor reads."""
    rng = np.random.default_rng(seed)
    n_steps_total = N_STEPS
    weekly = np.zeros((n_steps_total, N_STATES - 1))
    weekly[np.arange(n_steps_total), np.arange(n_steps_total) % 7 - 1] = 1.0
    weekly[np.arange(n_steps_total) % 7 == 0] = 0.0
    Z = np.concatenate([np.ones((n_steps_total, 1)), weekly], axis=1)

    # A concentrated posterior: draws scatter around a common state path, as NUTS output does.
    a_center = rng.normal(0.0, 0.3, size=(n_steps_total, N_STATES))
    at = a_center[None] + rng.normal(0.0, 0.05, size=(N_SAMPLES, n_steps_total, N_STATES))
    sigma_h = rng.uniform(0.15, 0.25, size=N_SAMPLES)

    return {
        "posterior_dict": {"at": at, "sigma_h": sigma_h},
        "response_norm": 3.5,
        "Z": Z,
        "a0": np.zeros(N_STATES),
        "P0": np.ones(N_STATES),
    }


def _make_z_future(seed: int = 1) -> np.ndarray:
    """A ``(horizon, n_states)`` future design matrix — intercept plus weekly dummies."""
    rng = np.random.default_rng(seed)
    Z_future = np.zeros((HORIZON, N_STATES))
    Z_future[:, 0] = 1.0
    for h in range(HORIZON):
        col = (h % 7) - 1
        if col >= 0:
            Z_future[h, 1 + col] = 1.0
    # keep an unused rng draw so the helper is stable if the layout changes
    _ = rng.random()
    return Z_future


def _legacy_predict_one_series(fit_result: dict, Z_future: np.ndarray) -> np.ndarray:
    """The pre-refactor m5 forecast, kept verbatim as the numerical reference."""
    posterior_dict = fit_result["posterior_dict"]
    response_norm = fit_result["response_norm"]
    at_samples = np.array(posterior_dict["at"])
    sigma_h_samples = np.array(posterior_dict["sigma_h"])

    a_last = at_samples[:, -1, :]
    mu_future = a_last @ Z_future.T
    eps = np.random.default_rng(42).normal(0, sigma_h_samples[:, None], size=mu_future.shape)
    yhat_samples = np.exp(mu_future + eps) * response_norm
    return np.median(yhat_samples, axis=0)


class TestDelegation:
    """Criterion 1: the m5 predictor is a wrapper, not a second implementation."""

    def test_calls_forecast_ssp(self, monkeypatch):
        """``predict_one_series`` routes the propagation through ``forecast_ssp``."""
        calls: list[dict] = []
        real_forecast_ssp = m5_ssp_mcmc.forecast_ssp

        def _spy(idata, Z_future, **kwargs):
            calls.append({"idata": idata, "Z_future": Z_future, **kwargs})
            return real_forecast_ssp(idata, Z_future, **kwargs)

        monkeypatch.setattr(m5_ssp_mcmc, "forecast_ssp", _spy)
        predict_one_series(_make_fit_result(), Z_future=_make_z_future())

        assert len(calls) == 1
        call = calls[0]
        assert call["Z_future"].shape == (HORIZON, 1, N_STATES)
        assert call["noise_embed"] is True
        assert not call["positivity"].any(), "the m5 model is linear-Gaussian on the log scale"

    def test_no_inline_propagation_math(self):
        """The module body no longer contracts the design matrix or draws eps itself."""
        source = Path(m5_ssp_mcmc.__file__).read_text()
        predict_src = source.split("def predict_one_series(")[1]
        assert "a_last @" not in predict_src
        assert "np.random.default_rng" not in predict_src

    def test_builds_design_when_z_future_missing(self):
        """Omitting ``Z_future`` still continues the weekly dummies via the shared helper."""
        out = predict_one_series(_make_fit_result(), horizon=HORIZON)
        assert out.shape == (HORIZON,)
        assert np.all(out > 0)


class TestNumericalRegression:
    """Criterion 2: same inputs, same forecast as the pre-refactor implementation."""

    def test_matches_legacy_median(self):
        """Medians agree with the legacy formula to within Monte-Carlo error."""
        fit_result = _make_fit_result()
        Z_future = _make_z_future()

        new = predict_one_series(fit_result, Z_future=Z_future)
        old = _legacy_predict_one_series(fit_result, Z_future)

        assert new.shape == old.shape == (HORIZON,)
        # Only the eps stream differs (forecast_ssp owns the RNG); the sampling
        # distribution is identical, so the medians agree up to MC noise.
        assert np.allclose(new, old, rtol=0.05)

    def test_mu_is_the_exact_legacy_contraction(self):
        """With sigma_q = 0 the shared core reproduces ``a_last @ Z_future.T`` exactly."""
        fit_result = _make_fit_result()
        Z_future = _make_z_future()

        idata = m5_ssp_mcmc._to_forecast_idata(
            np.asarray(fit_result["posterior_dict"]["at"]),
            np.asarray(fit_result["posterior_dict"]["sigma_h"]),
        )
        forecast = m5_ssp_mcmc.forecast_ssp(
            idata,
            Z_future[:, None, :],
            positivity=np.zeros(N_STATES, dtype=bool),
            noise_embed=False,
            seed=42,
        )
        mu = forecast["mu_samples"].to_numpy()[:, :, 0]
        expected = np.asarray(fit_result["posterior_dict"]["at"])[:, -1, :] @ Z_future.T
        assert np.allclose(mu, expected, atol=0, rtol=0)

    def test_seed_stability(self):
        """Repeated calls on the same inputs return the same forecast."""
        fit_result = _make_fit_result()
        Z_future = _make_z_future()
        first = predict_one_series(fit_result, Z_future=Z_future)
        second = predict_one_series(fit_result, Z_future=Z_future)
        assert np.array_equal(first, second)


@requires_m5_extra
class TestDeprecatedMapPredictors:
    """Criterion 3: the MAP predictors are untouched numerically and clearly deprecated."""

    def test_single_series_closed_form_unchanged(self):
        """``predict_one_series_opt`` still returns ``exp(mu + 0.5 var) * response_norm``."""
        rng = np.random.default_rng(3)
        at = rng.normal(size=(N_STEPS, N_STATES))
        Pt = rng.uniform(0.01, 0.2, size=(N_STEPS, N_STATES))
        fit_result = {"at": at, "Pt": Pt, "sigma_h": 0.25, "response_norm": 2.0}
        Z_future = _make_z_future()

        out = predict_one_series_opt(fit_result, Z_future=Z_future)
        expected = np.exp(Z_future @ at[-1] + 0.5 * (Z_future**2 @ Pt[-1] + 0.25**2)) * 2.0
        assert np.allclose(out, expected, atol=0, rtol=0)

    def test_batch_matches_single_series(self):
        """``predict_batch_series_opt`` agrees with the per-series function row by row."""
        rng = np.random.default_rng(4)
        B = 3
        at = rng.normal(size=(B, N_STEPS, N_STATES))
        Pt = rng.uniform(0.01, 0.2, size=(B, N_STEPS, N_STATES))
        sigma_h = rng.uniform(0.1, 0.4, size=B)
        response_norm = rng.uniform(1.0, 5.0, size=B)
        Z_future = _make_z_future()

        batch = predict_batch_series_opt(
            {"at": at, "Pt": Pt, "sigma_h": sigma_h, "response_norm": response_norm},
            Z_future=Z_future,
        )
        for b in range(B):
            single = predict_one_series_opt(
                {
                    "at": at[b],
                    "Pt": Pt[b],
                    "sigma_h": sigma_h[b],
                    "response_norm": response_norm[b],
                },
                Z_future=Z_future,
            )
            assert np.allclose(batch[b], single)

    @pytest.mark.parametrize("fn", [predict_one_series_opt, predict_batch_series_opt])
    def test_deprecation_note_points_at_forecast_ssp(self, fn):
        """Each deprecated predictor names its replacement in the docstring."""
        doc = fn.__doc__ or ""
        assert ".. deprecated::" in doc
        assert "forecast_ssp" in doc


class TestStyle:
    """Line-length budget for the touched modules."""

    @pytest.mark.parametrize(
        "module",
        [m5_ssp_mcmc, pytest.param(m5_ssp_optim, marks=requires_m5_extra)],
    )
    def test_lines_within_120_chars(self, module):
        """Project style caps source lines at 120 characters."""
        for i, line in enumerate(Path(module.__file__).read_text().splitlines(), start=1):
            assert len(line) <= 120, f"{module.__name__}:{i} is {len(line)} chars"
