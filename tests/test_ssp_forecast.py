"""Unit tests for :func:`bunobee.models.ssp.forecast.forecast_ssp`.

One test (class) per acceptance criterion of issue #26:

1. ``horizon = 1`` with ``sigma_q = 0`` and ``noise_embed = False`` reproduces
   ``Z_future[0] @ at_nat[:, -1, :].T`` exactly, per sample.
2. ``positivity = all-False`` (linear) matches ``einsum("hij,shj->shi", Z_future, a_T)``
   under ``sigma_q = 0``.
3. Cross-sample forecast variance is nondecreasing in ``time``.
4. Output dims / coords are exactly ``(sample, time, series)``; ``time`` length equals ``horizon``.
5. ``noise_embed = True`` inflates per-step variance by about ``sigma_h**2`` versus ``False``.
6. Exported from ``ssp/__init__.py``; NumPy-style docstring; lines <= 120 chars.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import bunobee.models.ssp as ssp
from bunobee.models.ssp import a_to_lam, forecast_ssp
from bunobee.models.ssp import forecast as forecast_mod


def _make_idata(
    *,
    n_chains: int = 2,
    n_draws: int = 50,
    n_steps: int = 12,
    n_states: int = 3,
    n_series: int = 2,
    sigma_q=0.1,
    sigma_h=0.2,
    seed: int = 0,
    with_positivity=None,
    exponent_attr: float | None = None,
) -> xr.Dataset:
    """Build a synthetic SSP posterior with the sites ``forecast_ssp`` consumes."""
    rng = np.random.default_rng(seed)
    at = rng.normal(0.0, 0.5, size=(n_chains, n_draws, n_steps, n_states))
    sigma_q_arr = np.broadcast_to(
        np.broadcast_to(np.asarray(sigma_q, dtype=float), (n_states,)),
        (n_chains, n_draws, n_states),
    ).copy()
    sigma_h_arr = np.broadcast_to(
        np.broadcast_to(np.asarray(sigma_h, dtype=float), (n_series,)),
        (n_chains, n_draws, n_series),
    ).copy()
    ds = xr.Dataset(
        data_vars={
            "at": (["chain", "draw", "step", "state"], at),
            "sigma_q": (["chain", "draw", "state"], sigma_q_arr),
            "sigma_h": (["chain", "draw", "series"], sigma_h_arr),
        },
        coords={"chain": np.arange(n_chains), "draw": np.arange(n_draws)},
    )
    if with_positivity is not None:
        ds["positivity"] = (["state"], np.asarray(with_positivity, dtype=bool))
    if exponent_attr is not None:
        ds.attrs["exponent"] = exponent_attr
    return ds


def _flat_at(ds: xr.Dataset) -> np.ndarray:
    """Flatten ``(chain, draw)`` of the ``at`` site to a leading ``sample`` axis."""
    at = ds["at"].to_numpy()
    return at.reshape(-1, at.shape[2], at.shape[3])


# ---------------------------------------------------------------------------
# Criterion 1 — horizon-1, sigma_q = 0 closed form (nonlinear default)
# ---------------------------------------------------------------------------


class TestHorizonOneIdentity:
    def test_matches_last_state_closed_form(self):
        n_states, n_series = 3, 2
        ds = _make_idata(sigma_q=0.0, n_states=n_states, n_series=n_series, n_steps=9)
        rng = np.random.default_rng(1)
        Z_future = rng.normal(size=(1, n_series, n_states))

        out = forecast_ssp(ds, Z_future, noise_embed=False, seed=0)

        a_T = _flat_at(ds)[:, -1, :]
        a_T_nat = a_to_lam(a_T, 0.5, None)  # positivity default: every state nonlinear
        expected = a_T_nat @ Z_future[0].T  # (sample, series)

        np.testing.assert_allclose(out["forecast_samples"].values[:, 0, :], expected, rtol=1e-12, atol=1e-12)

    def test_no_noise_forecast_equals_mu_and_omits_eps(self):
        ds = _make_idata(sigma_q=0.0)
        rng = np.random.default_rng(2)
        Z_future = rng.normal(size=(1, 2, 3))

        out = forecast_ssp(ds, Z_future, noise_embed=False, seed=0)

        np.testing.assert_array_equal(out["forecast_samples"].values, out["mu_samples"].values)
        assert "eps_samples" not in out


# ---------------------------------------------------------------------------
# Criterion 2 — linear (positivity all-False), sigma_q = 0 closed form
# ---------------------------------------------------------------------------


class TestLinearClosedForm:
    def test_matches_einsum_over_horizon(self):
        n_states, n_series, horizon = 3, 2, 5
        ds = _make_idata(sigma_q=0.0, n_states=n_states, n_series=n_series, n_steps=10)
        rng = np.random.default_rng(3)
        Z_future = rng.normal(size=(horizon, n_series, n_states))
        positivity = np.zeros(n_states, dtype=bool)

        out = forecast_ssp(ds, Z_future, positivity=positivity, seed=4)

        a_T = _flat_at(ds)[:, -1, :]  # (sample, n_states)
        a_nat = np.broadcast_to(a_T[:, None, :], (a_T.shape[0], horizon, n_states))
        expected = np.einsum("hij,shj->shi", Z_future, a_nat)

        np.testing.assert_allclose(out["mu_samples"].values, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(out["forecast_samples"].values, expected)


# ---------------------------------------------------------------------------
# Criterion 3 — cross-sample variance nondecreasing in time
# ---------------------------------------------------------------------------


class TestVarianceMonotonicity:
    def test_forecast_variance_nondecreasing(self):
        n_states, n_series, horizon = 2, 3, 12
        ds = _make_idata(
            sigma_q=0.4,
            n_states=n_states,
            n_series=n_series,
            n_steps=8,
            n_chains=4,
            n_draws=500,
        )
        rng = np.random.default_rng(5)
        # Constant design across the horizon so the random-walk variance growth is clean.
        z_block = rng.normal(size=(n_series, n_states))
        Z_future = np.broadcast_to(z_block, (horizon, n_series, n_states)).copy()
        positivity = np.zeros(n_states, dtype=bool)  # linear -> exact RW variance growth

        out = forecast_ssp(ds, Z_future, positivity=positivity, noise_embed=False, seed=0)

        var_t = out["forecast_samples"].var(dim="sample").values  # (time, series)
        diffs = np.diff(var_t, axis=0)
        assert np.all(diffs >= -1e-9)
        assert np.all(var_t[-1] > var_t[0])


# ---------------------------------------------------------------------------
# Criterion 4 — output dims / coords exactly (sample, time, series)
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_dims_coords_and_time_length(self):
        n_states, n_series, horizon = 3, 2, 7
        ds = _make_idata(n_states=n_states, n_series=n_series, n_steps=10, n_chains=2, n_draws=30)
        rng = np.random.default_rng(6)
        Z_future = rng.normal(size=(horizon, n_series, n_states))

        out = forecast_ssp(ds, Z_future, noise_embed=True, seed=0)

        for name in ("forecast_samples", "mu_samples", "eps_samples"):
            assert out[name].dims == ("sample", "time", "series")
        assert dict(out.sizes) == {"sample": 60, "time": horizon, "series": n_series}
        np.testing.assert_array_equal(out.coords["sample"].values, np.arange(60))
        np.testing.assert_array_equal(out.coords["time"].values, np.arange(horizon))
        np.testing.assert_array_equal(out.coords["series"].values, np.arange(n_series))

    @pytest.mark.parametrize("horizon", [1, 3, 20])
    def test_time_length_equals_horizon(self, horizon):
        ds = _make_idata(n_states=2, n_series=2, n_steps=6)
        rng = np.random.default_rng(horizon)
        Z_future = rng.normal(size=(horizon, 2, 2))

        out = forecast_ssp(ds, Z_future, seed=0)

        assert out.sizes["time"] == horizon

    def test_bad_z_future_shape_raises(self):
        ds = _make_idata(n_states=3, n_series=2)
        with pytest.raises(ValueError):
            forecast_ssp(ds, np.zeros((4, 3)), seed=0)  # 2-D
        with pytest.raises(ValueError):
            forecast_ssp(ds, np.zeros((4, 2, 5)), seed=0)  # wrong n_states
        with pytest.raises(ValueError):
            forecast_ssp(ds, np.zeros((4, 9, 3)), seed=0)  # wrong n_series


# ---------------------------------------------------------------------------
# Criterion 5 — noise_embed inflates per-step variance by about sigma_h**2
# ---------------------------------------------------------------------------


class TestNoiseEmbed:
    def test_variance_inflation_matches_sigma_h_squared(self):
        n_states, n_series, horizon = 2, 3, 6
        sigma_h = np.array([0.2, 0.5, 0.3])
        ds = _make_idata(
            sigma_q=0.05,
            sigma_h=sigma_h,
            n_states=n_states,
            n_series=n_series,
            n_steps=8,
            n_chains=4,
            n_draws=750,
        )
        rng = np.random.default_rng(7)
        Z_future = rng.normal(size=(horizon, n_series, n_states))

        out_plain = forecast_ssp(ds, Z_future, noise_embed=False, seed=11)
        out_noisy = forecast_ssp(ds, Z_future, noise_embed=True, seed=11)

        assert "eps_samples" not in out_plain
        assert "eps_samples" in out_noisy
        # Same seed: eta is drawn first, so mu is identical between the two calls.
        np.testing.assert_allclose(out_plain["mu_samples"].values, out_noisy["mu_samples"].values, rtol=0, atol=0)

        var_plain = out_plain["forecast_samples"].var(dim="sample").values
        var_noisy = out_noisy["forecast_samples"].var(dim="sample").values
        inflation = (var_noisy - var_plain).mean(axis=0)  # (series,)

        np.testing.assert_allclose(inflation, sigma_h**2, rtol=0.1)

    def test_eps_samples_scale_with_sigma_h(self):
        ds = _make_idata(sigma_h=np.array([0.1, 0.7]), n_states=2, n_series=2, n_chains=4, n_draws=750)
        rng = np.random.default_rng(8)
        Z_future = rng.normal(size=(5, 2, 2))

        out = forecast_ssp(ds, Z_future, noise_embed=True, seed=3)
        eps_sd = out["eps_samples"].std(dim="sample").values  # (time, series)

        np.testing.assert_allclose(eps_sd.mean(axis=0), np.array([0.1, 0.7]), rtol=0.1)


# ---------------------------------------------------------------------------
# Criterion 6 — export, docstring, line length, plus seed stability
# ---------------------------------------------------------------------------


class TestApiSurface:
    def test_exported_from_package(self):
        assert "forecast_ssp" in ssp.__all__
        assert ssp.forecast_ssp is forecast_ssp
        assert list(ssp.__all__) == sorted(ssp.__all__)

    def test_numpy_style_docstring(self):
        doc = forecast_ssp.__doc__
        assert doc is not None
        for section in ("Parameters\n    ----------", "Returns\n    -------"):
            assert section in doc

    def test_source_lines_within_120_chars(self):
        lines = Path(forecast_mod.__file__).read_text().splitlines()
        offenders = [(i + 1, len(ln)) for i, ln in enumerate(lines) if len(ln) > 120]
        assert not offenders, f"lines over 120 chars: {offenders}"

    def test_seed_is_bitwise_stable(self):
        ds = _make_idata(n_states=3, n_series=2, n_chains=2, n_draws=40)
        rng = np.random.default_rng(9)
        Z_future = rng.normal(size=(6, 2, 3))

        a = forecast_ssp(ds, Z_future, noise_embed=True, seed=123)
        b = forecast_ssp(ds, Z_future, noise_embed=True, seed=123)
        xr.testing.assert_identical(a, b)

        c = forecast_ssp(ds, Z_future, noise_embed=True, seed=124)
        assert not np.allclose(a["forecast_samples"].values, c["forecast_samples"].values)


# ---------------------------------------------------------------------------
# idata-carried exponent / positivity take precedence over the arguments
# ---------------------------------------------------------------------------


class TestIdataOverrides:
    def test_positivity_var_and_exponent_attr_win(self):
        n_states, n_series = 3, 2
        positivity_var = np.array([True, False, True])
        ds = _make_idata(
            sigma_q=0.0,
            n_states=n_states,
            n_series=n_series,
            n_steps=6,
            with_positivity=positivity_var,
            exponent_attr=0.25,
        )
        rng = np.random.default_rng(10)
        Z_future = rng.normal(size=(2, n_series, n_states))

        # Deliberately pass conflicting arguments; idata should override both.
        out = forecast_ssp(ds, Z_future, exponent=0.5, positivity=np.ones(n_states, dtype=bool), seed=0)

        a_T = _flat_at(ds)[:, -1, :]
        a_nat = a_to_lam(a_T, 0.25, positivity_var)
        expected0 = a_nat @ Z_future[0].T

        np.testing.assert_allclose(out["mu_samples"].values[:, 0, :], expected0, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Issue #30 — filter-native forecast path (method="filter")
# ---------------------------------------------------------------------------


class TestFilterMethod:
    def test_rejects_unknown_method(self):
        ds = _make_idata()
        rng = np.random.default_rng(20)
        Z_future = rng.normal(size=(3, 2, 3))
        with pytest.raises(ValueError, match="method must be one of"):
            forecast_ssp(ds, Z_future, method="posterior-replay")

    def test_matches_replay_in_mean_and_is_not_wider(self):
        """The filter path reproduces the replay mean and does not inflate the spread.

        With a random-walk state equation the predict-only covariance the filter
        propagates is exactly ``h * diag(sigma_q**2)``, so both paths must agree to
        the single precision the filter runs in.
        """
        n_states, n_series, horizon = 3, 2, 8
        ds = _make_idata(n_states=n_states, n_series=n_series, n_steps=10, n_draws=60, sigma_q=0.3)
        rng = np.random.default_rng(21)
        Z_future = rng.normal(size=(horizon, n_series, n_states))

        replay = forecast_ssp(ds, Z_future, method="replay", noise_embed=True, seed=7)
        filtered = forecast_ssp(ds, Z_future, method="filter", noise_embed=True, seed=7)

        for name in ("mu_samples", "forecast_samples"):
            lhs = replay[name].values
            rhs = filtered[name].values
            np.testing.assert_allclose(rhs.mean(axis=0), lhs.mean(axis=0), rtol=2e-4, atol=1e-6)
            replay_sd = lhs.std(axis=0)
            filter_sd = rhs.std(axis=0)
            assert np.all(filter_sd <= replay_sd * (1.0 + 1e-3) + 1e-8), f"{name} spread widened"

    def test_zero_process_noise_matches_replay_closed_form(self):
        """sigma_q = 0 makes the propagated covariance zero, so both paths are the state itself."""
        n_states, n_series, horizon = 3, 2, 4
        ds = _make_idata(sigma_q=0.0, n_states=n_states, n_series=n_series, n_steps=7)
        rng = np.random.default_rng(22)
        Z_future = rng.normal(size=(horizon, n_series, n_states))

        out = forecast_ssp(ds, Z_future, method="filter", seed=3)

        a_T = _flat_at(ds)[:, -1, :]
        a_nat = a_to_lam(np.broadcast_to(a_T[:, None, :], (a_T.shape[0], horizon, n_states)), 0.5, None)
        expected = np.einsum("hij,shj->shi", Z_future, a_nat)
        np.testing.assert_allclose(out["mu_samples"].values, expected, rtol=1e-8, atol=1e-8)

    def test_dims_and_variance_growth_match_the_replay_contract(self):
        n_states, n_series, horizon = 2, 3, 10
        ds = _make_idata(sigma_q=0.5, n_states=n_states, n_series=n_series, n_steps=6, n_chains=4, n_draws=500)
        rng = np.random.default_rng(23)
        # Constant design across the horizon so the random-walk variance growth is clean.
        z_block = rng.normal(size=(n_series, n_states))
        Z_future = np.broadcast_to(z_block, (horizon, n_series, n_states)).copy()

        out = forecast_ssp(ds, Z_future, method="filter", positivity=np.zeros(n_states, dtype=bool), seed=5)

        assert out["forecast_samples"].dims == ("sample", "time", "series")
        assert out.sizes["time"] == horizon
        var = out["forecast_samples"].values.var(axis=0)
        assert np.all(np.diff(var, axis=0) >= -1e-9), "filter-path variance is not nondecreasing in time"
        assert np.all(var[-1] > var[0])
