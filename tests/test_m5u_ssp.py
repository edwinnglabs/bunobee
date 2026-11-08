"""Tests for scoring SSP predictive quantiles by WSPL, ``playground/m5_prototype/m5u_ssp.py`` (issue #158).

The Gaussian forecast of a series is mapped through the transform's inverse for the bottom level; the aggregate levels
are summed sample paths whose series depend on each other as the ``dependence`` says. The tests build a
:class:`GaussianForecast` by hand on the synthetic four-series hierarchy of ``tests/test_m5_uncertainty.py``, so none of
them fits a model or needs the raw M5 data.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

pytest.importorskip("optax", reason="needs the optional 'm5' extra: pip install 'bunobee[m5]'")

from bunobee.models.m5.transforms import LogTransform  # noqa: E402
from tests.test_m5_uncertainty import (  # noqa: E402
    EXPECTED_FULL,
    N_LEVELS,
    N_SERIES,
    N_STEPS,
    _forecasts_at_actuals,
    _load_script,
    _make_ds,
    score,
)

_load_script("m5_common")
_load_script("m5_score_submission")
_load_script("m5u_submission")
_load_script("m5u_baseline")
m5_models = _load_script("m5_models")
ssp = _load_script("m5u_ssp")

QUANTILES = np.asarray(score.QUANTILES)
RESPONSE_NORM = np.array([1.0, 2.0, 3.0, 4.0])


def _gaussian(var: float = 0.25) -> "m5_models.GaussianForecast":
    """A lognormal forecast of the four series: log-mean 0, 0.5, 1, 1.5 and the same variance on every day."""
    mu = np.tile(np.array([0.0, 0.5, 1.0, 1.5])[:, None], (1, N_STEPS))
    return m5_models.GaussianForecast(
        mu=mu,
        var=np.full_like(mu, var),
        fit_mu=mu,
        fit_var=np.full_like(mu, var),
        transform=LogTransform(),
        params={"response_norm": RESPONSE_NORM},
    )


@pytest.fixture(scope="module")
def hierarchy():
    return score.build_hierarchy(_make_ds())


class TestCopulaCorrelation:
    def test_series_that_move_together_are_correlated_one(self):
        base = np.random.default_rng(0).standard_normal((1, 200))
        corr = ssp.copula_correlation(np.vstack([base, 3 * base + 1, -base]))
        np.testing.assert_allclose(corr, np.sign(np.outer([1, 1, -1], [1, 1, -1])), atol=1e-6)

    def test_independent_series_are_nearly_uncorrelated(self):
        residuals = np.random.default_rng(1).standard_normal((6, 4000))
        corr = ssp.copula_correlation(residuals, n_days=4000)
        assert np.abs(corr - np.eye(6)).max() < 0.08

    def test_is_a_correlation_matrix(self):
        residuals = np.random.default_rng(2).standard_t(3, size=(8, 120))
        corr = ssp.copula_correlation(residuals)
        np.testing.assert_allclose(corr, corr.T)
        np.testing.assert_allclose(np.diag(corr), 1.0)
        assert np.linalg.eigvalsh(corr).min() > -1e-9

    def test_uses_the_last_days_only(self):
        rng = np.random.default_rng(3)
        recent = rng.standard_normal((3, 100))
        old = np.vstack([np.arange(50.0), np.arange(50.0), np.arange(50.0)])  # perfectly correlated, but old
        np.testing.assert_allclose(
            ssp.copula_correlation(np.hstack([old, recent]), n_days=100), ssp.copula_correlation(recent, n_days=100)
        )

    def test_dependences(self):
        np.testing.assert_array_equal(ssp.dependence_matrix("independent", 3), np.eye(3))
        np.testing.assert_array_equal(ssp.dependence_matrix("comonotone", 3), np.ones((3, 3)))
        with pytest.raises(ValueError, match="residuals"):
            ssp.dependence_matrix("copula", 3)
        with pytest.raises(ValueError, match="dependence must be one of"):
            ssp.dependence_matrix("gaussian", 3)


class TestJointQuantiles:
    N_DRAWS = 20_000

    def _forecasts(self, hierarchy, dependence, gaussian=None, seed=158):
        gaussian = gaussian or _gaussian()
        corr = ssp.dependence_matrix(dependence, N_SERIES, np.random.default_rng(4).standard_normal((N_SERIES, 300)))
        return ssp.joint_quantiles(gaussian, hierarchy, corr, n_draws=self.N_DRAWS, seed=seed)

    def test_one_array_per_level(self, hierarchy):
        forecasts = self._forecasts(hierarchy, "independent")
        assert len(forecasts) == N_LEVELS
        for fq, n_groups in zip(forecasts, hierarchy.n_groups):
            assert fq.shape == (n_groups, len(QUANTILES), N_STEPS)
            assert np.isfinite(fq).all() and (np.diff(fq, axis=1) >= 0).all()

    @pytest.mark.parametrize("dependence", ssp.DEPENDENCES)
    def test_bottom_level_is_the_exact_quantiles(self, hierarchy, dependence):
        gaussian = _gaussian()
        bottom = self._forecasts(hierarchy, dependence, gaussian)[-1]
        for series in range(N_SERIES):
            group = hierarchy.group_idx[-1][series]
            expected = stats.lognorm.ppf(
                QUANTILES, s=0.5, scale=np.exp(gaussian.mu[series, 0]) * RESPONSE_NORM[series]
            )
            np.testing.assert_allclose(bottom[group, :, 0], expected, rtol=1e-6)

    def test_bottom_level_follows_the_hierarchy_order_of_a_shuffled_subset(self):
        ds = _make_ds()
        rows = np.array([2, 0, 3, 1])
        hierarchy = score.build_hierarchy(ds, rows)
        gaussian = m5_models.GaussianForecast(
            mu=_gaussian().mu[rows],
            var=_gaussian().var[rows],
            fit_mu=_gaussian().fit_mu[rows],
            fit_var=_gaussian().fit_var[rows],
            transform=LogTransform(),
            params={"response_norm": RESPONSE_NORM[rows]},
        )
        bottom = ssp.joint_quantiles(gaussian, hierarchy, np.eye(N_SERIES), n_draws=50)[-1]
        median = bottom[:, QUANTILES.tolist().index(0.5), 0]
        for position, series in enumerate(rows):
            assert median[hierarchy.group_idx[-1][position]] == pytest.approx(
                np.exp(gaussian.mu[position, 0]) * RESPONSE_NORM[series]
            )

    def test_comonotone_total_is_the_sum_of_the_bottom_quantiles(self, hierarchy):
        forecasts = self._forecasts(hierarchy, "comonotone")
        # Each day's quantile is taken over its own draws, so average the days to shrink the sampling noise.
        np.testing.assert_allclose(forecasts[0][0].mean(axis=-1), forecasts[-1].sum(axis=0).mean(axis=-1), rtol=1e-2)

    def test_independence_is_narrower_than_comonotone_with_the_same_median_region(self, hierarchy):
        independent = self._forecasts(hierarchy, "independent")[0][0]
        comonotone = self._forecasts(hierarchy, "comonotone")[0][0]
        width = lambda q: q[-1] - q[0]  # noqa: E731
        assert (width(independent) < 0.8 * width(comonotone)).all()

    def test_independent_total_matches_a_direct_simulation(self, hierarchy):
        rng = np.random.default_rng(5)
        gaussian = _gaussian()
        draws = np.exp(gaussian.mu[:, :1] + 0.5 * rng.standard_normal((N_SERIES, 400_000))) * RESPONSE_NORM[:, None]
        expected = np.quantile(draws.sum(axis=0), QUANTILES)
        total = self._forecasts(hierarchy, "independent")[0][0, :, 0]
        np.testing.assert_allclose(total, expected, rtol=3e-2)

    def test_same_seed_same_draws(self, hierarchy):
        first, second = self._forecasts(hierarchy, "independent"), self._forecasts(hierarchy, "independent")
        other = self._forecasts(hierarchy, "independent", seed=1)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        assert not np.array_equal(first[0], other[0])

    def test_correlated_series_widen_the_total(self, hierarchy):
        corr = np.full((N_SERIES, N_SERIES), 0.6) + 0.4 * np.eye(N_SERIES)
        correlated = ssp.joint_quantiles(_gaussian(), hierarchy, corr, n_draws=self.N_DRAWS)[0][0]
        independent = self._forecasts(hierarchy, "independent")[0][0]
        assert ((correlated[-1] - correlated[0]) > (independent[-1] - independent[0])).all()


class TestScoring:
    def test_compare_methods_scores_each_set_per_level_and_overall(self):
        ds = _make_ds()
        hierarchy = score.build_hierarchy(ds)
        exact = _forecasts_at_actuals(ds, hierarchy, "validation", offset_series_0=0.0)
        off = _forecasts_at_actuals(ds, hierarchy, "validation")
        table = ssp.compare_methods(ds, hierarchy, {"exact": exact, "off": off}, "validation")
        assert list(table.columns) == ["exact", "off"]
        assert len(table) == N_LEVELS + 1 and table.index[-1] == "overall"
        np.testing.assert_allclose(table["exact"], 0.0, atol=1e-12)
        np.testing.assert_allclose(table["off"].iloc[:-1], EXPECTED_FULL)
        assert table.loc["overall", "off"] == pytest.approx(EXPECTED_FULL.mean())

    def test_coverage_is_the_share_of_actuals_at_or_below_each_quantile(self):
        ds = _make_ds()
        hierarchy = score.build_hierarchy(ds)
        exact = _forecasts_at_actuals(ds, hierarchy, "validation", offset_series_0=0.0)
        cover = ssp.coverage(ds, hierarchy, exact, "validation")
        assert cover.shape == (N_LEVELS, len(QUANTILES)) and list(cover.index) == list(range(1, N_LEVELS + 1))
        np.testing.assert_array_equal(cover.to_numpy(), 1.0)
        lower = ssp.coverage(ds, hierarchy, [fq - 100 for fq in exact], "validation")
        np.testing.assert_array_equal(lower.to_numpy(), 0.0)
