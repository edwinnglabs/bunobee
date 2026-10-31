import jax
import numpy as np
import pandas as pd
import pytest

jax.config.update("jax_enable_x64", True)

pytest.importorskip("skopt")

from bunobee.hyper_tune import DEFAULT_SEARCH_SPACE, hyper_tuning_dlt_with_wbic, resolve_search_space
from bunobee.regression import RegressionScheme

N_TRAIN = 40
H = 4
# skopt's gp_minimize needs at least 10 calls
N_CALLS = 10
MCMC_ARGS = {"num_warmup": 30, "num_samples": 30, "num_chains": 1, "progress_bar": False}


@pytest.fixture(scope="module")
def simulated():
    rng = np.random.default_rng(0)
    x = rng.normal(size=N_TRAIN)
    y = np.linspace(10.0, 14.0, N_TRAIN) + 0.5 * x + rng.normal(scale=0.2, size=N_TRAIN)
    return y, pd.DataFrame({"x1": x})


def _check_output(out):
    assert {"lev_sm", "slp_sm", "theta", "result"} <= set(out)
    assert 0.0001 <= out["lev_sm"] <= 0.1
    assert 0.001 <= out["slp_sm"] <= 0.1
    assert 0.0 <= out["theta"] <= 1.0
    assert np.isfinite(out["result"].fun)


def test_tune_without_regressors(simulated):
    y, _ = simulated
    out = hyper_tuning_dlt_with_wbic(y, mcmc_run_args=MCMC_ARGS, h=H, n_calls=N_CALLS)
    _check_output(out)
    # no search_space tunes over the defaults
    assert _dims(out) == DEFAULT_SEARCH_SPACE


def test_tune_with_regressors(simulated):
    y, covariates_df = simulated
    scheme = RegressionScheme()
    scheme.add_regressors(["x1"])
    out = hyper_tuning_dlt_with_wbic(
        y,
        regression_scheme=scheme,
        covariates_df=covariates_df,
        mcmc_run_args=MCMC_ARGS,
        h=H,
        n_calls=N_CALLS,
    )
    _check_output(out)


def _dims(out):
    return {d.name: (d.low, d.high, d.prior) for d in out["result"].space.dimensions}


def test_default_search_space_unchanged():
    expected = {
        "lev_sm": (0.0001, 0.1, "log-uniform"),
        "slp_sm": (0.001, 0.1, "log-uniform"),
        "theta": (0.0, 1.0, "uniform"),
    }
    assert DEFAULT_SEARCH_SPACE == expected
    assert resolve_search_space() == expected
    assert resolve_search_space({}) == expected


def test_resolve_search_space_partial_and_prior():
    resolved = resolve_search_space({"lev_sm": (0.01, 0.5), "theta": (0.1, 0.9, "log-uniform")})
    # a (low, high) entry keeps the default prior
    assert resolved["lev_sm"] == (0.01, 0.5, "log-uniform")
    assert resolved["theta"] == (0.1, 0.9, "log-uniform")
    # a name left out keeps its default range and prior
    assert resolved["slp_sm"] == DEFAULT_SEARCH_SPACE["slp_sm"]
    assert list(resolved) == ["lev_sm", "slp_sm", "theta"]


@pytest.mark.parametrize(
    "search_space, match",
    [
        ({"alpha": (0.1, 0.2)}, "Unknown hyper-param"),
        ({"lev_sm": (0.1,)}, r"must be \(low, high\)"),
        ({"lev_sm": (0.1, 0.2, "normal", 1)}, r"must be \(low, high\)"),
        ({"theta": (0.1, 0.2, "normal")}, "prior must be one of"),
    ],
)
def test_resolve_search_space_rejects(search_space, match):
    with pytest.raises(ValueError, match=match):
        resolve_search_space(search_space)


def test_tune_unknown_name_raises(simulated):
    y, _ = simulated
    with pytest.raises(ValueError, match="Unknown hyper-param"):
        hyper_tuning_dlt_with_wbic(y, mcmc_run_args=MCMC_ARGS, h=H, n_calls=N_CALLS, search_space={"sm": (0, 1)})


def test_tune_with_search_space(simulated):
    y, _ = simulated
    search_space = {"slp_sm": (0.01, 0.05), "theta": (0.2, 0.4, "uniform")}
    out = hyper_tuning_dlt_with_wbic(y, mcmc_run_args=MCMC_ARGS, h=H, n_calls=N_CALLS, search_space=search_space)
    assert _dims(out) == {
        "lev_sm": DEFAULT_SEARCH_SPACE["lev_sm"],
        "slp_sm": (0.01, 0.05, "log-uniform"),
        "theta": (0.2, 0.4, "uniform"),
    }
    assert 0.01 <= out["slp_sm"] <= 0.05
    assert 0.2 <= out["theta"] <= 0.4
    assert all(0.01 <= x[1] <= 0.05 and 0.2 <= x[2] <= 0.4 for x in out["result"].x_iters)
