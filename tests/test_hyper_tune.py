import jax
import numpy as np
import pandas as pd
import pytest

jax.config.update("jax_enable_x64", True)

pytest.importorskip("skopt")

from bunobee.hyper_tune import hyper_tuning_dlt_with_wbic
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
