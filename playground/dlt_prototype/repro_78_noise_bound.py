"""Reproduction for #78: does the DLT predict path use future state, and is the noise bound too small?

Simulates many futures from the true DLT generative process starting at a fixed forecast origin, then
compares the spread of ``make_inference`` draws against it.  Also checks that the out-of-sample states
are an exact function of ``(last_lev, last_slp)`` and the noise draws, i.e. nothing beyond the train end.

Run: ``python playground/dlt_prototype/repro_78_noise_bound.py [out.png]``
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

jax.config.update("jax_enable_x64", True)

from bunobee.models.dlt import generate_dlt_components, generate_noise_components, make_inference

LEV_SM, SLP_SM, THETA = 0.3, 0.2, 0.9
SIGMA, N_TRAIN, H, N_SAMPLES = 1.0, 30, 12, 20000
LAST_LEV, LAST_SLP = 10.0, 0.5


def make_idata():
    """Hand-built posterior at a fixed origin (no MCMC): state at train end plus a fixed sigma."""
    return xr.Dataset(
        {
            "dlt_comp": (["chain", "draw", "time"], np.zeros((1, N_SAMPLES, N_TRAIN))),
            "sigma": (["chain", "draw"], np.full((1, N_SAMPLES), SIGMA)),
            "last_lev": (["chain", "draw"], np.full((1, N_SAMPLES), LAST_LEV)),
            "last_slp": (["chain", "draw"], np.full((1, N_SAMPLES), LAST_SLP)),
        }
    )


def simulate_truth(rng, n, h):
    """True future ``y`` from the DLT innovations form, started at the same origin state."""
    lev, slp = np.full(n, LAST_LEV), np.full(n, LAST_SLP)
    ys = np.empty((n, h))
    for t in range(h):
        comp = lev + THETA * slp
        ys[:, t] = comp + rng.normal(scale=SIGMA, size=n)
        new_lev = LEV_SM * ys[:, t] + (1 - LEV_SM) * comp
        slp = SLP_SM * (new_lev - lev) + (1 - SLP_SM) * slp
        lev = new_lev
    return ys


def numpy_recursion(eps):
    """Reference OOS recursion using only the train-end state and the noise draws."""
    lev, slp = np.full(eps.shape[0], LAST_LEV), np.full(eps.shape[0], LAST_SLP)
    comps = np.empty_like(eps)
    for t in range(eps.shape[1]):
        comps[:, t] = lev + THETA * slp
        new_lev = LEV_SM * (comps[:, t] + eps[:, t]) + (1 - LEV_SM) * comps[:, t]
        slp = SLP_SM * (new_lev - lev) + (1 - SLP_SM) * slp
        lev = new_lev
    return comps


def main(out_path=None):
    idata = make_idata()
    key = jax.random.PRNGKey(0)

    # 1. leak check: OOS dlt_comp == reference recursion from (last_lev, last_slp) + the same eps draws
    comp = np.asarray(generate_dlt_components(key, idata, N_TRAIN + H, LEV_SM, SLP_SM, THETA))[:, N_TRAIN:]
    eps = np.asarray(generate_noise_components(key, idata, H))
    np.testing.assert_allclose(comp, numpy_recursion(eps), rtol=1e-10)
    assert np.allclose(comp[:, 0], LAST_LEV + THETA * LAST_SLP)
    print("leak check: OOS states depend only on (last_lev, last_slp, eps) -> no future state used")

    # 2. spread vs the true generative process
    truth = simulate_truth(np.random.default_rng(1), N_SAMPLES, H)
    rows = {}
    for noise_embed in (False, True):
        out = make_inference(key, idata, LEV_SM, SLP_SM, THETA, end_step=N_TRAIN + H, noise_embed=noise_embed)
        rows[noise_embed] = np.asarray(out["forecast_samples"])[:, N_TRAIN:]

    print(f"{'h':>3} {'true std':>9} {'embed=F':>9} {'embed=T':>9}  90% cov(embed=T)")
    lo, hi = np.percentile(rows[True], [5, 95], axis=0)
    cov = ((truth >= lo) & (truth <= hi)).mean(axis=0)
    for h in range(H):
        print(
            f"{h + 1:>3} {truth[:, h].std():>9.3f} {rows[False][:, h].std():>9.3f} {rows[True][:, h].std():>9.3f}  {cov[h]:.3f}"
        )

    if out_path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.arange(1, H + 1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, truth.std(0), "k-", label="true predictive std")
        ax.plot(x, rows[False].std(0), "r--", label="forecast, noise_embed=False")
        ax.plot(x, rows[True].std(0), "g:", label="forecast, noise_embed=True")
        ax.set_xlabel("horizon")
        ax.set_ylabel("std")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
