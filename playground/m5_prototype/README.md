# M5 prototype

An end-to-end run of the state space model (level + weekly seasonality, fitted by MAP) on the
[M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) competition:
fetch the data, fit all 30,490 item-store series, score locally and write a Kaggle submission.
Forecast quality is not the point yet; this is the workflow the model work builds on.

All commands run from the repo root. Every downloaded or generated file is gitignored.

[WORKFLOW.md](WORKFLOW.md) explains what to rerun after a change, how runs are scored, where to work on
the model, and what is still open.

## Prerequisites

1. The `m5` extra: `pip install -e ".[m5]"` (adds `optax`, `tqdm`).
2. The Kaggle CLI: `pip install kaggle`. It must be on `PATH` (with a venv, activate it or prefix
   `PATH="$PWD/.venv/bin:$PATH"`).
3. Kaggle credentials: `kaggle auth login`, or a token from <https://www.kaggle.com/settings/api> saved
   to `~/.kaggle/access_token` (or exported as `KAGGLE_API_TOKEN`).
4. The competition rules, accepted once while signed in:
   <https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules>.

## Runbook

```sh
# 1. Fetch: the 5 Kaggle files, plus the 4 official reference files (test actuals and scoring
#    weights) from a public mirror. Skips files already present; --force re-downloads.
python playground/m5_prototype/m5_fetch.py

# 2. Preprocess: build the cached dataset m5_ds.nc (~550 MB). A no-op once built; --force rebuilds.
python playground/m5_prototype/m5_preprocess.py

# 3. Check the scorer's weights against the official ones (all 42,840 series, both windows).
python playground/m5_prototype/m5_check_weights.py

cd playground/m5_prototype

# 4. Sampled run: fit 50 random series at both origins and score them. Takes seconds.
python m5_run.py --n-demo 50

# 5. Full run: fit all 30,490 series at both origins, write submission.csv and score it.
python m5_run.py --n-demo 0
#    About 5.5 minutes and a ~22 GiB peak (see below). With less RAM, cap memory with e.g.
#    --fit-chunk 8192.

# 6. Optional: re-score a submission file on either window.
python m5_score_submission.py --submission output/<timestamp>/submission.csv --window validation
python m5_score_submission.py --submission output/<timestamp>/submission.csv --window evaluation

# 7. Optional: submit to Kaggle (public leaderboard = _validation rows, private = _evaluation rows).
kaggle competitions submit -c m5-forecasting-accuracy -f output/<timestamp>/submission.csv -m "<message>"
kaggle competitions submissions -c m5-forecasting-accuracy | head -3
```

### What a run writes

Each `m5_run.py` run writes to `output/<timestamp>/`:

| File | Contents |
|---|---|
| `submission.csv` | 60,980 rows: `_validation` rows from the fit through d_1913, `_evaluation` rows from the fit through d_1941 (full runs only) |
| `wrmsse_per_level.csv` | 12-level WRMSSE and bias table, one `window` per origin |
| `run_stats.json` | Fit time (total and per origin), total time, peak memory |
| `batch_params.npz` | Fitted parameters and forecasts of both origins |
| `diagnostics.png`, `forecasts.pdf` | Loss / parameter histograms and forecast plots (validation origin) |

A sampled run aggregates and weights only its own series, so its WRMSSE is a smoke test, not
comparable to a full-run score.

### Forecast origins

| Origin | Fit on | Forecasts | Submission rows | Kaggle leaderboard |
|---|---|---|---|---|
| validation | d_1–d_1913 | d_1914–d_1941 | `_validation` | public |
| evaluation | d_1–d_1941 | d_1942–d_1969 | `_evaluation` | private |

## Reference results

Full run on an 18-core, 128 GB Mac, JAX on CPU, 100 Adam steps, no `--fit-chunk`:

| | Value |
|---|---|
| Fit time | 158.9 s (validation origin) + 160.2 s (evaluation origin); 328 s for the whole run |
| Peak memory | 22.3 GiB (14.4 GiB after the first origin: the second fit does not release all of the first one's memory) |
| Local WRMSSE, validation window | 1.08136 |
| Local WRMSSE, evaluation window | 0.97573 |

How well the local scorer matches the official one:

- **Weights:** all 42,840 weights match `weights_validation.csv` and `weights_evaluation.csv` to within
  about 1e-9, the precision of the official files (`m5_check_weights.py`).
- **Actuals:** the validation-window actuals equal the official `sales_test_validation.csv`.
- **Score:** for the same validation-origin submission, Kaggle's public score was 1.08127 against a
  local 1.08136. The remaining 0.00009 (about 0.01%) is not explained by the weights, the actuals,
  forecast clipping, float32 arithmetic or the trimming variants tried; it is far below the score
  differences between leaderboard entries.

## Scripts

| Script | Purpose |
|---|---|
| `m5_common.py` | Shared data directory, `load_m5_ds()`, day-of-week dummies |
| `m5_fetch.py` | Download the Kaggle files and the reference files |
| `m5_preprocess.py` | Build and cache `m5_ds.nc` |
| `m5_run.py` | Fit, forecast, score and write the submission |
| `m5_score_submission.py` | 12-level WRMSSE of a submission for one window |
| `m5_check_weights.py` | Compare the scorer's weights with the official files |
| `m5_submission_diff.py` | Diagnostic diff of two submissions |
| `m5_make_accuracy_fixture.py` | Build the committed CI fixture `src/bunobee/datasets/data/m5_aggregate.csv` |
