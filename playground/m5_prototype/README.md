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

# 4. Quick loop: fit 3,000 random series (seed 2026) at the validation origin only, with a
#    registered model (see Models below), and score them. About 20 s and a ~3.3 GiB peak. --model
#    defaults to ssp_dow, the baseline; run it once as the sample baseline to compare against.
python m5_run.py --model ssp_dow --n-demo 3000 --origins validation

# 5. Compare a run with the sample baseline: overall and per-level WRMSSE and bias deltas for each
#    window both runs scored. Refuses runs fitted on different series.
python m5_compare.py output/<baseline timestamp> output/<timestamp>

# 6. Full run: fit all 30,490 series at both origins, write submission.csv and score it.
python m5_run.py --model ssp_dow --n-demo 0
#    About 5.5 minutes and a ~22 GiB peak (see below). With less RAM, register an entry that fits
#    in runs of fewer series, e.g. "ssp_dow_8k": SspDow(fit_chunk=8192) in m5_models.py, and pass
#    --model ssp_dow_8k.

# 7. Optional: re-score a submission file on either window.
python m5_score_submission.py --submission output/<timestamp>/submission.csv --window validation
python m5_score_submission.py --submission output/<timestamp>/submission.csv --window evaluation

# 8. Optional: submit to Kaggle (public leaderboard = _validation rows, private = _evaluation rows),
#    then type the two scores into the run's row of output/runs.csv.
kaggle competitions submit -c m5-forecasting-accuracy -f output/<timestamp>/submission.csv -m "<message>"
kaggle competitions submissions -c m5-forecasting-accuracy | head -3
```

`--origins validation|evaluation|both` (default `both`) picks the forecast origins to fit. The quick
loop fits only the validation origin, so it never looks at the evaluation window, the holdout.

### What a run writes

Each `m5_run.py` run writes to `output/<timestamp>/`:

| File | Contents |
|---|---|
| `submission.csv` | 60,980 rows: `_validation` rows from the fit through d_1913, `_evaluation` rows from the fit through d_1941 (full runs with both origins only) |
| `wrmsse_per_level.csv` | 12-level WRMSSE and bias table, one `window` per fitted origin |
| `run_stats.json` | Model name and settings, origins, fit time (total and per origin), total time, peak memory |
| `batch_params.npz` | `fit_idx` (the fitted series) and the model's fitted outputs and forecasts of each fitted origin; evaluation-origin arrays carry an `_evaluation` suffix |
| `diagnostics.png` | Histograms of the model's fitted outputs (validation origin, or evaluation if only it is fitted) |
| `forecasts.pdf` | Forecast review of each fitted origin: the total, store and department sums of the fitted series, then `--n-plot` series (default 50) picked by `--plot-select`: `random` (default), `worst` (highest RMSSE) or `top` (highest dollar-sales weight). Each panel shows the last 90 fit days, the actuals and the forecast over the forecast window, and the model's in-sample fit if it returns one; its title gives the RMSSE, bias and weight |

and appends one row to the run log `output/runs.csv`: the timestamp, the run directory, the model and
its settings, the number of series, sample mode and seed, the origins, the WRMSSE of each scored window,
the fit time, the git commit and whether tracked files had uncommitted changes (`git_dirty`). After a
Kaggle submission, type its scores into the run's `kaggle_public` and `kaggle_private` columns; later
runs keep them.

A sampled run aggregates and weights only its own series, so its WRMSSE is comparable with other runs
on the same sample (same `--n-demo`, `--sample-mode` and `--seed`), not with a full-run score.
`m5_compare.py` checks that two runs fitted the same series before it compares them.

### Forecast origins

| Origin | Fit on | Forecasts | Submission rows | Kaggle leaderboard |
|---|---|---|---|---|
| validation | d_1–d_1913 | d_1914–d_1941 | `_validation` | public |
| evaluation | d_1–d_1941 | d_1942–d_1969 | `_evaluation` | private |

### Models

`--model` picks an entry of `MODELS` in `m5_models.py`. An entry is a model plus its settings: a
frozen dataclass whose `fit_forecast(ds, series_idx, fit_mask, forecast_mask)` builds the model's
inputs from the dataset, calls the model in `src/bunobee/models/m5/` and returns the forecasts
(series × 28) and a dict of per-series arrays to save. An optional `insample` array (series × fit days,
`INSAMPLE_KEY`) is the model's in-sample fit, which `forecasts.pdf` draws instead of saving. To try another
model, add an entry; `m5_run.py` needs no change.

| Entry | Model | Settings |
|---|---|---|
| `ssp_dow` | Weekday dummies from Tuesday to Sunday, no level state; MAP fit by Adam. The default, and the model of the #93 and #94 Kaggle submissions | `level=False`, `n_iter=100`, `lr=0.03`, `chunk_size=4096`, `fit_chunk=None` |
| `ssp_level_dow` | `ssp_dow` with a level state: a column of ones leads the weekday dummies, so 7 states (#106) | `level=True`, otherwise as `ssp_dow` |

## Reference results

Sample baseline: `ssp_dow` with its default settings on 3,000 random series (seed 2026), on the same
Mac as below:

| | Value |
|---|---|
| Fit time | 16.9 s (validation origin) + 17.2 s (evaluation origin) |
| Whole run | about 20 s with `--origins validation`, 37 s with both origins |
| Peak memory | 3.3 GiB with `--origins validation`, 3.5 GiB with both origins |
| Local WRMSSE, validation window | 0.93487 |
| Local WRMSSE, evaluation window | 0.88578 |

Full run of `ssp_dow` with its default settings on an 18-core, 128 GB Mac, JAX on CPU:

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
| `m5_common.py` | Shared data directory and `load_m5_ds()` |
| `m5_fetch.py` | Download the Kaggle files and the reference files |
| `m5_preprocess.py` | Build and cache `m5_ds.nc` |
| `m5_models.py` | Model registry: each entry is a model plus its settings, picked with `--model` |
| `m5_run.py` | Fit a registered model, forecast, score, write the submission and append to the run log |
| `m5_compare.py` | Per-level WRMSSE and bias deltas of two runs on the same series |
| `m5_score_submission.py` | 12-level WRMSSE of a submission for one window |
| `m5_check_weights.py` | Compare the scorer's weights with the official files |
| `m5_submission_diff.py` | Diagnostic diff of two submissions |
| `m5_make_accuracy_fixture.py` | Build the committed CI fixture `src/bunobee/datasets/data/m5_aggregate.csv` |
