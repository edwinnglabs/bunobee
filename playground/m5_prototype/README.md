# M5 prototype

An end-to-end run of the state space model (level + weekly seasonality, fitted by MAP) on the
[M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) competition:
fetch the data, fit all 30,490 item-store series, score locally and write a Kaggle submission.
Forecast quality is not the point yet; this is the workflow the model work builds on.

All commands run from the repo root. Every downloaded or generated file is gitignored.

[WORKFLOW.md](WORKFLOW.md) explains what to rerun after a change, how runs are scored, where to work on
the model, and what is still open.

## Prerequisites

1. The `m5` extra: `pip install -e ".[m5]"` (adds `optax`, `tqdm`, `scikit-learn`).
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

# 4. Quick loop: fit 3,000 random series (seed 2026) at the pre-validation origin only, with a
#    registered model (see Models below), and score them. About 20 s and a ~3.3 GiB peak. Run it once
#    without --model, i.e. with the default ssp_level_dow_yeojohnson_halfnormal (#118), as the sample
#    baseline, then once per candidate.
python m5_run.py --n-demo 3000 --origins prevalidation
python m5_run.py --model <candidate> --n-demo 3000 --origins prevalidation

# 5. Compare a run with the sample baseline: overall and per-level WRMSSE and bias deltas for each
#    window both runs scored. Refuses runs fitted on different series. output/runs.csv lists the runs.
python m5_compare.py output/<baseline timestamp> output/<timestamp>

# 6. Full run: fit all 30,490 series at all three origins, write submission.csv and score it.
python m5_run.py --model <candidate> --n-demo 0
#    About 14 minutes for the default model (8 for ssp_dow) and a ~23 GiB peak (see below). Its
#    validation and evaluation scores are read here, once per candidate. With less RAM, register an
#    entry that fits in runs of fewer series, e.g. "ssp_dow_8k": SspDow(fit_chunk=8192) in
#    m5_models.py, and pass --model ssp_dow_8k.

# 7. Optional: re-score a submission file on either window.
python m5_score_submission.py --submission output/<timestamp>/submission.csv --window validation
python m5_score_submission.py --submission output/<timestamp>/submission.csv --window evaluation

# 8. Optional: submit to Kaggle (public leaderboard = _validation rows, private = _evaluation rows),
#    then type the two scores into the run's row of output/runs.csv.
kaggle competitions submit -c m5-forecasting-accuracy -f output/<timestamp>/submission.csv -m "<message>"
kaggle competitions submissions -c m5-forecasting-accuracy | head -3

# 9. Optional: diagnose the level bias. Fits each model on the quick-loop sample at the pre-validation
#    origin (--window validation for the validation one) and compares its forecasts with the arithmetic
#    and geometric means of the sales, per department, volume tercile and zero-share bin, and logs the
#    share of series with a sigma on a prior bound. About 20 s per model (30 s with a power transform);
#    writes output/bias_<timestamp>/.
python m5_bias.py --model ssp_dow ssp_level_dow
```

[WORKFLOW.md](WORKFLOW.md), "The experiment loop", walks through the cycle these steps make up: develop,
smoke test, quick fit, compare, full run, submit.

### `m5_run.py` flags

| Flag | Default | Effect |
|---|---|---|
| `--model NAME` | `ssp_level_dow_yeojohnson_halfnormal` | The registered entry of `MODELS` to fit (see Models below) |
| `--n-demo N` | 10 | Number of series to fit; 0 fits all 30,490 |
| `--sample-mode random\|top` | `random` | Draw the `--n-demo` series at random, or take the top ones by total volume |
| `--seed N` | 2026 | Seed of the random sample |
| `--origins prevalidation\|validation\|evaluation\|both\|all` | `all` | The forecast origins to fit; `both` is validation and evaluation, the two a submission needs |
| `--round` | off | Clip the forecasts at 0 and round them to the nearest integer before they are plotted, saved, submitted and scored |
| `--n-plot-ranked N` | 10 | Series in each ranked section of `forecasts.pdf` per origin; 0 turns the ranked sections off |
| `--n-plot-random N` | 30 | Series picked at random for `forecasts.pdf` per origin, the same at every origin; 0 = none |
| `--plots-per-page N` | 5 | Forecast panels per page of `forecasts.pdf` |
| `--data-dir DIR` | `playground/resource/m5-forecasting-accuracy/` | Directory with `m5_ds.nc` and `sample_submission.csv` |
| `--output-dir DIR` | `playground/m5_prototype/output/` | Root of the run directories and of `runs.csv` |

The quick loop fits only the pre-validation origin, so it never looks at the two windows Kaggle scores.
Choose models on the pre-validation score, and read the validation and evaluation scores once per
candidate, on its full run (see [WORKFLOW.md](WORKFLOW.md), "The holdout rule"). A sample score compares
only with runs on the same sample (see "Comparing on a sample" there).

`--round` is off by default, because WRMSSE is a squared error and the best forecast is the mean, which is
rarely an integer. The results are in [WORKFLOW.md](WORKFLOW.md), "Integer rounding (#117)".

### What a run writes

Each `m5_run.py` run writes to `output/<timestamp>/`:

| File | Contents |
|---|---|
| `submission.csv` | 60,980 rows: `_validation` rows from the fit through d_1913, `_evaluation` rows from the fit through d_1941 (full runs with the validation and evaluation origins only; it has no pre-validation rows) |
| `wrmsse_per_level.csv` | 12-level WRMSSE and bias table, one `window` per fitted origin |
| `run_stats.json` | Model name and settings, origins, whether the forecasts were rounded (`round`), fit time (total and per origin), total time, peak memory |
| `batch_params.npz` | `fit_idx` (the fitted series) and the model's fitted outputs and forecasts of each fitted origin; pre-validation- and evaluation-origin arrays carry a `_prevalidation` or `_evaluation` suffix |
| `diagnostics.png` | Histograms of the model's fitted outputs at the first origin fitted (pre-validation by default) |
| `forecasts.pdf` | Forecast review of each fitted origin: the total, store, category and department sums of the fitted series, then four ranked sections of `--n-plot-ranked` series each (default 10): the worst and best RMSSE, and the worst and best contribution (weight × RMSSE, the series' term in the level-12 WRMSSE; series of weight 0 are left out), then `--n-plot-random` series at random (default 30, the same at every origin). 0 turns those sections off. Each panel shows the last 90 fit days, the actuals and the forecast over the forecast window, and the model's in-sample fit if it returns one; its title gives the RMSSE, bias and weight, and for a single series its contribution and share of the level-12 score |

and appends one row to the run log `output/runs.csv`: the timestamp, the run directory, the model and
its settings, the number of series, sample mode and seed, the origins, whether the forecasts were rounded
(`round`), the WRMSSE of each scored window (`wrmsse_prevalidation`, `wrmsse_validation`,
`wrmsse_evaluation`), the Kaggle scores (`kaggle_public`, `kaggle_private`), the fit time, the git commit
and whether tracked files had uncommitted changes (`git_dirty`). After a Kaggle submission, type its scores
into the run's `kaggle_public` and `kaggle_private` columns; later runs keep them. The columns follow the
latest run's order, so a column added later is blank for earlier runs.

A sampled run aggregates and weights only its own series, so its WRMSSE is comparable with other runs
on the same sample (same `--n-demo`, `--sample-mode` and `--seed`), not with a full-run score.
`m5_compare.py` checks that two runs fitted the same series before it compares them.

### Forecast origins

| Origin | Fit on | Forecasts | Submission rows | Kaggle leaderboard |
|---|---|---|---|---|
| prevalidation | d_1–d_1885 | d_1886–d_1913 | none | none: choose models on this one |
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
| `ssp_dow` | Weekday dummies from Tuesday to Sunday, no level state; MAP fit by Adam. The baseline, the default until #118, and the model of the #93 and #94 Kaggle submissions | `level=False`, `transform=LogTransform()`, `sigma_h_prior=SIGMA_H_PRIOR` (U(0.1, 0.5)), `sigma_q_prior=SIGMA_Q_PRIOR` (U(0.01, 0.1)), `n_iter=100`, `lr=0.03`, `chunk_size=4096`, `fit_chunk=None` |
| `ssp_level_dow` | `ssp_dow` with a level state: a column of ones leads the weekday dummies, so 7 states (#106) | `level=True`, otherwise as `ssp_dow` |
| `ssp_dow_boxcox_shift`, `ssp_level_dow_boxcox_shift` | `ssp_dow`, `ssp_level_dow` fitted to a Box-Cox power of `sales + 0.5`, λ fitted per series (#115) | `transform=PowerTransform("box-cox", shift=0.5)` |
| `ssp_dow_boxcox_floor`, `ssp_level_dow_boxcox_floor` | The same with Box-Cox of `max(sales, 0.1)` | `transform=PowerTransform("box-cox", floor=0.1)` |
| `ssp_dow_yeojohnson`, `ssp_level_dow_yeojohnson` | The same with Yeo-Johnson of `sales`, which on sales is Box-Cox of `sales + 1` | `transform=PowerTransform("yeo-johnson")` |
| `ssp_level_dow_yeojohnson_wide` | `ssp_level_dow_yeojohnson` with uniform sigma priors wide enough for the standardised scale (#116) | `sigma_h_prior=SigmaPrior("uniform", (0.05, 1.5))`, `sigma_q_prior=SigmaPrior("uniform", (0.0005, 0.5))` |
| `ssp_level_dow_yeojohnson_halfnormal` | The same with soft half-normal priors, no bound. The default since #118: best on the 3,000-series sample (0.74610), on the full run at all three origins and on Kaggle (0.70888 public, 0.79394 private) | `sigma_h_prior=SigmaPrior("half_normal", (1.0,))`, `sigma_q_prior=SigmaPrior("half_normal", (0.05,))` |
| `ssp_level_dow_yeojohnson_lognormal` | The same with soft log-normal priors | `sigma_h_prior=SigmaPrior("log_normal", (log(0.8), 0.5))`, `sigma_q_prior=SigmaPrior("log_normal", (log(0.02), 1.0))` |
| `ssp_level_dow_halfnormal` | `ssp_level_dow` with the half-normal priors: on the log target it forecasts 23% high | As `ssp_level_dow_yeojohnson_halfnormal`, with the log transform |

## Reference results

Sample baseline: `ssp_dow` with its default settings on 3,000 random series (seed 2026), on the same
Mac as below:

| | Value |
|---|---|
| Fit time | 16.7 s (pre-validation origin) + 17.2 s (validation origin) + 17.5 s (evaluation origin) |
| Whole run | about 20 s with `--origins prevalidation`, 60 s with all three origins |
| Peak memory | 3.2 GiB with `--origins prevalidation`, 4.4 GiB with all three origins |
| Local WRMSSE, pre-validation window | 1.00659 |
| Local WRMSSE, validation window | 0.94808 (0.93487 before the #116 fix) |
| Local WRMSSE, evaluation window | 0.90002 (0.88578 before the #116 fix) |

Full run of `ssp_dow` with its default settings on an 18-core, 128 GB Mac, JAX on CPU:

| | Value |
|---|---|
| Fit time | 152.2 s (pre-validation origin) + 161.2 s (validation origin) + 161.1 s (evaluation origin); 496 s for the whole run |
| Peak memory | 22.7 GiB (13.8 GiB after the first origin: later fits do not release all of the earlier ones' memory) |
| Local WRMSSE, pre-validation window | 1.05983 |
| Local WRMSSE, validation window | 1.10175 (1.08136 before the #116 fix, the #94 submission) |
| Local WRMSSE, evaluation window | 0.99403 (0.97573 before the #116 fix) |

Full run of the default model, `ssp_level_dow_yeojohnson_halfnormal`, on the same Mac (#118):

| | Value |
|---|---|
| Fit time | 259.0 s (pre-validation origin) + 269.9 s (validation origin) + 269.5 s (evaluation origin); 820 s for the whole run |
| Peak memory | 22.9 GiB |
| Local WRMSSE, pre-validation window | 0.79563 |
| Local WRMSSE, validation window | 0.70891 (Kaggle public 0.70888) |
| Local WRMSSE, evaluation window | 0.79394 (Kaggle private 0.79394) |

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
| `m5_bias.py` | Which mean of the sales a model's forecast tracks, per department, volume tercile and zero-share bin |
| `m5_score_submission.py` | 12-level WRMSSE of a submission for one window |
| `m5_check_weights.py` | Compare the scorer's weights with the official files |
| `m5_submission_diff.py` | Diagnostic diff of two submissions |
| `m5_make_accuracy_fixture.py` | Build the committed CI fixture `src/bunobee/datasets/data/m5_aggregate.csv` |
