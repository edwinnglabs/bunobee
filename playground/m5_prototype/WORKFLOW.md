# M5 workflow

How the M5 pipeline fits together, what to rerun after a change, and where to work on the model. For
prerequisites, every command and the reference results, see [README.md](README.md).

## The pipeline

```
 one-time setup                        every experiment
 ──────────────                        ────────────────
 m5_fetch.py  →  m5_preprocess.py  →   m5_run.py  →  kaggle submit (optional)
 9 CSVs          m5_ds.nc              fit, forecast, score, submission.csv
```

| Step | Writes | Rerun when |
|---|---|---|
| `m5_fetch.py` | 9 CSVs in `playground/resource/m5-forecasting-accuracy/` | A file is missing (it skips files already there) or corrupted (`--force`) |
| `m5_preprocess.py` | `m5_ds.nc` in the same directory | You change `m5_preprocess.py`, e.g. to add a variable. Pass `--force`: on its own the script rebuilds only when the cache lacks the test-window actuals |
| `m5_check_weights.py` | Nothing; prints a comparison | You change how the scorer computes weights |
| `m5_run.py` | `output/<timestamp>/` and a row of `output/runs.csv` | Every experiment |
| `m5_compare.py` | Nothing; prints the deltas | You want to compare a run with a baseline run on the same series |
| `m5_score_submission.py` | Nothing; prints the score | You want to re-score a saved `submission.csv`, e.g. after changing the scorer |
| `kaggle competitions submit` | A late submission on Kaggle | You want the official score |

`bunobee` is installed in editable mode (`pip install -e ".[m5]"`), so edits under `src/` take effect on
the next `m5_run.py` without a reinstall.

## The data

`m5_ds.nc` holds every series over every day:

- `sales`: 30,490 series × 1,969 days. One series is one item in one store.
- `sell_price`, `snap` (whether the state's SNAP benefits apply that day), and calendar fields: day of
  week, day of month, month, and event names and types.
- The hierarchy of each series: item → department → category, and store → state.
- A `split` label per day:

| Split | Days |
|---|---|
| `train` | d_1–d_1913 |
| `validation` | d_1914–d_1941 |
| `test` | d_1942–d_1969 |

## A run

Kaggle wants two 28-day forecasts, so by default (`--origins both`) `m5_run.py` fits the model twice,
once per cutoff day:

```
d_1 ─────────────────────────── d_1913 │ d_1914 ── d_1941 │ d_1942 ── d_1969
 fit (validation origin) ───────────────┤ forecast →       │
 fit (evaluation origin) ──────────────────────────────────┤ forecast →
                                         _validation rows    _evaluation rows
                                         Kaggle public       Kaggle private
```

For each cutoff day it hands the model picked with `--model` (by default `ssp_dow`) all sales up to that
day. The model builds its own inputs, fits all 30,490 series at once and forecasts the next 28 days. It
then writes `submission.csv` (60,980 rows, half from each cutoff day), scores both halves against the true
sales and saves everything to `output/<timestamp>/`.

`--n-demo 3000 --origins validation` fits 3,000 random series at the first cutoff day only, in about
20 s. A sampled run aggregates and weights its own series, so its score can be compared with another run
on the same sample (`m5_compare.py` checks that), not with a full run's.

## The score

WRMSSE, the competition metric, as computed by `m5_score_submission.py`:

1. **Per series:** the 28-day forecast error is divided by how much that series moves from one day to the
   next over its training history. A score of 1.0 means the forecast error equals the error of
   "predict each day with the day before", so lower is better.
2. **12 levels:** the same is done at 12 levels, from total sales down to one item in one store.
3. **Weights:** within a level, series are weighted by their dollar sales over the last 28 training days.
   The final score is the plain average of the 12 levels.

Every level counts equally, so the store and department totals matter as much as the 30,490 items.

In `output/<timestamp>/`, `wrmsse_per_level.csv` gives the score of each level and a bias column showing
whether forecasts run high or low. That's where to look for what to fix. `forecasts.pdf` plots forecast
against actual for 50 series.

## The experiment loop

1. Add or edit a model entry in `m5_models.py` (see below).
2. `python m5_run.py --model NAME --n-demo 50` to check nothing crashes.
3. `python m5_run.py --model NAME --n-demo 0` for the real score: about 5.5 minutes, ~22 GiB peak.
4. Compare with the baseline run: `python m5_compare.py output/<baseline> output/<run>`.
5. Submit only when you want the official number: the local scorer matches Kaggle to within 0.0001.

Make decisions on the validation score only, and keep the evaluation score as the final check. Both
windows' actuals are on disk, so it is easy to tune to both without noticing, and then the private score
stops telling you anything.

Baseline, the current model (`ssp_dow`):

| | Validation (public) | Evaluation (private) |
|---|---|---|
| Local | 1.08136 | 0.97573 |
| Kaggle | 1.08127 | 0.97572 |

## Working on the model

### Where the code is

| File | What to change there |
|---|---|
| `src/bunobee/models/m5/m5_ssp_optim.py` | `fit_batch_series_opt`: the transform, parameters, priors, loss and optimizer. `predict_batch_series_opt`: the forecast and back-transform |
| `playground/m5_prototype/m5_models.py` | The registry `MODELS`. An entry's `fit_forecast` builds the model's inputs for one cutoff day (`SspDow` calls `make_periodic_dummies`) and calls the model. Settings such as `n_iter`, `lr` and `fit_chunk` are the entry's fields |
| `src/bunobee/regression.py` | `make_periodic_dummies`, which builds the weekday dummies, the model's only input today |
| `src/bunobee/models/ssp/kalman_1d.py` | `kalman_filter_1d`, the filter inside the loss |

An entry's `fit_forecast(ds, series_idx, fit_mask, forecast_mask)` gets the whole dataset, the series to
fit and the fit and forecast days. It returns forecasts (series × 28) with no NaNs and a dict of per-series
arrays, which `m5_run.py` saves to `batch_params.npz` and plots in `diagnostics.png`. Everything else
(data loading, both cutoff days, the submission, the scoring) lives in `m5_run.py` and works with any model,
so adding a model needs only a new entry.

### The current model (`ssp_dow`)

- **Transform:** it models `log(max(sales, 0.1) / mean)` with Gaussian noise, where `mean` is the series'
  mean of `max(sales, 0.1)` over the fit days. Every zero-sales day becomes `log(0.1 / mean)`.
- **Structure:** `y_t = Σ_i Z[t, i] · a[t, i] + noise`. The columns of `Z` are weekday indicators for
  Tuesday to Sunday (Monday is dropped), and each state `a_i` drifts as a random walk.
- **Parameters:** 3 per series: the noise level, the drift of the first state and a drift shared by the
  other five. Uniform priors; 100 Adam steps at learning rate 0.03.
- **Forecast:** the last filtered state times the future `Z`, back-transformed as a lognormal mean and
  rescaled by `mean`.

### Known problems

- **There is no level state.** The docstrings call the model "local level + weekly seasonality", but `Z`
  has no column of ones. Monday rows of `Z` are all zero, so Monday sales never update the states and every
  Monday forecast is `mean · exp(σ_h² / 2)`: the series' average over all fit days, whatever its recent
  sales. In the baseline run, all four Monday forecasts of every series equal that value. The "level"
  drift actually applies to the Tuesday state.
- **Totals are forecast too low.** In the baseline run, total sales are 12.7% low on the validation window
  and 9.3% low on the evaluation window, and every level from 1 to 9 is low by a similar amount. The item
  levels (10–12) are not. Untested candidates: the missing level state (an all-time average sits below
  recent sales if sales grew), the 0.1 clip on zero days, and the back-transform.
- **Unused data:** `sell_price`, `snap`, events, day of month and month are in `m5_ds.nc` but not in `Z`.
  New inputs are new columns of `Z`, so each becomes a state; `fit_batch_series_opt` gives every column
  after the first the shared drift.
- **No sharing between series:** each series is fitted on its own, which leaves sparse series with noisy
  parameters.

### Open items

- **Hierarchical parameter sharing:** share the weekday drift within department × store groups (70 groups),
  either outright or through a prior that pulls each series toward its group. Most useful for sparse
  series.
- **No test that the batch fit is correct:** nothing checks that `fit_batch_series_opt` matches the
  single-series `fit_one_series_opt` on the same series.
- **float32 log-likelihood:** the loss sums 1,913 steps in float32. Summing in float64 has not been tried.
- **Memory:** the second fit does not release the first one's memory, so a full run peaks at 22.3 GiB
  instead of about 14.4 GiB. The `fit_chunk` setting of `SspDow` caps it meanwhile.
- **Optimizer loop:** a Python loop over a compiled step, not `lax.scan`. A cutoff day already fits in
  about 159 s, so this only matters for speed.
- **Multiple devices:** no `pmap`; only useful with several GPUs.
