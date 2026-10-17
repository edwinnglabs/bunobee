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
| `m5_bias.py` | `output/bias_<timestamp>/` | You change the transform or the back-transform and want to see which mean of the sales the forecast tracks |
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
whether forecasts run high or low. That's where to look for what to fix. `forecasts.pdf` shows where: for
each cutoff day, the total, store and department sums of the fitted series, then 50 series picked at random,
by worst RMSSE or by top weight (`--plot-select random|worst|top`). Each panel draws the actuals, the
model's in-sample fit and the forecast, with the RMSSE, bias and weight in its title.

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

### `ssp_level_dow` (#106)

`ssp_dow` with the missing level state, run through the loop against `ssp_dow` on the same series: the
3,000-series sample (seed 2026) and a full run, both origins. A negative delta means `ssp_level_dow` scores
better.

| WRMSSE | `ssp_dow` | `ssp_level_dow` | Delta |
|---|---|---|---|
| Sample, validation | 0.93487 | 0.94145 | +0.00657 |
| Sample, evaluation | 0.88578 | 0.90137 | +0.01559 |
| Full, validation | 1.08136 | 1.05052 | −0.03084 |
| Full, evaluation | 0.97573 | 0.98431 | +0.00858 |
| Kaggle public | 1.08127 | 1.05043 | −0.03084 |
| Kaggle private | 0.97572 | 0.98431 | +0.00859 |

Per level (`m5_compare.py`):

- **Full run, validation:** levels 1–11 improve by 0.009–0.047 and level 12 (item × store) worsens by
  0.019. Total sales are forecast 10.3% low instead of 12.7%.
- **Full run, evaluation:** only levels 3 (store) and 8 (store × category) improve. The item levels worsen
  most (10–12: +0.005, +0.027, +0.039). Total sales are 3.8% low instead of 9.3%.
- **Sample:** worse on both windows at most levels, with the same lift in the totals (bias −9.6% → −7.6%
  on validation, −9.2% → −3.7% on evaluation).

The full run fits in 361 s against 334 s for `ssp_dow` and peaks at 24.8 GiB against 22.3 GiB.

**Decision:** `ssp_dow` stays the default `--model`. The full run's validation gain does not hold on the
evaluation window, and the sample scores `ssp_level_dow` worse on both windows, so the gain is not robust.
The level lifts the totals, but only by 2.4 points of the 12.7% shortfall on validation, so it is not the
main cause of the low totals. `ssp_level_dow` stays registered as the starting point for the next
experiments.

### Level bias diagnosis (#113)

`m5_bias.py --model ssp_dow ssp_level_dow` fits both entries on the 3,000-series sample at the validation
origin and compares each series' forecast with the arithmetic mean (am) and the geometric mean (gm) of
`max(sales, 0.1)` over the last 28 fit days. Ratios are ratios of sums over a group's series, so the bias is
the group's forecast total over its actual total, minus one. `slope` measures how the forecast moves with the
gap between the two means across a group's series: 0 if it is the geometric mean times a constant, 1 if it is
the arithmetic mean times a constant.

| Group | Bias, `ssp_dow` | Bias, `ssp_level_dow` | act/am | fc/am | fc/gm | am/gm | slope |
|---|---|---|---|---|---|---|---|
| Total | −9.6% | −7.6% | 0.973 | 0.879 | 1.362 | 1.549 | 0.24 |
| HOBBIES_1 | −20.5% | −17.4% | 0.961 | 0.764 | 1.607 | 2.102 | 0.26 |
| Low-volume tercile | −17.3% | −13.5% | 0.843 | 0.698 | 1.342 | 1.923 | 0.02 |
| Mid-volume tercile | −24.7% | −21.0% | 0.966 | 0.727 | 1.471 | 2.023 | 0.07 |
| High-volume tercile | −5.1% | −3.7% | 0.989 | 0.938 | 1.344 | 1.433 | 0.41 |

The ratios and the slope are those of `ssp_dow`. Volume terciles split the sample into thirds by mean sales
over the fit days.

- **The forecast tracks the geometric mean.** It is the recent geometric mean times a lift that barely
  depends on the gap to the arithmetic mean (slope 0.24 on the total, 0.02–0.41 by tercile; 0.25 for
  `ssp_level_dow`). The lift is the lognormal correction, with `sigma_h` on its 0.5 cap for 96% of series.
  Where zero days drive the means apart, the lift can't close the gap: series with zeros on under a quarter
  of their days since their first sale are forecast 4% high, those with zeros on half to three quarters 23%
  low.
- **The recent arithmetic mean is close to the actuals:** they are 0.97 of it on the total and 0.96 on
  HOBBIES_1, so a forecast that tracked it would be about 3–4% off instead of 10–20% low. On the
  low-volume tercile the 0.1 floor on zero days puts the mean 19% above the actuals; without the floor it
  is 8% below them.
- **The level state (#106) removes about a fifth of the bias:** 2.0 of the 9.6 points on the total, 3.1 of
  20.5 on HOBBIES_1 and 1.4–3.8 by tercile. It does not change which mean the forecast tracks.
- **Means over all fit days can't show which mean the forecast tracks.** Half the sample has more than 100
  zero days before its item was first stocked (median 193). They pull the all-day means, the geometric one
  most, far below the level the forecast starts from: the actuals are 1.17 times the all-day mean of
  `max(sales, 0.1)`, which is the model's `response_norm`.

So the log transform is the main cause of the low totals and the missing level a minor one; the transform
work (#114, #115) and the `sigma_h` cap (#116) are where the bias can be closed.

## Working on the model

### Where the code is

| File | What to change there |
|---|---|
| `src/bunobee/models/m5/m5_ssp_optim.py` | `fit_batch_series_opt`: the parameters, priors, loss and optimizer. `predict_batch_series_opt`: the forecast, back-transformed by the fit's transform |
| `src/bunobee/models/m5/transforms.py` | The target transforms: the `Transform` protocol (`fit` per-series parameters on the fit days, `forward`, and `inverse` with its bias correction) and `LogTransform`, the default. A new transform goes here; an entry picks it with its `transform` field, without an edit of the fitter |
| `playground/m5_prototype/m5_models.py` | The registry `MODELS`. An entry's `fit_forecast` builds the model's inputs for one cutoff day (`SspDow` calls `make_periodic_dummies`) and calls the model. Settings such as `transform`, `n_iter`, `lr` and `fit_chunk` are the entry's fields |
| `src/bunobee/regression.py` | `make_periodic_dummies`, which builds the weekday dummies and the column of ones, the models' only inputs today |
| `src/bunobee/models/ssp/kalman_1d.py` | `kalman_filter_1d`, the filter inside the loss |

An entry's `fit_forecast(ds, series_idx, fit_mask, forecast_mask)` gets the whole dataset, the series to
fit and the fit and forecast days. It returns forecasts (series × 28) with no NaNs and a dict of per-series
arrays, which `m5_run.py` saves to `batch_params.npz` and plots in `diagnostics.png`. The dict can also hold
the model's in-sample fit (series × fit days) under `insample`, which `forecasts.pdf` draws; `ssp_dow`
returns each fit day's one-step-ahead prediction, back-transformed as its forecast is. Everything else
(data loading, both cutoff days, the submission, the scoring) lives in `m5_run.py` and works with any model,
so adding a model needs only a new entry.

### The current model (`ssp_dow`)

- **Transform:** it models `log(max(sales, 0.1) / mean)` with Gaussian noise, where `mean` is the series'
  mean of `max(sales, 0.1)` over the fit days. Every zero-sales day becomes `log(0.1 / mean)`. This is
  `LogTransform()`, the default `transform` of `SspDow`.
- **Structure:** `y_t = Σ_i Z[t, i] · a[t, i] + noise`. The columns of `Z` are weekday indicators for
  Tuesday to Sunday (Monday is dropped), and each state `a_i` drifts as a random walk.
- **Parameters:** 3 per series: the noise level, the drift of the first state and a drift shared by the
  other five. Uniform priors; 100 Adam steps at learning rate 0.03.
- **Forecast:** the last filtered state times the future `Z`, back-transformed as a lognormal mean and
  rescaled by `mean`.

`ssp_level_dow` is the same model with `level=True`: a column of ones leads `Z`, so it has 7 states. The
first is the level, with its own drift; the six weekday states are offsets from Monday and share the other
drift.

### Known problems

- **`ssp_dow` has no level state.** The docstrings call the model "local level + weekly seasonality", but
  its `Z` has no column of ones. Monday rows of `Z` are all zero, so Monday sales never update the states
  and every Monday forecast is `mean · exp(σ_h² / 2)`: the series' average over all fit days, whatever its
  recent sales. In the baseline run, all four Monday forecasts of every series equal that value. The
  "level" drift actually applies to the Tuesday state. `ssp_level_dow` adds the level (see its results
  above), but is not the default.
- **Totals are forecast too low.** In the baseline run, total sales are 12.7% low on the validation window
  and 9.3% low on the evaluation window, and every level from 1 to 9 is low by a similar amount. The item
  levels (10–12) are not. The missing level state explains only part of it: `ssp_level_dow` lifts the
  totals by 2.4 points on validation and 5.6 on evaluation. The main cause is the log transform: the
  forecast tracks the geometric mean of `max(sales, 0.1)`, and the lognormal correction, with `sigma_h`
  capped, lifts it only part of the way to the arithmetic mean (see "Level bias diagnosis (#113)").
- **Unused data:** `sell_price`, `snap`, events, day of month and month are in `m5_ds.nc` but not in `Z`.
  New inputs are new columns of `Z`, so each becomes a state; `fit_batch_series_opt` gives every column
  after the first the shared drift.
- **No sharing between series:** each series is fitted on its own, which leaves sparse series with noisy
  parameters.

### Open items

- **The sample did not predict the full run:** for `ssp_level_dow`, the 3,000-series sample scored the
  validation window +0.0066 worse while the full run scored it −0.0308 better. Enlarge the sample or
  stratify it by store × department before steering the next model by it.
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
