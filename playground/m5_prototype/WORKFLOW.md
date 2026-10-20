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
| `m5_bias.py` | `output/bias_<timestamp>/` | You change the transform or the back-transform and want to see which mean of the sales the forecast tracks, or how many fitted sigmas sit on a prior bound |
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

Kaggle wants two 28-day forecasts, and a third cutoff day 28 days before the first gives a window Kaggle
never scores (#125). By default (`--origins all`) `m5_run.py` fits the model three times, once per cutoff
day:

```
d_1 ────────────── d_1885 │ d_1886 ── d_1913 │ d_1914 ── d_1941 │ d_1942 ── d_1969
 fit (pre-validation) ─────┤ forecast →       │                  │
 fit (validation origin) ─────────────────────┤ forecast →       │
 fit (evaluation origin) ────────────────────────────────────────┤ forecast →
                            no submission rows  _validation rows   _evaluation rows
                            not on Kaggle       Kaggle public      Kaggle private
```

For each cutoff day it hands the model picked with `--model` (by default `ssp_level_dow_yeojohnson_halfnormal`,
#118) all sales up to that day. The model builds its own inputs, fits all 30,490 series at once and forecasts the next 28 days. It
then writes `submission.csv` (60,980 rows, half from each of the two Kaggle cutoff days), scores all three
windows against the true sales and saves everything to `output/<timestamp>/`. The pre-validation window's
actuals are in the `train` split, so it needs nothing beyond the Kaggle files.

`--n-demo 3000 --origins prevalidation` fits 3,000 random series at the pre-validation cutoff day only, in
about 20 s. A sampled run aggregates and weights its own series, so its score can be compared with another
run on the same sample (`m5_compare.py` checks that), not with a full run's.

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
each cutoff day, the total, store, category and department sums of the fitted series, then the 10 series with
the worst RMSSE, the 10 with the best, the 10 with the worst contribution (weight × RMSSE, what a series costs
the item-level score once its dollar sales are counted), the 10 with the best, and 30 series at random
(`--n-plot-ranked`, `--n-plot-random`). Each panel draws the actuals, the model's in-sample fit and the
forecast, with the RMSSE, bias, weight and contribution in its title.

## The experiment loop

1. Add or edit a model entry in `m5_models.py` (see below).
2. `python m5_run.py --model NAME --n-demo 50` to check nothing crashes.
3. Quick loop: `python m5_run.py --model NAME --n-demo 3000 --origins prevalidation`, then compare with the
   sample baseline: `python m5_compare.py output/<baseline> output/<run>`.
4. `python m5_run.py --model NAME --n-demo 0` for the full score at all three origins: about 8 minutes for
   `ssp_dow`, ~23 GiB peak. Compare it with the full baseline run the same way.
5. Submit only when you want the official number: the local scorer matches Kaggle to within 0.0001.

Make decisions on the pre-validation score only, and read the validation and evaluation scores once per
candidate, on its full run, as the final check. Those two windows are Kaggle's public and private
leaderboards, and the local scores match Kaggle's, so a choice made on either is tuned on the test: their
actuals are on disk, so it is easy to tune to them without noticing, and then the leaderboard stops telling
you anything. Until #125 the quick loop ran the validation window, so every choice from #106 to #117 was
made on it, and every full run printed the evaluation score.

Baseline, `ssp_dow`, the default model until #118:

| | Pre-validation | Validation (public) | Evaluation (private) |
|---|---|---|---|
| Local, #94 submission | not run | 1.08136 | 0.97573 |
| Kaggle, #94 submission | not scored | 1.08127 | 0.97572 |
| Local, after the #116 fix | 1.05983 | 1.10175 | 0.99403 |

The #116 fix of the best-parameter tracking changed `ssp_dow`'s forecasts (see "Sigma priors and best-parameter
tracking (#116)"); compare new full runs with the last row. The default model is now
`ssp_level_dow_yeojohnson_halfnormal`, which scores 0.79563, 0.70891 and 0.79394 on the three windows (Kaggle 0.70888
and 0.79394); see "Full run and the default model (#118)".

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

### Power transforms (#115)

`PowerTransform` (`transforms.py`) fits each series to a Box-Cox or Yeo-Johnson power of its sales, with λ
fitted per series on the fit days by scikit-learn's `PowerTransformer` and the powered sales standardised, as
`PowerTransformer` does by default. Three variants are registered, each on `ssp_dow` and on `ssp_level_dow`
(`ssp_dow_yeojohnson`, `ssp_level_dow_yeojohnson`, ...):

| Suffix | Transform | A zero-sales day before the power |
|---|---|---|
| `boxcox_shift` | Box-Cox of `sales + 0.5` | 0.5 |
| `boxcox_floor` | Box-Cox of `max(sales, 0.1)`, the floor of the log target | 0.1 |
| `yeojohnson` | Yeo-Johnson of `sales`. On sales ≥ 0 it is Box-Cox of `sales + 1`, with the same λ | 1 |

The M5 sales forced two choices:

- **λ is bounded to [−1, 2]**, the range of R's `forecast::BoxCox.lambda`. The zero days make the unbounded
  maximum-likelihood λ of most series strongly negative: median −1.2 for the floor and −3.3 for Yeo-Johnson,
  down to −134. Such a power maps every day with sales to almost one value, and its inverse returns NaN.
  Bounded, λ sits on −1 for 54% (floor), 67% (shift) and 75% (Yeo-Johnson) of the 3,000 series.
- **The inverse is the mean of the censored sales.** The inverse power of the forecast mean is a median. With
  λ < 0 the power is bounded above, so a Gaussian forecast puts mass on infinite sales. The inverse censors
  each day's sales to the series' range, 0 to its largest fit-day sales, and takes their mean by quadrature,
  within 2e-5 of adaptive quadrature on the forecasts of 600 sample series (Gauss-Hermite nodes were up to
  35% off).
  Forecasts are never negative.

Results on the 3,000-series sample (seed 2026) at the validation origin, against `ssp_dow`; a negative delta
is better. Bias is the forecast total over the actual total, minus one. Position and slope are `m5_bias.py`'s
over the last 28 fit days (#113): 0 if the forecast follows the geometric mean of `max(sales, 0.1)`, 1 if it
follows the arithmetic mean.

| Entry | WRMSSE | Delta | Bias | Position | Slope | Fit |
|---|---|---|---|---|---|---|
| `ssp_dow` | 0.93487 | | −9.6% | 0.71 | 0.24 | 17 s |
| `ssp_dow_boxcox_shift` | 1.19569 | +0.26081 | −5.8% | 0.80 | 0.90 | 31 s |
| `ssp_dow_boxcox_floor` | 1.27161 | +0.33674 | +1.0% | 0.96 | 0.35 | 32 s |
| `ssp_dow_yeojohnson` | 1.12406 | +0.18919 | −6.8% | 0.78 | 0.78 | 26 s |
| `ssp_level_dow` | 0.94145 | +0.00657 | −7.6% | 0.76 | 0.25 | 20 s |
| `ssp_level_dow_boxcox_shift` | 0.89975 | −0.03512 | +2.3% | 0.99 | 1.00 | 35 s |
| `ssp_level_dow_boxcox_floor` | 1.09899 | +0.16412 | +11.4% | 1.18 | 0.48 | 35 s |
| `ssp_level_dow_yeojohnson` | **0.83171** | **−0.10316** | −0.4% | 0.93 | 0.85 | 29 s |

Per level (`m5_compare.py`):

| Level | `ssp_dow` | `_boxcox_shift` | `_boxcox_floor` | `_yeojohnson` | `ssp_level_dow` | `_boxcox_shift` | `_boxcox_floor` | `_yeojohnson` |
|---|---|---|---|---|---|---|---|---|
| 1 Total | 0.8926 | 1.3043 | 1.4416 | 1.2066 | 0.8946 | 0.7672 | 1.1732 | 0.6690 |
| 2 State | 0.9021 | 1.2254 | 1.3399 | 1.1438 | 0.8910 | 0.7845 | 1.1135 | 0.7066 |
| 3 Store | 0.9468 | 1.2366 | 1.3178 | 1.1640 | 0.9572 | 0.9083 | 1.1139 | 0.8413 |
| 4 Category | 0.8945 | 1.2963 | 1.3825 | 1.1960 | 0.8879 | 0.7797 | 1.1158 | 0.6881 |
| 5 Department | 0.9834 | 1.3639 | 1.4309 | 1.2546 | 1.0025 | 0.9160 | 1.1544 | 0.8138 |
| 6 State × category | 0.9056 | 1.2121 | 1.2946 | 1.1296 | 0.9114 | 0.8376 | 1.0855 | 0.7660 |
| 7 State × department | 0.9494 | 1.2632 | 1.3330 | 1.1734 | 0.9719 | 0.9470 | 1.1230 | 0.8681 |
| 8 Store × category | 0.9311 | 1.1743 | 1.2430 | 1.1084 | 0.9546 | 0.9331 | 1.0816 | 0.8760 |
| 9 Store × department | 0.9676 | 1.1764 | 1.2345 | 1.1126 | 0.9820 | 0.9892 | 1.1043 | 0.9302 |
| 10 Item | 0.9534 | 1.0448 | 1.0968 | 1.0105 | 0.9499 | 0.9791 | 1.0473 | 0.9399 |
| 11 Item × state | 0.9456 | 1.0274 | 1.0747 | 0.9963 | 0.9466 | 0.9757 | 1.0361 | 0.9390 |
| 12 Item × store | 0.9466 | 1.0235 | 1.0700 | 0.9930 | 0.9477 | 0.9796 | 1.0394 | 0.9425 |
| Bias, levels 1–9 | −9.6 to −10.6% | −5.8 to −7.3% | −0.2 to +1.0% | −6.8 to −8.1% | −7.6 to −8.5% | +1.3 to +2.3% | +10.8 to +11.6% | −0.3 to −1.1% |
| Bias, levels 10–12 | +3.8 to +6.6% | +5.6 to +8.8% | +14.3 to +16.9% | +5.7 to +9.3% | +2.8 to +5.2% | +12.3 to +14.2% | +24.1 to +26.1% | +10.5 to +12.2% |

The bias of levels 10–12 is the dollar-weighted mean of each series' relative bias, so it is not the total's.

Share of series with a fitted sigma within 1% of its prior's width of a bound, or past it (`m5_bias.py`,
`bounds.csv`; the priors are `sigma_h` ~ U(0.1, 0.5) and each drift ~ U(0.01, 0.1)). No `sigma_h` and at
most 5% of the drifts sit on a lower bound. These fits predate the #116 fix, which keeps sigmas inside their
bounds; `m5_bias.py` now counts within one Adam step of a bound instead.

| Entry | Any sigma | `sigma_h` on 0.5 | First drift on 0.1 | Shared drift on 0.1 |
|---|---|---|---|---|
| `ssp_dow` | 99.2% | 95.6% | 65.4% | 68.6% |
| `ssp_dow_boxcox_shift` | 99.9% | 99.9% | 62.2% | 74.8% |
| `ssp_dow_boxcox_floor` | 99.9% | 99.9% | 63.7% | 74.3% |
| `ssp_dow_yeojohnson` | 99.9% | 99.9% | 60.6% | 75.1% |
| `ssp_level_dow` | 99.5% | 92.2% | 71.0% | 57.2% |
| `ssp_level_dow_boxcox_shift` | 99.8% | 92.5% | 83.9% | 46.6% |
| `ssp_level_dow_boxcox_floor` | 99.8% | 92.3% | 84.2% | 46.5% |
| `ssp_level_dow_yeojohnson` | 99.9% | 92.7% | 84.2% | 46.9% |

- **Without the level state every power is worse, because of Monday.** `ssp_dow`'s Monday rows of `Z` are all
  zero, so its Monday forecast is the inverse of `mu = 0`, the centre of the transformed fit days. For the log
  target that is the arithmetic mean over all fit days. For a standardised power with λ = −1 its median is the
  harmonic mean of the shifted sales over all fit days, less the shift, which the zero days before an item
  was stocked pull far down: the Monday totals are 0.51–0.56 of the actuals. Tuesday to Sunday, `ssp_dow_boxcox_shift` and `ssp_dow_yeojohnson`
  forecast 0.95–1.03 of the actuals, against 0.84–0.92 for `ssp_dow`.
- **With the level state, Yeo-Johnson is better than `ssp_dow` at every level**: −0.103 overall, −0.224 on the
  total and −0.170 on departments. The totals are 0.4% low instead of 9.6%. Box-Cox of `sales + 0.5` improves
  levels 1–7 and worsens 8–12. Box-Cox of `max(sales, 0.1)` forecasts 11% high and is worse at every level.
- **The forecast now follows the arithmetic mean** (#113's question): the slope rises from 0.24 to 0.85 with
  `ssp_level_dow_yeojohnson` and to 1.00 with `ssp_level_dow_boxcox_shift`. The bias does not vanish but
  moves: `ssp_level_dow_yeojohnson` forecasts the low-volume tercile 29% low (`ssp_dow`: 17%) and the
  high-volume tercile 6% high (`ssp_dow`: 5% low), and the series with zeros on 75% or more of their days
  since their first sale 25% low.
- **The sigma priors don't fit the standardised scale.** They were set for log sales; the standardised
  powered sales have unit variance, so `sigma_h` sits on its 0.5 cap for 92–100% of series. The inverse
  takes the capped variance, which understates the spread and so the correction. #116 removes that cap (see
  "Sigma priors and best-parameter tracking (#116)").
- **Fitting takes longer:** 26–35 s against 17–20 s on the sample, for scikit-learn's λ fit (slowest for
  Box-Cox) and the quadrature over the in-sample fit.

**Decision:** `ssp_dow` stays the default `--model` until `ssp_level_dow_yeojohnson` passes a full run and the
evaluation window, which this issue did not run. The sample misled once before (see "Open items").

### Sigma priors and best-parameter tracking (#116)

**The bug.** Each Adam step computes the loss of the current parameters, then updates them.
`fit_batch_series_opt` kept the updated parameters as the best, one step after the loss they were credited with.
A step past a uniform prior's bound has an infinite loss, so a sigma its bound held came back up to one step past
the bound (`sigma_h` up to 0.509). `fit_one_series_opt` returned the parameters after its last step. Both now
return the parameters of their lowest loss, which lie inside the bounds; a sigma a bound holds now stops up to one
step short of it.

The fix alone, with the default priors, on the 3,000-series sample at the validation origin unless the row says
otherwise:

| Entry | WRMSSE before | After | Delta | Bias before | After |
|---|---|---|---|---|---|
| `ssp_dow` | 0.93487 | 0.94808 | +0.01320 | −9.6% | −10.2% |
| `ssp_level_dow` | 0.94145 | 0.94956 | +0.00811 | −7.6% | −8.2% |
| `ssp_level_dow_yeojohnson` | 0.83171 | 0.83233 | +0.00062 | −0.4% | −0.8% |
| `ssp_dow`, sample, evaluation origin | 0.88578 | 0.90002 | +0.01424 | | |
| `ssp_dow`, full run, validation | 1.08136 | 1.10175 | +0.02038 | −12.7% | −13.2% |
| `ssp_dow`, full run, evaluation | 0.97573 | 0.99403 | +0.01830 | −9.3% | −9.9% |

Every level's bias falls by 0.5–0.8 points. `sigma_h` used to sit just past its 0.5 cap and now sits just inside
it, so the lognormal correction `exp(σ_h² / 2)` is smaller: levels 1–9 worsen and the item levels (10–12), which
the log entries forecast high, improve by up to 0.002. The baseline's full-run scores are 0.018–0.020 worse.

**The priors** are now settings: `SspDow` has `sigma_h_prior` and `sigma_q_prior` fields, each a `SigmaPrior`
(uniform, half-normal or log-normal), passed to the fitter. The defaults are the old U(0.1, 0.5) and U(0.01, 0.1).
The fit starts each sigma at its prior's mean, which for the defaults is the old starting point. `m5_bias.py`
reads each entry's priors and counts a sigma as on a bound within one Adam step of it, `lr · (1 − exp(−bound))`,
where it used to count 1% of the prior's width; with the default priors the shares match #115's (95.8% of
`ssp_dow`'s `sigma_h` on 0.5, against 95.6%).

**What the sigmas want on the Yeo-Johnson scale.** `ssp_level_dow_yeojohnson` fitted to convergence with no
effective bound (1,000 steps at lr 0.03, or 300 at lr 0.1) scores 0.74839. Its noise sits at 0.16–1.0 (median
0.88), the level's drift at 0.004–0.33 (median 0.047) and the weekdays' shared drift at 0.0005–0.036 (median
0.006). The default priors hold `sigma_h` at 0.5 and so leave noise over for the states, whose drifts rise to
their own 0.1 cap; freed, the weekday drift falls below the old 0.01 floor for 93% of series.

The registered variants are `ssp_level_dow_yeojohnson` with other priors, at today's 100 steps and lr 0.03. Each
family got one setting, read off the ranges above (fit days only), not tuned on the validation window:

| Entry | `sigma_h` prior | Drift prior | WRMSSE | Delta | Bias | On a bound |
|---|---|---|---|---|---|---|
| `ssp_level_dow_yeojohnson` (#115, before the fix) | U(0.1, 0.5) | U(0.01, 0.1) | 0.83171 | | −0.4% | 99.9% |
| `ssp_level_dow_yeojohnson` | U(0.1, 0.5) | U(0.01, 0.1) | 0.83233 | +0.00062 | −0.8% | 99.6% |
| `ssp_level_dow_yeojohnson_wide` | U(0.05, 1.5) | U(0.0005, 0.5) | 0.81790 | −0.01382 | +5.6% | 0.0% |
| `ssp_level_dow_yeojohnson_halfnormal` | HalfNormal(1) | HalfNormal(0.05) | **0.74610** | **−0.08561** | +1.9% | no bound |
| `ssp_level_dow_yeojohnson_lognormal` | LogNormal(log 0.8, 0.5) | LogNormal(log 0.02, 1) | 0.74740 | −0.08431 | +2.0% | no bound |

Delta is against #115's best, the first row. "On a bound" is the share of series with any sigma on a bound
(`m5_bias.py`, `bounds.csv`); #115's 99.9% was counted the old way.

Per level (`m5_compare.py`):

| Level | #115 best | After the fix | `_wide` | `_halfnormal` | `_lognormal` |
|---|---|---|---|---|---|
| 1 Total | 0.6690 | 0.6696 | 0.6812 | 0.5127 | 0.5137 |
| 2 State | 0.7066 | 0.7075 | 0.7050 | 0.5796 | 0.5813 |
| 3 Store | 0.8413 | 0.8419 | 0.8352 | 0.7744 | 0.7771 |
| 4 Category | 0.6881 | 0.6901 | 0.6914 | 0.5784 | 0.5797 |
| 5 Department | 0.8138 | 0.8157 | 0.8041 | 0.7121 | 0.7155 |
| 6 State × category | 0.7660 | 0.7669 | 0.7491 | 0.6699 | 0.6703 |
| 7 State × department | 0.8681 | 0.8689 | 0.8463 | 0.7885 | 0.7900 |
| 8 Store × category | 0.8760 | 0.8764 | 0.8622 | 0.8182 | 0.8197 |
| 9 Store × department | 0.9302 | 0.9303 | 0.9091 | 0.8700 | 0.8712 |
| 10 Item | 0.9399 | 0.9397 | 0.9116 | 0.8842 | 0.8845 |
| 11 Item × state | 0.9390 | 0.9387 | 0.9084 | 0.8813 | 0.8816 |
| 12 Item × store | 0.9425 | 0.9422 | 0.9110 | 0.8839 | 0.8842 |
| Bias, levels 1–9 | −0.3 to −1.1% | −0.8 to −1.6% | +4.9 to +5.6% | +1.0 to +2.0% | +1.1 to +2.1% |
| Bias, levels 10–12 | +10.5 to +12.2% | +9.9 to +11.6% | +18.2 to +19.7% | +14.2 to +15.6% | +14.1 to +15.5% |

Bias by group over the last 28 fit days (`m5_bias.py`; position and slope as in #113):

| Group | #115 best, after the fix | `_halfnormal` |
|---|---|---|
| Total | −0.8% | +1.9% |
| HOBBIES_1 | −2.6% | −1.9% |
| Low-volume tercile | −29.6% | −32.4% |
| Mid-volume tercile | −14.3% | −13.0% |
| High-volume tercile | +5.2% | +8.9% |
| Position, slope (total) | 0.92, 0.85 | 0.98, 0.68 |

- **The soft priors are better at every level**: `_halfnormal` by 0.056–0.156, most on the totals. They reach the
  unbounded optimum (0.74839) within today's 100 steps because they start near it, at their means: 0.80 for the
  noise and 0.040 for the drifts under the half-normals. The fit time is unchanged (29 s).
- **The wide uniform priors are right but start too far away.** They hold no series on a bound, so the cap is
  gone, but the fit starts at their midpoints, 0.775 and 0.25, and in 100 steps the weekday drift gets only to
  0.04: 0.81790, and 5.6% high. At lr 0.1 the same priors score 0.74973 (not registered).
- **The half-normal drift prior is tight for the level's drift**: it puts 9.3% of them past its 99th percentile,
  0.129 (2.4% under the log-normal). Separate priors for the level's and the weekdays' drifts, which differ by an
  order of magnitude, are an open item.
- **The bias moves up and splits further by volume.** The totals go from 0.8% low to 1.9% high, and the
  dollar-weighted item bias from +10–12% to +14–16%. The low-volume tercile stays about 30% low and the
  high-volume one goes 9% high.
- **On the log target, removing the cap backfires.** `ssp_level_dow_halfnormal` (`ssp_level_dow` with the
  half-normal priors) scores 1.39880, +0.449 against `ssp_level_dow`, and forecasts 23% high at levels 1–9. Freed,
  log-scale `sigma_h` goes to 0.47–1.33 (5th to 95th percentile), so the lognormal correction `exp(σ_h² / 2)`
  reaches ×2.4. The log of
  intermittent sales is far from Gaussian, so the variance that fits it overstates the correction; the cap had
  hidden that. The log entries keep the default priors.

**Decision:** `ssp_level_dow_yeojohnson_halfnormal` is the best entry on the sample. `ssp_dow` stays the default
`--model` until #118 full-runs the winner and checks it on the evaluation window.

### Integer rounding (#117)

M5 sales are integers, but WRMSSE is a squared error, so the best point forecast of a day's sales is their mean,
which is rarely an integer. `m5_run.py --round` clips the forecasts at 0 and rounds them to the nearest integer
(halves to even) before it plots, saves, submits and scores them. It is off by default. The fits are deterministic,
so a rounded run's forecasts are exactly the unrounded run's, rounded.

Results on the 3,000-series sample (seed 2026) at the validation origin; a positive delta means rounding is worse.
Zeros is the share of forecasts that round to 0, and all-zero the share of series whose 28 forecasts all do.

| Entry | WRMSSE | Rounded | Delta | Bias | Rounded | Zeros | All-zero |
|---|---|---|---|---|---|---|---|
| `ssp_dow` | 0.94808 | 1.17867 | +0.23059 | −10.2% | −17.0% | 58.5% | 27.1% |
| `ssp_level_dow_yeojohnson` | 0.83233 | 0.91244 | +0.08011 | −0.8% | −5.4% | 56.2% | 31.4% |
| `ssp_level_dow_yeojohnson_halfnormal` | 0.74610 | 0.81296 | +0.06686 | +1.9% | −3.3% | 50.9% | 43.2% |

On a full run of `ssp_level_dow_yeojohnson_halfnormal`, both origins, rounding scores the validation window 0.85819
against 0.70891 (+0.14928) and the evaluation window 0.83142 against 0.79394 (+0.03747). Rounded, 51% (validation)
and 49% (evaluation) of the forecasts are 0, and 43% and 41% of the series are forecast 0 on all 28 days.

Per level of `ssp_level_dow_yeojohnson_halfnormal` (`m5_compare.py`):

| Level | Sample | Rounded | Full, validation | Rounded | Full, evaluation | Rounded |
|---|---|---|---|---|---|---|
| 1 Total | 0.5127 | 0.5464 | 0.5318 | 0.7080 | 0.5972 | 0.4713 |
| 2 State | 0.5796 | 0.6153 | 0.5725 | 0.7163 | 0.6508 | 0.5527 |
| 3 Store | 0.7744 | 0.8119 | 0.6464 | 0.7856 | 0.6918 | 0.6566 |
| 4 Category | 0.5784 | 0.7154 | 0.5690 | 0.8123 | 0.6301 | 0.7206 |
| 5 Department | 0.7121 | 0.8755 | 0.6980 | 0.9476 | 0.7407 | 0.8969 |
| 6 State × category | 0.6699 | 0.7546 | 0.6181 | 0.8089 | 0.7064 | 0.7574 |
| 7 State × department | 0.7885 | 0.8990 | 0.7288 | 0.9364 | 0.7924 | 0.9211 |
| 8 Store × category | 0.8182 | 0.8779 | 0.6856 | 0.8403 | 0.7462 | 0.8016 |
| 9 Store × department | 0.8700 | 0.9386 | 0.7809 | 0.9440 | 0.8358 | 0.9410 |
| 10 Item | 0.8842 | 0.9090 | 0.9019 | 0.9646 | 1.1193 | 1.1805 |
| 11 Item × state | 0.8813 | 0.9046 | 0.8896 | 0.9262 | 1.0475 | 1.0839 |
| 12 Item × store | 0.8839 | 0.9072 | 0.8843 | 0.9081 | 0.9691 | 0.9933 |
| WRMSSE | 0.74610 | 0.81296 | 0.70891 | 0.85819 | 0.79394 | 0.83142 |
| Bias, levels 1–9 | +1.0 to +2.0% | −3.2 to −5.3% | −1.3 to −1.5% | −6.4 to −7.6% | +4.2 to +5.2% | −1.8 to +0.2% |
| Bias, levels 10–12 | +14.2 to +15.6% | +2.9 to +4.1% | +6.0 to +12.9% | −3.8 to +2.5% | +23.9 to +47.7% | +15.1 to +36.9% |

- **Rounding is worse at every level** on the sample, for every entry, and on the full run's validation window. It
  hurts the small series: in the sample's low-volume tercile 95% of the forecasts round to 0 and the tercile's
  forecast total falls by 74%. The high-volume tercile barely moves (its squared error ×1.005). Across the sample,
  the days whose forecast rounds to 0 sell 0.36 units on average, and rounding raises the squared error of 76% of
  the series.
- **The aggregates suffer most.** The low-volume series all round down, so their rounding errors share a sign and
  add up in every sum instead of cancelling: rounding takes 5% off the forecast total, and on the full run's
  validation window the category and department levels (4–7) worsen by 0.19–0.25.
- **At the item levels the bias falls but the error rises.** On the full run's validation window rounding pulls the
  dollar-weighted item bias from +6 to +13% to −4 to +3%, yet levels 10–12 worsen by 0.024–0.063. For a forecast at
  a series' mean, rounding adds `(f − round(f))²`, up to 0.25 squared units, to each day's expected squared error.
- **The evaluation window's gain at levels 1–3 is a bias correction, not an effect of integers.** There the
  unrounded totals run 5.1% high, and rounding's 5% cut brings them to +0.1%, so the total, state and store levels
  improve (by 0.126, 0.098 and 0.035). Levels 4–12 still worsen. A level correction would get the totals' gain
  without the cost below them; on the validation window, where the totals are 1.3% low, rounding brings only the
  cost.

The unrounded full run is the first of the #116 winner: 0.70891 on the validation window and 0.79394 on the
evaluation window, against 1.10175 and 0.99403 for `ssp_dow` after the #116 fix. It fits in 524 s (257 s and 267 s
per origin) against `ssp_dow`'s 319 s, at a 22.1 GiB peak. Neither run's `submission.csv` was submitted to Kaggle; #118 decides
the default model.

**Decision:** rounding stays off by default; `--round` is opt-in. It is worse at every level on the validation
window, on the sample and on the full run, and worse overall on the evaluation window.

### Pre-validation window (#125)

The validation and evaluation windows are Kaggle's public and private leaderboards, and the local scorer matches
Kaggle to within 0.0001, so a local score on either is a leaderboard score. Every choice from #106 to #117 was made
on the validation window, and every full run printed the evaluation score. #125 adds a window Kaggle never scores:
fit on d_1–d_1885 and score d_1886–d_1913, the validation window moved 28 days earlier. It follows Kaggle's recipe:
the RMSSE scale runs over all the fit days, and the weights are the dollar sales of the last 28 (d_1858–d_1885).
Its actuals are in the `train` split. `--origins all`, the default, fits all three windows; a submission still
takes only the validation and evaluation forecasts.

All three windows, on the 3,000-series sample (seed 2026) and on full runs. Of these forecasts only the full run of
`ssp_level_dow_yeojohnson_halfnormal` has been submitted, for #118 (the #94 and #106 submissions predate the #116
fix), and a sample cannot be.

| Entry | Series | Pre-validation | Validation | Evaluation | Kaggle public | Kaggle private |
|---|---|---|---|---|---|---|
| `ssp_dow` | 3,000 | 1.00659 | 0.94808 | 0.90002 | | |
| `ssp_level_dow` | 3,000 | 0.94898 | 0.94956 | 0.90309 | | |
| `ssp_level_dow_yeojohnson_halfnormal` | 3,000 | 0.82376 | 0.74610 | 0.74957 | | |
| `ssp_dow` | 30,490 | 1.05983 | 1.10175 | 0.99403 | | |
| `ssp_level_dow` | 30,490 | 0.97009 | 1.06591 | 0.98720 | | |
| `ssp_level_dow_yeojohnson_halfnormal` | 30,490 | 0.79563 | 0.70891 | 0.79394 | 0.70888 | 0.79394 |

Per level of the full runs on the pre-validation window, with the bias of the total:

| Level | `ssp_dow` | `ssp_level_dow` | `ssp_level_dow_yeojohnson_halfnormal` |
|---|---|---|---|
| 1 Total | 1.0617 | 0.8183 | 0.5793 |
| 2 State | 1.0221 | 0.8280 | 0.6463 |
| 3 Store | 1.0376 | 0.8721 | 0.7024 |
| 4 Category | 1.0957 | 0.9190 | 0.6704 |
| 5 Department | 1.2003 | 1.1442 | 0.8856 |
| 6 State × category | 1.0556 | 0.9353 | 0.7209 |
| 7 State × department | 1.1284 | 1.0986 | 0.8744 |
| 8 Store × category | 1.0406 | 0.9482 | 0.7776 |
| 9 Store × department | 1.0903 | 1.0663 | 0.8903 |
| 10 Item | 1.0480 | 1.0345 | 0.9558 |
| 11 Item × state | 0.9895 | 0.9990 | 0.9350 |
| 12 Item × store | 0.9480 | 0.9776 | 0.9096 |
| WRMSSE | 1.05983 | 0.97009 | 0.79563 |
| Bias of the total | −11.4% | −6.8% | +4.4% |

- **On the full runs the three windows rank the entries the same:** the #116 winner first, then `ssp_level_dow`,
  then `ssp_dow`.
- **On the sample, the pre-validation window agrees with the full runs where the validation window does not.**
  The sample's validation and evaluation windows score `ssp_level_dow` worse than `ssp_dow` (+0.0015 and
  +0.0031), the full runs better on every window. The sample's pre-validation window scores it better
  (−0.0576). That is one pair of entries, so it does not yet show the pre-validation sample predicts full runs.
- **The #116 winner runs high on two windows of three:** its total is 4.4% high on pre-validation, 1.3% low on
  validation and 5.1% high on evaluation. `ssp_dow` is 11.4%, 12.7% and 9.3% low.
- **Easter Sunday is d_1885**, the window's last fit day. The ranking matches the Kaggle windows, so it does
  not reorder these three entries; whether it moves the level at the cutoff was not checked.
- **Nothing else moved:** the validation and evaluation scores of the sample `ssp_dow` run and of the full
  `ssp_dow` and `ssp_level_dow_yeojohnson_halfnormal` runs equal those of the earlier runs at every level
  (`20261017_104210`, `_104252` and `_114423`), and the two full runs' `submission.csv` files are identical to
  theirs. There is no earlier full run of `ssp_level_dow` after the #116 fix to compare with.
- **Cost:** a third origin adds one fit, so a full run of `ssp_dow` takes 496 s instead of 328 s, and of the
  #116 winner 820 s; peak memory is 22.7, 25.5 and 22.9 GiB for the three entries. A pre-validation quick loop
  on the sample takes 20 s, as the validation one did.

**Decision:** choose models on the pre-validation window from now on, and read the validation and evaluation
windows once per candidate, on its full run (see "The experiment loop"). For #118, all three windows put
`ssp_level_dow_yeojohnson_halfnormal` first.

### Full run and the default model (#118)

The #116 winner, `ssp_level_dow_yeojohnson_halfnormal`, against `ssp_dow` on full runs (30,490 series), read on the
validation and evaluation windows once, here. The runs are the #125 ones, `20261017_195848` (`ssp_dow`) and
`20261017_201617` (the winner), both at all three origins; they were not re-fitted. Both were made at commit
`da6ebca` with a dirty working tree (`git_dirty` in `runs.csv`): the uncommitted #125 change, which the #125 notes
show left both runs' validation and evaluation forecasts unchanged. The winner's `submission.csv` was submitted to
Kaggle by hand; its scores are in `runs.csv`.

Per level (`m5_compare.py`), against `ssp_dow` before the #116 fix (the #94 submission, `20261016_215316`) and
after it. A dash means the forecasts were not submitted.

| Level | Validation, #94 | `ssp_dow` | Default | Evaluation, #94 | `ssp_dow` | Default |
|---|---|---|---|---|---|---|
| 1 Total | 1.1067 | 1.1426 | 0.5318 | 0.8628 | 0.8970 | 0.5972 |
| 2 State | 1.0784 | 1.1093 | 0.5725 | 0.8833 | 0.9113 | 0.6508 |
| 3 Store | 1.0961 | 1.1229 | 0.6464 | 0.9290 | 0.9527 | 0.6918 |
| 4 Category | 1.1297 | 1.1627 | 0.5690 | 0.9108 | 0.9416 | 0.6301 |
| 5 Department | 1.1972 | 1.2270 | 0.6980 | 0.9890 | 1.0174 | 0.7407 |
| 6 State × category | 1.0947 | 1.1224 | 0.6181 | 0.9371 | 0.9611 | 0.7064 |
| 7 State × department | 1.1394 | 1.1636 | 0.7288 | 1.0008 | 1.0219 | 0.7924 |
| 8 Store × category | 1.0762 | 1.0981 | 0.6856 | 0.9491 | 0.9680 | 0.7462 |
| 9 Store × department | 1.1041 | 1.1209 | 0.7809 | 0.9938 | 1.0088 | 0.8358 |
| 10 Item | 1.0325 | 1.0329 | 0.9019 | 1.1666 | 1.1658 | 1.1193 |
| 11 Item × state | 0.9772 | 0.9762 | 0.8896 | 1.0817 | 1.0800 | 1.0475 |
| 12 Item × store | 0.9440 | 0.9424 | 0.8843 | 1.0047 | 1.0028 | 0.9691 |
| WRMSSE | 1.08136 | 1.10175 | **0.70891** | 0.97573 | 0.99403 | **0.79394** |
| Kaggle | 1.08127 | – | **0.70888** | 0.97572 | – | **0.79394** |
| Bias, levels 1–9 | −12.6 to −13.2% | −13.1 to −13.8% | −1.3 to −1.5% | −9.3 to −10.2% | −9.9 to −10.8% | +4.2 to +5.2% |
| Bias, levels 10–12 | −2.6 to +8.1% | −3.2 to +7.4% | +6.0 to +12.9% | +7.1 to +28.2% | +6.4 to +27.4% | +23.9 to +47.7% |

- **The winner is better at every level on both windows**, against both baselines: −0.37245 on validation and
  −0.18179 on evaluation against the #94 submission (−0.39283 and −0.20008 against `ssp_dow` after the #116 fix).
  The gain is largest on the totals (level 1: −0.575 and −0.266) and smallest on the item levels (level 12: −0.060
  and −0.036).
- **The epic's bias target is met:** the validation total is 1.3% low, against the baseline's 12.7%. On the
  evaluation window it is 5.1% high, against 9.3% low, and on pre-validation 4.4% high (#125).
- **The item levels run high, most on the evaluation window.** The dollar-weighted item bias rises from +6 to +13%
  on validation to +24 to +48% on evaluation, and level 10 (item) still scores 1.1193 there, worse than the naive
  forecast; `ssp_dow`'s item levels run high too (+6 to +27%). The open item is in "Known problems".
- **Kaggle matches the local scorer:** public 0.70888 against a local 0.70891, private 0.79394 against 0.79394.
- **Cost:** the full run takes 820 s against 496 s for `ssp_dow` (259–270 s per origin against 152–161 s), at a
  22.9 GiB peak against 22.7 GiB.

**Decision:** `ssp_level_dow_yeojohnson_halfnormal` is the default `--model` (`DEFAULT_MODEL` in `m5_models.py`).
It wins on the sample, on all three full-run windows and on both Kaggle leaderboards. `ssp_dow` stays registered as
the baseline that runs are compared with.

## Working on the model

### Where the code is

| File | What to change there |
|---|---|
| `src/bunobee/models/m5/m5_ssp_optim.py` | `fit_batch_series_opt`: the parameters, loss and optimizer. `SigmaPrior`: the priors of the noise and the drifts, which the fitter takes as arguments (by default `SIGMA_H_PRIOR` and `SIGMA_Q_PRIOR`). `predict_batch_series_opt`: the forecast, back-transformed by the fit's transform |
| `src/bunobee/models/m5/transforms.py` | The target transforms: the `Transform` protocol (`fit` per-series parameters on the fit days, `forward`, and `inverse` with its bias correction), `LogTransform`, the default, and `PowerTransform`, a Box-Cox or Yeo-Johnson power fitted per series. A new transform goes here; an entry picks it with its `transform` field, without an edit of the fitter |
| `playground/m5_prototype/m5_models.py` | The registry `MODELS`. An entry's `fit_forecast` builds the model's inputs for one cutoff day (`SspDow` calls `make_periodic_dummies`) and calls the model. Settings such as `transform`, `sigma_h_prior`, `sigma_q_prior`, `n_iter`, `lr` and `fit_chunk` are the entry's fields |
| `src/bunobee/regression.py` | `make_periodic_dummies`, which builds the weekday dummies and the column of ones, the models' only inputs today |
| `src/bunobee/models/ssp/kalman_1d.py` | `kalman_filter_1d`, the filter inside the loss |

An entry's `fit_forecast(ds, series_idx, fit_mask, forecast_mask)` gets the whole dataset, the series to
fit and the fit and forecast days. It returns forecasts (series × 28) with no NaNs and a dict of per-series
arrays, which `m5_run.py` saves to `batch_params.npz` and plots in `diagnostics.png`. The dict can also hold
the model's in-sample fit (series × fit days) under `insample`, which `forecasts.pdf` draws; `ssp_dow`
returns each fit day's one-step-ahead prediction, back-transformed as its forecast is. Everything else
(data loading, the three cutoff days, the submission, the scoring) lives in `m5_run.py` and works with any model,
so adding a model needs only a new entry.

### The baseline model (`ssp_dow`)

`ssp_dow` was the default until #118. The default, `ssp_level_dow_yeojohnson_halfnormal`, is the same model with the
level state, a Yeo-Johnson target and half-normal sigma priors (see "Power transforms (#115)" and "Sigma priors and
best-parameter tracking (#116)").

- **Transform:** it models `log(max(sales, 0.1) / mean)` with Gaussian noise, where `mean` is the series'
  mean of `max(sales, 0.1)` over the fit days. Every zero-sales day becomes `log(0.1 / mean)`. This is
  `LogTransform()`, the default `transform` of `SspDow`.
- **Structure:** `y_t = Σ_i Z[t, i] · a[t, i] + noise`. The columns of `Z` are weekday indicators for
  Tuesday to Sunday (Monday is dropped), and each state `a_i` drifts as a random walk.
- **Parameters:** 3 per series: the noise level, the drift of the first state and a drift shared by the
  other five. Uniform priors, U(0.1, 0.5) on the noise and U(0.01, 0.1) on each drift, which hold most series
  on their upper bounds; 100 Adam steps at learning rate 0.03, keeping the parameters of the lowest loss.
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
  above), and so does the default model.
- **Totals are forecast too low.** In the baseline run, total sales are 12.7% low on the validation window
  and 9.3% low on the evaluation window, and every level from 1 to 9 is low by a similar amount. The item
  levels (10–12) are not. The missing level state explains only part of it: `ssp_level_dow` lifts the
  totals by 2.4 points on validation and 5.6 on evaluation. The main cause is the log transform: the
  forecast tracks the geometric mean of `max(sales, 0.1)`, and the lognormal correction, with `sigma_h`
  capped, lifts it only part of the way to the arithmetic mean (see "Level bias diagnosis (#113)").
  `ssp_level_dow_yeojohnson` follows the arithmetic mean and forecasts the sample's totals 0.4% low, but
  not yet on a full run (see "Power transforms (#115)"). With soft sigma priors in place of the caps,
  `ssp_level_dow_yeojohnson_halfnormal` forecasts them 1.9% high and scores 0.74610 on the sample (see "Sigma
  priors and best-parameter tracking (#116)"). On the full run, now the default, it forecasts them 1.3% low on
  validation and 5.1% high on evaluation (see "Full run and the default model (#118)").
- **The default model forecasts the items too high.** On the full run its dollar-weighted item bias (levels 10–12)
  is +6 to +13% on validation and +24 to +48% on evaluation, and the item level scores 1.1193 on evaluation. On
  the sample the bias splits by volume: the low-volume tercile runs about 30% low and the high-volume one 9% high
  (see "Sigma priors and best-parameter tracking (#116)").
- **Unused data:** `sell_price`, `snap`, events, day of month and month are in `m5_ds.nc` but not in `Z`.
  New inputs are new columns of `Z`, so each becomes a state; `fit_batch_series_opt` gives every column
  after the first the shared drift.
- **No sharing between series:** each series is fitted on its own, which leaves sparse series with noisy
  parameters.

### Open items

- **The sample did not predict the full run:** for `ssp_level_dow`, the 3,000-series sample scored the
  validation window +0.0066 worse while the full run scored it −0.0308 better (+0.0015 and −0.0358 after the
  #116 fix). On the pre-validation window the sample gets the sign right (−0.0576, full run −0.0897; see
  "Pre-validation window (#125)"), but that is one pair of entries. Enlarge the sample or stratify it by
  store × department before steering the next model by it.
- **Hierarchical parameter sharing:** share the weekday drift within department × store groups (70 groups),
  either outright or through a prior that pulls each series toward its group. Most useful for sparse
  series.
- **One prior for both drifts:** `sigma_q_prior` covers the level's drift and the weekdays' shared drift, which
  on the Yeo-Johnson scale sit an order of magnitude apart (medians 0.047 and 0.006). A prior for each would let
  the level's drift prior be wider than `HalfNormal(0.05)`.
- **No test that the batch fit is correct:** nothing checks that `fit_batch_series_opt` matches the
  single-series `fit_one_series_opt` on the same series.
- **float32 log-likelihood:** the loss sums 1,913 steps in float32. Summing in float64 has not been tried.
- **Memory:** later fits do not release the earlier ones' memory, so a full run of `ssp_dow` at three origins
  peaks at 22.7 GiB, against 13.8 GiB after the first (25.5 GiB for `ssp_level_dow`). The `fit_chunk` setting
  of `SspDow` caps it meanwhile.
- **Optimizer loop:** a Python loop over a compiled step, not `lax.scan`. A cutoff day already fits in
  about 159 s, so this only matters for speed.
- **Multiple devices:** no `pmap`; only useful with several GPUs.
