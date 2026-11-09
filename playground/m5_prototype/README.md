# M5 aggregate

`bunobee` was developed against the [M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
data. What stays here is the packaged aggregate panel (`bunobee.datasets.load_m5_aggregate()`), the way to rebuild it,
and two notebooks that use it. The research on the full 30,490 series (model comparisons, scoring, submissions) is no
longer part of this repository.

| File | What it is |
|---|---|
| [`ssp_m5_aggregate_forecast_seasonality.ipynb`](ssp_m5_aggregate_forecast_seasonality.ipynb) | Tutorial. Fits the national series with `fit_map` and a Yeo-Johnson target, then adds annual Fourier and event columns as fixed-coefficient covariates under a loose and a tight prior. Needs only the packaged panel. Committed without outputs. |
| [`m5_aggregate_forecast_accuracy.ipynb`](m5_aggregate_forecast_accuracy.ipynb) | The readable spec behind `tests/test_m5_accuracy.py`: the SSP and DLT accuracy guards on the packaged panel. Committed with its outputs. |
| [`m5_make_accuracy_fixture.py`](m5_make_accuracy_fixture.py) | Rebuilds `src/bunobee/datasets/data/m5_aggregate.csv` from the raw M5 files. Its docstring defines every column. |
| `m5_fetch.py`, `m5_preprocess.py`, `m5_common.py` | Download and preprocess the raw files. Only needed to rebuild the panel. |

For the tutorial, install the `optim` extra: `pip install -e ".[optim]"` (adds `optax`, `tqdm`, `scikit-learn`).

## Rebuilding the packaged panel

Only needed if you change `m5_make_accuracy_fixture.py`. No unit test reads the raw files; `--check` is the way to
compare the packaged panel with them. All commands run from the repo root, and every downloaded
or generated file is gitignored.

1. The Kaggle CLI: `pip install kaggle`, on `PATH`.
2. Kaggle credentials: `kaggle auth login`, or a token from <https://www.kaggle.com/settings/api> saved to
   `~/.kaggle/access_token` (or exported as `KAGGLE_API_TOKEN`).
3. The competition rules, accepted once while signed in:
   <https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules>, and, for the Uncertainty
   `sample_submission.csv` that `m5_fetch.py` also downloads,
   <https://www.kaggle.com/competitions/m5-forecasting-uncertainty/rules>.

```sh
python playground/m5_prototype/m5_fetch.py          # download the raw files into playground/resource/
python playground/m5_prototype/m5_preprocess.py     # build the cached dataset m5_ds.nc (~550 MB)
python playground/m5_prototype/m5_make_accuracy_fixture.py          # rewrite the packaged panel
python playground/m5_prototype/m5_make_accuracy_fixture.py --check  # or only compare it with the raw files
pytest tests/test_m5_accuracy.py                                    # the pinned digest must match the rebuilt file
```
