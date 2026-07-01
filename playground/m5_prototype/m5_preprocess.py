"""M5 Forecasting — data preprocessing.

Builds a single xarray Dataset from the four M5 CSV files, covering all
30 490 item-store series over the full 1 969-day calendar window.

Dataset structure
-----------------
Dimensions
    series_id : 30 490  (item × store, e.g. "FOODS_1_001_CA_1")
    date      : 1 969   (2011-01-29 … 2016-06-19)

Data variables
    sales         (series_id, date) float32 — unit sales; NaN for forecast window
    sell_price    (series_id, date) float32 — weekly price expanded to daily; NaN
                                              where not listed in sell_prices.csv
    snap          (series_id, date) int8    — SNAP eligibility for the series' state
    day_of_week   (date,)          int8    — day of week (Mon=0, Sun=6)
    day_of_month  (date,)          int8    — day of month (1–31)
    month_of_year (date,)          int8    — month of year (1–12)

Coordinates on series_id
    item_id, dept_id, cat_id, store_id, state_id

Coordinates on date
    event_name_1, event_type_1, event_name_2, event_type_2,
    snap_CA, snap_TX, snap_WI, split

    split : str — "train" (d_1–d_1913), "validation" (d_1914–d_1941),
                  "test"  (d_1942–d_1969)

Attributes
    train_end_date    : last training date
    valid_end_date    : last validation date
    calendar_end_date : last calendar date
    train_steps       : number of training days
    valid_steps       : number of validation days
    test_steps        : number of test days
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "resource" / "m5-forecasting-accuracy"

# Competition split boundaries (day indices, 1-based)
_TRAIN_END_D = 1913  # sales_train_validation.csv ends here
_VALID_END_D = 1941  # sales_train_evaluation.csv ends here
# Calendar ends at d_1969 → test window is d_1942–d_1969


def build_m5_dataset(
    data_dir: Path | str = _DEFAULT_DATA_DIR,
    output_path: Path | str | None = None,
) -> xr.Dataset:
    """Build a single xarray Dataset from all M5 source CSV files.

    Parameters
    ----------
    data_dir : Path | str
        Directory containing the raw M5 CSV files.
    output_path : Path | str | None
        If provided, the dataset is saved as NetCDF to this path.

    Returns
    -------
    xr.Dataset
        Unified M5 dataset with dimensions (series_id, date).
    """
    data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # 1. Load raw CSVs
    # ------------------------------------------------------------------
    print("Loading CSVs …")
    calendar = pd.read_csv(data_dir / "calendar.csv", parse_dates=["date"])
    sell_prices = pd.read_csv(data_dir / "sell_prices.csv")

    # Prefer evaluation CSV (d_1 … d_1941); fall back to validation (d_1 … d_1913)
    eval_path = data_dir / "sales_train_evaluation.csv"
    val_path = data_dir / "sales_train_validation.csv"
    sales_train = pd.read_csv(eval_path if eval_path.exists() else val_path)

    # ------------------------------------------------------------------
    # 2. Series metadata
    # ------------------------------------------------------------------
    meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    meta = sales_train[meta_cols].copy()

    # Strip the competition suffix ("_validation" / "_evaluation") from id
    meta["series_id"] = (
        meta["id"].str.replace("_validation$", "", regex=True).str.replace("_evaluation$", "", regex=True)
    )
    series_ids = meta["series_id"].values  # (30490,)

    # ------------------------------------------------------------------
    # 3. Date index and split labels
    # ------------------------------------------------------------------
    # Calendar has one row per day; column 'd' = "d_1" … "d_1969"
    day_numbers = calendar["d"].str.replace("d_", "", regex=False).astype(int).values
    all_dates = calendar["date"].values  # np.datetime64[ns], shape (1969,)

    split_labels = np.where(
        day_numbers <= _TRAIN_END_D,
        "train",
        np.where(day_numbers <= _VALID_END_D, "validation", "test"),
    )

    # ------------------------------------------------------------------
    # 4. Sales matrix (series_id × date)  shape (30490, 1969)
    # ------------------------------------------------------------------
    print("Building sales matrix …")
    day_cols = sorted(
        [c for c in sales_train.columns if c.startswith("d_")],
        key=lambda x: int(x.split("_")[1]),
    )
    sales_mat = sales_train[day_cols].values.astype(np.float32)  # (30490, ≤1941)

    # Pad with NaN for any remaining calendar days beyond the sales CSV
    n_extra = len(all_dates) - sales_mat.shape[1]
    if n_extra > 0:
        sales_mat = np.concatenate(
            [sales_mat, np.full((len(series_ids), n_extra), np.nan, dtype=np.float32)],
            axis=1,
        )

    # ------------------------------------------------------------------
    # 5. Sell-price matrix (series_id × date)  shape (30490, 1969)
    #
    # sell_prices.csv is weekly (wm_yr_wk granularity).  Efficient
    # expansion strategy:
    #   a) pivot to (series_id × wm_yr_wk)   ~ (30490 × 282)
    #   b) reindex columns by the per-date wm_yr_wk sequence from
    #      calendar — this duplicates each week's column 7 times,
    #      giving a (30490 × 1969) daily matrix in one step.
    # ------------------------------------------------------------------
    print("Building sell-price matrix …")
    price_df = sell_prices.copy()
    price_df["series_id"] = price_df["item_id"] + "_" + price_df["store_id"]

    price_weekly = price_df.pivot_table(
        index="series_id",
        columns="wm_yr_wk",
        values="sell_price",
        aggfunc="first",
    )  # (n_series × n_weeks)

    # Expand weekly → daily by reindexing with the per-date wm_yr_wk sequence
    wm_yr_wk_seq = calendar["wm_yr_wk"].values  # (1969,) with repeats
    price_daily = price_weekly.reindex(index=series_ids, columns=wm_yr_wk_seq)
    price_daily.columns = all_dates
    price_mat = price_daily.values.astype(np.float32)  # (30490, 1969)

    # ------------------------------------------------------------------
    # 6. SNAP matrix (series_id × date)  shape (30490, 1969)
    #    Each series belongs to one state; select that state's SNAP flag.
    # ------------------------------------------------------------------
    print("Building SNAP matrix …")
    snap_by_state = {
        "CA": calendar["snap_CA"].values.astype(np.int8),
        "TX": calendar["snap_TX"].values.astype(np.int8),
        "WI": calendar["snap_WI"].values.astype(np.int8),
    }
    states = meta["state_id"].values  # (30490,)
    snap_mat = np.stack([snap_by_state[s] for s in states], axis=0)  # (30490, 1969)

    # ------------------------------------------------------------------
    # 7. Assemble xarray Dataset
    # ------------------------------------------------------------------
    print("Assembling xarray Dataset …")
    ds = xr.Dataset(
        data_vars={
            "sales": (["series_id", "date"], sales_mat),
            "sell_price": (["series_id", "date"], price_mat),
            "snap": (["series_id", "date"], snap_mat),
            # date-stamp features (re-derivable from date, stored for convenience)
            "day_of_week": ("date", pd.DatetimeIndex(all_dates).day_of_week.astype(np.int8)),
            "day_of_month": ("date", pd.DatetimeIndex(all_dates).day.astype(np.int8)),
            "month_of_year": ("date", pd.DatetimeIndex(all_dates).month.astype(np.int8)),
        },
        coords={
            # dimension coordinates
            "series_id": series_ids,
            "date": all_dates,
            # series metadata (indexed by series_id)
            "item_id": ("series_id", meta["item_id"].values),
            "dept_id": ("series_id", meta["dept_id"].values),
            "cat_id": ("series_id", meta["cat_id"].values),
            "store_id": ("series_id", meta["store_id"].values),
            "state_id": ("series_id", meta["state_id"].values),
            # calendar metadata (indexed by date)
            "event_name_1": ("date", calendar["event_name_1"].values),
            "event_type_1": ("date", calendar["event_type_1"].values),
            "event_name_2": ("date", calendar["event_name_2"].values),
            "event_type_2": ("date", calendar["event_type_2"].values),
            "snap_CA": ("date", calendar["snap_CA"].values.astype(np.int8)),
            "snap_TX": ("date", calendar["snap_TX"].values.astype(np.int8)),
            "snap_WI": ("date", calendar["snap_WI"].values.astype(np.int8)),
            "split": ("date", split_labels),
        },
        attrs={
            "train_end_date": str(calendar.loc[calendar["d"] == f"d_{_TRAIN_END_D}", "date"].iloc[0].date()),
            "valid_end_date": str(calendar.loc[calendar["d"] == f"d_{_VALID_END_D}", "date"].iloc[0].date()),
            "calendar_end_date": str(calendar["date"].iloc[-1].date()),
            "train_steps": int(_TRAIN_END_D),
            "valid_steps": int(_VALID_END_D - _TRAIN_END_D),
            "test_steps": int(len(all_dates) - _VALID_END_D),
        },
    )

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    if output_path is not None:
        output_path = Path(output_path)
        print(f"Saving to {output_path} …")
        ds.to_netcdf(output_path)
        print(f"Saved  ({output_path.stat().st_size / 1e6:.0f} MB)")

    return ds


if __name__ == "__main__":
    ds = build_m5_dataset(output_path=_DEFAULT_DATA_DIR / "m5_ds.nc")
    print(ds)
