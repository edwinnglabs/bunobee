"""Shared paths and helpers for the M5 playground scripts.

Every M5 script reads from one data directory: ``m5_fetch.py`` downloads the raw competition CSVs into
it, and ``m5_preprocess.py`` caches the unified xarray Dataset there as ``m5_ds.nc``. The scripts that
consume that cache open it with :func:`load_m5_ds`, which says which step to run when it is missing.

The scripts import this module by name, which works because running a script puts its own directory
on ``sys.path``. A caller that loads a script by path instead (e.g. from ``tests/``) has to register
this module in ``sys.modules`` first — see ``m5_make_accuracy_fixture._load_sibling``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DATA_DIR = Path(__file__).resolve().parent.parent / "resource" / "m5-forecasting-accuracy"
DS_FILE = "m5_ds.nc"


def dow_dummies(dow: np.ndarray) -> np.ndarray:
    """Day-of-week dummies with Monday dropped as the reference category.

    Parameters
    ----------
    dow : np.ndarray
        Day of week per step (Mon=0, Sun=6), e.g. ``ds["day_of_week"].values``.

    Returns
    -------
    np.ndarray
        Float32 dummies of shape ``(len(dow), 6)``, one column per weekday from Tuesday to Sunday.
    """
    dummies = pd.get_dummies(dow, dtype=np.float32).reindex(columns=range(7), fill_value=0.0)
    return dummies.iloc[:, 1:].values


def load_m5_ds(data_dir: Path | str = DATA_DIR) -> xr.Dataset:
    """Open the preprocessed M5 dataset cached by ``m5_preprocess.py``.

    Parameters
    ----------
    data_dir : Path | str
        Directory holding ``m5_ds.nc``.

    Returns
    -------
    xr.Dataset
        The dataset built by ``m5_preprocess.build_m5_dataset``.

    Raises
    ------
    SystemExit
        If ``m5_ds.nc`` is not in ``data_dir``, with a message naming the steps that create it.
    """
    path = Path(data_dir) / DS_FILE
    if not path.is_file():
        raise SystemExit(f"{path} not found; run m5_fetch.py / m5_preprocess.py first.")
    return xr.open_dataset(path)
