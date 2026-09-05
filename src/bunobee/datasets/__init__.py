"""Small, packaged datasets for demos, prototyping, and the test suite.

Data files live alongside this module in ``datasets/data/`` and ship with the installed package
(see ``[tool.setuptools.package-data]`` in ``pyproject.toml``), so a loader here works the same way
in a notebook, a script, or a test — no relative path back into the repo required.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent / "data"

M5_AGGREGATE_PATH = _DATA_DIR / "m5_aggregate.csv"


def load_m5_aggregate() -> pd.DataFrame:
    """Load the aggregated M5 panel: 10 dense daily series, 756 days each.

    Nine ``state_id x cat_id`` sums (``CA_FOODS``, ``CA_HOBBIES``, ...) plus a national ``TOTAL``,
    derived from the M5 Forecasting — Accuracy competition dump by summing all 30 490 item-store
    series into buckets that are dense and strictly positive, so ordinary MAPE is well-defined on
    them (raw M5 series are intermittent and zero-inflated). Built by
    ``playground/m5_prototype/m5_make_accuracy_fixture.py``; see that script's docstring for the
    Christmas-closure repair and other provenance details. Backs the accuracy guard in
    ``tests/test_m5_accuracy.py``, which also asserts this file's shape and content digest.

    Returns
    -------
    pd.DataFrame
        Long panel with columns ``date`` (datetime64), ``series_id`` (str), ``sales`` (int),
        sorted by ``(date, series_id)``.
    """
    return pd.read_csv(M5_AGGREGATE_PATH, parse_dates=["date"])
