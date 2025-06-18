from copy import deepcopy
import pandas as pd
import numpy as np
from typing import List, Optional, Union


class RegressionSchemeConstants:
    REGRESSOR = "regressor"
    SIGN = "sign"
    LOC_PRIOR = "loc_prior"
    SCALE_PRIOR = "scale_prior"

    RegressionSignTypes = ["=", "+", "-"]
    RegressionSignMapping = {
        "=": "Regular",
        "+": "Positive",
        "-": "Negative",
    }


class RegressionScheme:
    """A Data class to store regression scheme"""

    def __init__(
        self,
        scheme: Optional[pd.DataFrame] = None,
    ) -> None:
        """

        Args:
            scheme : pd.DataFrame

            - Index: [RegressionSchemeConstants.REGRESSORS]

            - Columns:
                SIGN : str[=, +, -], =, +, - maps to regular(neutral), positive, negative respectively
                LOC_PRIOR : float,
                SCALE_PRIOR : float,
        """

        self.create_empty_scheme()
        if scheme is not None:
            self.set_scheme(scheme)

    def create_empty_scheme(self) -> None:
        self.scheme = pd.DataFrame(
            columns=[
                RegressionSchemeConstants.REGRESSOR,
                RegressionSchemeConstants.SIGN,
                RegressionSchemeConstants.LOC_PRIOR,
                RegressionSchemeConstants.SCALE_PRIOR,
            ],
            index=[],
        ).set_index(RegressionSchemeConstants.REGRESSOR)

    @staticmethod
    def validate_scheme(scheme: pd.DataFrame) -> bool:
        required_columns = {
            RegressionSchemeConstants.SIGN,
            RegressionSchemeConstants.LOC_PRIOR,
            RegressionSchemeConstants.SCALE_PRIOR,
        }
        if not required_columns.issubset(scheme.columns):
            raise ValueError(f"Scheme must have columns: {required_columns} and index name as regressor")

        if scheme.index.name != RegressionSchemeConstants.REGRESSOR:
            raise ValueError(f"Scheme must have index name as {RegressionSchemeConstants.REGRESSOR}")
        if not scheme[RegressionSchemeConstants.SIGN].isin(RegressionSchemeConstants.RegressionSignTypes).all():
            raise ValueError(
                f"Scheme must have {RegressionSchemeConstants.SIGN} "
                f"column with values in {RegressionSchemeConstants.RegressionSignTypes}"
            )

    def set_scheme(self, scheme: pd.DataFrame) -> None:
        self.validate_scheme(scheme)
        self.scheme = deepcopy(scheme)

    def get_scheme(self) -> pd.DataFrame:
        return deepcopy(self.scheme)

    def update_scheme(self, scheme: pd.DataFrame) -> None:
        """Given a new scheme, add new indexes if not present, update existing indexes if present"""
        self.validate_scheme(scheme)
        # this will
        # 1. add new indexes if not present
        # 2. update existing indexes if present using preferred values from new scheme
        self.scheme = scheme.combine_first(self.scheme)

    def add_regressors(
        self,
        regressors: List[str],
        coef_sign: Union[str, List[str]] = "=",
        loc_prior: Union[float, List[str]] = 0.0,
        scale_prior: Union[float, List[float]] = 1.0,
    ) -> None:
        """A lazy interface to add multiple regressors at once with flat values of priors"""

        new_scheme_to_combine = pd.DataFrame(
            {
                RegressionSchemeConstants.REGRESSOR: regressors,
                RegressionSchemeConstants.SIGN: coef_sign,
                RegressionSchemeConstants.LOC_PRIOR: loc_prior,
                RegressionSchemeConstants.SCALE_PRIOR: scale_prior,
            }
        ).set_index(RegressionSchemeConstants.REGRESSOR)

        self.update_scheme(new_scheme_to_combine)

    def get_regressors(self) -> List[str]:
        """Get a list of regressors in the scheme"""
        return deepcopy(self.scheme.index.tolist())


def make_fourier_series_with_index(n: int, period: Union[int, float], order: int = 3, shift: int = 0) -> np.ndarray:
    """Given time series length, cyclical period and order, return a set of fourier series.

    Parameters
    ----------
    n : int
        Length of time series
    period : float
        Length of a cyclical period. E.g., with daily data, `period = 7` means weekly seasonality.
    order : int
        Number of components for each sin() or cos() series.
    shift : int
        shift of time step/index to generate the series

    Returns
    -------
    2D array-like
        2D array in shape (n, 2 * order) where each column represents the series with a specific order 
        fourier constructed by cos(i) for i = 1, 2, ... order then sin(j) for j = 1, 2, ... order. 
    Notes
    -----
        1. See https://otexts.com/fpp2/complexseasonality.html
        2. Original idea from https://github.com/facebook/prophet under
    """

    # (n, 1)
    t = np.expand_dims(np.arange(1, n + 1) + shift, -1)
    # shape (order,)
    i = np.arange(1, order + 1)  
    # shape (n, order)
    x = 2.0 * np.pi * t * i / period  
     # shape (n, 2 * order)
    out = np.concatenate([np.cos(x), np.sin(x)], axis=1) 
    return out


def make_fourier_series_with_ts(dt_arr: pd.Series, period: Union[int, float], order: int = 3) -> np.ndarray:
    """Given time series length, cyclical period and order, return a set of fourier series.

    Parameters
    ----------
    dt_arr
        Time series
    period
        Length of a cyclical period. E.g., with daily data, `period = 7` means weekly seasonality.
    order
        Number of components for each sin() or cos() series.

    Returns
    -------
    2D array-like
        2D array where each column represents the series with a specific order fourier constructed by sin() or cos().

    Notes
    -----
    1. See https://otexts.com/fpp2/complexseasonality.html
    2. Original idea from https://github.com/facebook/prophet under
    """
    # constant converting nanoseconds to seconds
    nanosec_to_sec = 1000 * 1000 * 1000
    # convert to days since epoch
    t = dt_arr.to_numpy(dtype=np.int64) // nanosec_to_sec / (3600 * 24.0)
    # (n, 1)
    t = np.expand_dims(t, -1)

    # shape (order,)
    i = np.arange(1, order + 1)  
    # shape (n, order)
    x = 2.0 * np.pi * t * i / period  
     # shape (n, 2 * order)
    out = np.concatenate([np.cos(x), np.sin(x)], axis=1) 
    return out
