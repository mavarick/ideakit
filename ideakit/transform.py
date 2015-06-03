# coding=utf-8
import pandas as pd
import numpy as np
from util import *
from outlier import find_outliers, get_minmax, get_minmax_exclude_outliers

def transform_log(series, robust=True):
    """Perform element-wise logarithm transformation in a numerical series.
    
    Parameters
    ----------
    series : pandas.Series
        series to transform
    robust : bool
        True - handle negative and zero values properly
        False - transform negative value to nan, zero to -inf

    Returns
    -------
    log_series : pandas.Series
        ANOTHER series consisting of the transformed values
    """
    # TODO: support log10
    # TODO: separate log1p and log explicitly
    if not isinstance(series, pd.Series):
        raise TypeError("argument 'series' is NOT 'pandas.Series' type")
    if not is_numerical_type(series):
        raise ValueError("value type of argument 'series' is NOT numerical")
    
    if robust:
        return series.apply(lambda x: np.log1p(x) if x>=0 else -np.log1p(-x))
    else:
        return series.apply(np.log)

def transform_minmax(series, exclude_outliers=False):
    """Perform element-wise minmax transformation in a numerical series, where
        minmax(v) = (v - min_{v}) / (max_{v} - min_{v})
    
    Parameters
    ----------
    series : pandas.Series
        series to transform
    exclude_outliers : bool
        True - exclude outliers when calculating min & max; outliers are transformed to 0 or 1
        False - calculate min & max on entire series

    Returns
    -------
    minmax_series : pandas.Series
        ANOTHER series consisting of the transformed values
    """
    if not isinstance(series, pd.Series):
        raise TypeError("argument 'series' is NOT 'pandas.Series' type")
    if not is_numerical_type(series):
        raise ValueError("value type of argument 'series' is NOT numerical")

    vmin, vmax = get_minmax_exclude_outliers(series) if exclude_outliers else get_minmax(series)
    vrange = max(1e-6, vmax - vmin) # TODO: better handle min == max
    
    ret = series.apply(lambda v: (v - vmin) / vrange)
    if exclude_outliers: # bound outliers
        ret[series < vmin] = 0.0
        ret[series > vmax] = 1.0
    # TODO: handle NaNs
    return ret

def transform_zscore(series, exclude_outliers=False):
    """Perform element-wise z-score transformation in a numerical series, where
        zscore(v) = (v - mean_{v}) / std_{v}
    
    Parameters
    ----------
    series : pandas.Series
        series to transform
    exclude_outliers : bool
        True - exclude outliers when calculating mean & std
        False - calculate mean & std on entire series

    Returns
    -------
    minmax_series : pandas.Series
        ANOTHER series consisting of the transformed values
    """
    if not isinstance(series, pd.Series):
        raise TypeError("argument 'series' is NOT 'pandas.Series' type")
    if not is_numerical_type(series):
        raise ValueError("value type of argument 'series' is NOT numerical")

    inlier_series = None
    if exclude_outliers:
        outliers = find_outliers(series)
        if outliers is not None:
            inlier_series = series[~outliers]

    vmean, vstd = (series.mean(), series.std()) if inlier_series is None else \
                    (inlier_series.mean(), inlier_series.std())
    vstd = max(1e-6, vstd) # TODO: replace hard-code

    # TODO: handle NaNs
    return series.apply(lambda v: (v - vmean) / vstd)
