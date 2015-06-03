# coding=utf-8
import pandas as pd
import numpy as np
from util import *

def calc_bounds_zscore(series, max_abs_zscore=3.0):
    """Calculate (lowerbound, upperbound) values based on zscore.
    NOT robust because extreme outliers lead to overestimated variance and range.
    """
    mean, std = series.mean(), series.std()
    if is_valid_numerical_value(mean) and is_valid_numerical_value(std):
        return (mean - max_abs_zscore * std,  mean + max_abs_zscore * std)
    else:
        return None

def calc_bounds_iqr(series, ignore_invalid_values=True):
    """Calculate (lowerbound, upperbound) values based on interquartile range.
    More robust than calc_bounds_zscore.
    """
    # 1st & 3rd quartiles
    # NOTE: pd.Series.quantile() can handle nan/inf, but 6X SLOWER than np.percentile()
    q1, q3 = series.quantile([0.25, 0.75]) if ignore_invalid_values else \
             np.percentile(series, [25, 75])

    if is_valid_numerical_value(q1) and is_valid_numerical_value(q3):
        expanded_iqr = 1.5 * (q3 - q1)
        return (q1 - expanded_iqr, q3 + expanded_iqr)
    else:
        return None

def calc_bounds(series):
    """Calculate (lowerbound, upperbound) values based on adaptive criterion."""
    is_close_to_gaussian = False # TODO: implement gaussian test
    if is_close_to_gaussian:
        return calc_bounds_zscore(series)
    else:
        return calc_bounds_iqr(series)

def find_outliers(series, func_calc_bounds=calc_bounds_iqr):
    """Find outlier values in a numerical series.
    
    Parameters
    ----------
    series : pandas.Series
        series to process
    func_calc_bounds : 
        function used to calculate bounds for outlier finding
        or use 'lambda x: (lower, upper)' to specify fixed bounds

    Returns
    -------
    outliers : pandas.Series
        a bool series with 'True' for outliers and 'False' for inliers
        None if failed to found valid lowerbound/upperbound
    """
    lowerbound, upperbound = func_calc_bounds(series)
    if is_valid_numerical_value(lowerbound) and is_valid_numerical_value(upperbound) \
        and lowerbound <= upperbound:
        return (series < lowerbound) | (series > upperbound)

    return None

def get_minmax(series):
    """Get min/max values in a numerical series.
    
    Parameters
    ----------
    series : pandas.Series
        series to process

    Returns
    -------
    minmax : tuple of (min_val, max_val)
    """
    return (series.min(), series.max())

def get_minmax_exclude_outliers(series, func_calc_bounds=calc_bounds_iqr):
    """Get min/max values excluding outlier values in a numerical series.
    
    Parameters
    ----------
    series : pandas.Series
        series to process
    func_calc_bounds : 
        function used to calculate bounds for outlier finding
        or use 'lambda x: (lower, upper)' to specify fixed bounds

    Returns
    -------
    minmax : tuple of (min_val, max_val)
        min/val of the entire series if failed to find outliers
    """
    outliers = find_outliers(series, func_calc_bounds)
    if outliers is None:
        return get_minmax(series) # failed in outlier finding
    else:
        return get_minmax(series[~outliers]) # calc minmax from inliers
