# coding=utf-8
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import metrics
from outlier import find_outliers
import util
from constants import LABEL_COLUMN_NAME, BINARY_LABEL_TH

def eval_features(dataframe, label_col_name=LABEL_COLUMN_NAME):
    y_series = dataframe[label_col_name]
    evals = [eval_feature(y_series, dataframe[col]) for col in dataframe.columns \
            if col != label_col_name]
    ret = pd.DataFrame(evals, columns=['name', 'type', '%cov', 'iv', 'auc', 'ks', 'ks_p'])
    for col in ret.columns[2:]:
        ret[col] = ret[col].round(3)
    return ret

def eval_feature(y_series, x_series):
    assert(isinstance(y_series, pd.Series))
    assert(isinstance(x_series, pd.Series))
    
    ret = {'name': x_series.name}

    #total_num = x_series.size
    total_num = len(x_series)
    valid_num = x_series.count()
    ret['%cov'] = float(valid_num) / total_num; # proportion of non-NaN values

    # convert labels to {0,1}
    y = y_series.round().astype(int).apply(lambda v: 1 if v > BINARY_LABEL_TH else 0).values

    if util.is_categorical_type(x_series): # categorical => iv
        ret['type'] = 'C'
        ret['iv'] = eval_iv(y, x_series)
    else: # numerical => iv, auc, ks, ks_p
        ret['type'] = 'N'
        # ret['inlier'] = float(valid_num - find_outliers(x_series).sum()) / total_num # inlier rate

        num_iv_bins = 5 # PARA
        iv_smoother_per_bin = 10 # PARA
        x_bin_series = (x_series.rank(na_option='keep', pct=True) * num_iv_bins).round() # TEMP
        ret['iv'] = eval_iv(y, x_bin_series, count_smoother=iv_smoother_per_bin)
        
        x = x_series.values
        ret['auc'] = eval_auc(y, x) # directly use 'x' as predicted scores
        ks, ks_p = eval_ks(x[y > BINARY_LABEL_TH], x[y <= BINARY_LABEL_TH]) # ks with p-value
        ret['ks'] = ks
        ret['ks_p'] = ks_p

    return ret

def eval_iv(y, x_series, count_smoother=1):
    """Evaluate Information-Value from ground-truth labels and feature values.
    
    Parameters
    ----------
    y : np.ndarray
        ground-truth, with 2 unique values
    x_series : np.ndarray
        continuous predictions
    count_smoother : int
        pseudo count added to each bin, to avoid infinite WoE

    Returns
    -------
    auc :
    """

    x_str_series = None # TEMP: convert to string so as to preserve NAs
    try:
        x_str_series = x_series.astype(str) 
        pass
    except Exception, e:
        x_str_series = x_series.astype(unicode)
        
    pos_value_counts = x_str_series[y > BINARY_LABEL_TH].value_counts() #dropna=False)
    neg_value_counts = x_str_series[y <= BINARY_LABEL_TH].value_counts() #dropna=False)
    distr = pd.concat([pos_value_counts, neg_value_counts], axis=1)
    
    distr.fillna(0, inplace=True) # fill NA due to outer join
    distr += count_smoother # avoid empty bins
    distr[0] /= distr[0].sum()
    distr[1] /= distr[1].sum()

    # TODO: bound woe instead of add count smoother
    woe = np.log(distr[0] / distr[1]) # woe of each bin
    return ((distr[0] - distr[1]) * woe).sum()

def eval_auc(y_truth, y_score):
    """Evaluate AUC score from ground-truth labels and predicted scores.
    
    Parameters
    ----------
    y_truth : np.ndarray
        ground-truth, with 2 unique values
    y_score : np.ndarray
        continuous predictions

    Returns
    -------
    auc : float
    """
    assert(isinstance(y_truth, np.ndarray))
    assert(isinstance(y_score, np.ndarray))
    # NOTE: must exclude NaNs
    indices = ~np.isnan(y_score)
    return metrics.roc_auc_score(y_truth[indices], y_score[indices])

def eval_ks(arr1, arr2):
    """Computes the Kolmogorov-Smirnov statistic on 2 samples.
    
    Parameters
    ----------
    arr1 : np.ndarray
    arr2 : np.ndarray

    Returns
    -------
    D : float
        KS statistic
    p-value : float
        two-tailed p-value
    """
    assert(isinstance(arr1, np.ndarray))
    assert(isinstance(arr2, np.ndarray))
    # NOTE: must exclude NaNs
    return stats.ks_2samp(arr1[-np.isnan(arr1)], arr2[-np.isnan(arr2)])



