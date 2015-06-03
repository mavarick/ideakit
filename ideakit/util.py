# coding=utf-8
import pandas as pd
import numpy as np

def is_numerical_type(series_or_arr):
    '''判断是否是数值类型的
    '''
    # TODO: support other argument type
    type_ = series_or_arr.dtype.type
    # TODO: consider whether bool is numerical
    return (issubclass(type_, (np.number, np.bool_)) \
        and not issubclass(type_, (np.datetime64, np.timedelta64)))
 
def is_categorical_type(series_or_arr):
    '''判断是否是离散类型的数值
    '''
    # TEMP: NOT is_numerical <==> is_categorical
    # TODO: better handle different types
    return (str(series_or_arr.dtype) in ['category'])

def is_numerical_value(value):
    if value is None:
        return False
    return issubclass(np.dtype(type(value)).type, np.number)

def is_valid_numerical_value(value):
    return is_numerical_value(value) and (not np.isnan(value)) and (not np.isinf(value))
