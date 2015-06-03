#!/usr/bin/env python
#encoding:utf8
import pandas as pd

def describe(df):
    ''' generate relevant information of df
    @Parameters: df
    @Returns: %cov, uniq, dtype, count
    '''
    # s_stat = df.describe().transpose().iloc[:,:2])
    s_stat = df.apply(pd.Series.describe).transpose()[['count']]#, 'min']]
    s_type = pd.DataFrame(df.dtypes, columns=['dtype'])
    s_uniq = pd.DataFrame(df.apply(pd.Series.nunique), columns=['uniq'])
    s_covr = pd.DataFrame(df.apply(lambda x: (x.notnull().mean() * 100).round(2)), columns=['%cov'])
    # s_top  = pd.DataFrame(df.apply(lambda x: x.value_counts(dropna=False).index[0] \
        # if x.dtype.type is np.object_ else '_OBJ_'), columns=['top'])
    return pd.concat([s_type,s_covr,s_uniq], axis=1).join(s_stat)

def set_category(data_frame, cat_columns_names):
    ''' 设置df中的columns为category类型的
    '''
    for name in cat_columns_names:
        data_frame[name] = data_frame[name].astype('category')
    return data_frame

def extend_features(df):
    '''扩展dataframe的features
    如果是数值类型的，那么不进行处理
    如果是离散类型的，那么进行dummy扩展
    TODO: 是否判断离散类型的覆盖率或者相关的合并
    '''
    new_df = []
    for name in df.columns:
        ser = df[name]
        ser_type = str(ser.dtype)
        if ser_type in ['int64', 'int', 'float64', 'float32', 'float']:
            # 数值类型
            new_ser = extend_numerical_feature(ser)
        if ser_type in ['category']:
            new_ser = extend_categorical_feature(ser)
        new_df.append(new_ser)
    new_df = pd.concat(new_df, axis=1)
    return new_df

def extend_numerical_feature(ser):
    '''对数值类型的数据进行处理，包括：
    1, 是否处理缺失值
    2，是否对最大最小值进行处理；
    3，是否进行平滑
    # TODO
    '''
    return ser

def extend_categorical_feature(ser):
    ''' 对离散类型的数值进行处理，包括：
    1，进行扩展；pandas.get_dummy
    2，对比例小的数值是否进行合并等
    [x], 对缺失值的处理. 
    # TODO
    '''
    #ser = ser.fillna(na_value) # 这个地方如果na_value不在category范围内，会raise
    return pd.get_dummies(ser, prefix=ser.name, prefix_sep="#", dummy_na=True)




