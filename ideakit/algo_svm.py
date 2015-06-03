#!/usr/bin/env python
#encoding:utf8
import pandas as pd

from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from feature_eng import eval_auc, eval_ks
# TRAIN
'''notice:
1，在训练的数据集中不应该包含NaN，Inf等非可计算数值
'''
def cv(x, y, model, K=4):
    '''cross validation for model 
    @Parameters
        x: numpy.array, with shape: (n, m)
        y: numpy.array, with shape: (n,)
        model: model, 
            e.x.: model = RandomForestClassifier(n_jobs=-1, n_estimators=100, 
            criterion='entropy', max_depth=8, random_state=123)
        K: n_folds
    @Return
        info: trained result info
    '''
    skf = cross_validation.StratifiedKFold(y, n_folds=K, shuffle=True, random_state=0)

    result_list = []
    for train_index, test_index in skf:
        x_train, y_train = x[train_index], y[train_index]
        model.fit(x_train, y_train)

        x_test, y_test = x[test_index], y[test_index]
        y_pred = model.predict_proba(x_test)[:, 1]
        auc = eval_auc(y_test, y_pred)
        info = {
            "train_cnt": len(y_train),
            "train_pos_cnt": y_train.sum(),
            "train_pos_ratio": y_train.sum() * 1.0 / len(y_train),
            "test_cnt": len(y_test),
            "test_pos_cnt": y_test.sum(),
            "test_pos_ratio": y_test.sum() * 1.0 / len(y_test),
            'auc': auc
        }
        result_list.append(pd.Series(info))
    return pd.concat(result_list, axis=1)

def train(x, y, model):
    '''training and return trained model
    '''
    model.fit(x, y)
    return model 

def predict(model, x):
    y_pred = model.predict_proba(x)[:,1]
    return y_pred

import pickle
def serialize(model, file_path):
    # check if the file_path exist, TODO
    fp = open(file_path, 'w')
    pickle.dump(model, file_path, protocol=0)
    fp.close()
    return 0

def deserialize(file_path):
    fp = open(file_path)
    model = pickle.load(fp)
    return model









# TEST
