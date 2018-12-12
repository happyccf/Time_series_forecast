# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb

# 随机森林方法，这里参数都使用默认参数
def rf_method(train_x, train_y, test_x):
    rf=RandomForestRegressor()
    rf.fit(train_x, train_y)

    return(rf.predict(test_x))

# gbdt方法，这里简单调用函数，具体寻参等可类似于xgboost_algorithm.py文件
def gbdt_method(train_x, train_y, test_x):
    gbr = GradientBoostingClassifier(n_estimators=3000, max_depth=2, 
                                     min_samples_split=2, learning_rate=0.1)
    gbr.fit(train_x, train_y)

    return(gbr.predict(test_x))

# lightgbm方法，这里简单调用函数
def gbm_method(train_x, train_y, test_x):
    gbm = lgb.LGBMRegressor(objective='regression',num_leaves=31,learning_rate=0.05,n_estimators=20)
    gbm.fit(train_x, train_y)

    return(gbm.predict(test_x, num_iteration=gbm.best_iteration_))

# xgb方法参考xgboost_algorithm.py文件