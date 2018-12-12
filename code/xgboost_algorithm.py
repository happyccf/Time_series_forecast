# -*- coding: utf-8 -*-
"""
@author: Cai Chengfei
@license: (C) Copyright 1997-2018, NetEase, Inc.
@contact: caichengfei@corp.netease.com
@time: 7/2018
"""
"""This module is complementation of XgbAlogrithm class."""

import warnings
import xgboost as xgb
import sklearn
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")  # ignore the warning information

class XgbAlogrithm:
    """Xgboost alogrithm class here.

    This class contains training and predicting method used to 
    analysis time series.

    Attributes:
        xgb_feature: features used to train model and predict result.
            Format of features is M x N array, M is time length, N is
            the length of each feature. 
        pre_time: time length of predict result i.e. 35 in this program.
         
    Method:
        xgb_model: using features to train xgboost model.
        xgb_predict: using xgboost model to predict result.
    """
    
    def __init__(self, xgb_feature, xgb_label, pre_time):
        self.xgb_feature = xgb_feature
        self.xgb_label = xgb_label
        self.pre_time = pre_time
        
    def xgb_model(self):
        """Train xgboost model here.

        Returns:
            A xgb model that can be used to predict result.

        Raises:
            IOError: An error occurred training xgb model.
        """
        
        xgb_feature = self.xgb_feature
        xgb_label = self.xgb_label
        pre_time = self.pre_time

        train_Y = xgb_label.values[: -pre_time]
        train_X = xgb_feature.values[: -pre_time]

        learning_rate = 0.01
        subsample = 1
        colsample_bylevel = 1
        scale_pos_weight = 1
        random_state = 1000
        gamma = 0
        param_1 = {
            #  Use below prameter when your CPU computing ability is powerful 
            
            'n_estimators': range(180, 350, 5),
            'max_depth': range(4, 16, 2),
            'min_child_weight': range(3, 11, 2)
        }

        gsearch = GridSearchCV(
            estimator = xgb.XGBRegressor(
                gamma = gamma, 
                scale_pos_weight = scale_pos_weight,
                learning_rate = learning_rate, 
                subsample = subsample, 
                colsample_bylevel = colsample_bylevel,
                random_state = random_state, 
                reg_lambda = 2, 
                silent = 1, 
                nthread = 4), 
            param_grid = param_1, 
            scoring = "neg_mean_absolute_error",
            n_jobs = 8, 
            cv = 5)
        gresult = gsearch.fit(train_X, train_Y)
        
        model = xgb.XGBRegressor(
            n_estimators = gresult.best_params_['n_estimators'],  
            max_depth = gresult.best_params_['max_depth'],
            min_child_weight = gresult.best_params_['min_child_weight'],  
            gamma = gamma, 
            scale_pos_weight = scale_pos_weight,
            learning_rate = learning_rate, 
            subsample = subsample, 
            colsample_bylevel = colsample_bylevel, 
            random_state = random_state, 
            reg_lambda = 2,
            silent = 1,  
            n_jobs = 8)
        model.fit(train_X, train_Y)

        return model
    
    def xgb_predict(self, model):
        """Use xgb model to redict result here.

        Returns:
            The result used model to predict. Format of result
            is M x 1 array. M is the predict time length.

        Raises:
            IOError: An error occurred predict result using xgb model.
        """  
        
        xgb_feature = self.xgb_feature
        pre_time = self.pre_time

        test_X = xgb_feature.values[-pre_time:]
        result = model.predict(test_X) 

        return result
    
