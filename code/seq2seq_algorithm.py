# -*- coding: utf-8 -*-
"""
@author: Cai Chengfei
@license: (C) Copyright 1997-2018, NetEase, Inc.
@contact: caichengfei@corp.netease.com
@time: 9/2018
"""
"""This module is complementation of Seq2Seq class."""

import warnings
import numpy as np
from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq

warnings.filterwarnings("ignore")  # ignore the warning information

class Seq2seqAlogrithm:
    """Seq2seq alogrithm class here.

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
    
    def __init__(self, X, Y, pre_time, input_len, input_dim, output_len, output_dim):
        self.train_X = X[:-pre_time + output_len]
        self.train_Y = Y[:-pre_time + output_len]
        self.pre_time = pre_time
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.output_dim = output_dim

    def s2s_model(self):
        """Train s2s model here.

        Returns:
            A s2s model that can be used to predict result.

        Raises:
            IOError: An error occurred training xgb model.
        """
        
        model = AttentionSeq2Seq(input_dim=self.input_dim, input_length=self.input_len, 
                                 hidden_dim=16, output_length=self.output_len, 
                                 output_dim=self.output_dim, depth=(1,1),
                                 stateful=False, dropout=0.5)
        model.compile(loss='mape', optimizer='adam', metrics=['mse'])
        model.fit(self.train_X, self.train_Y, epochs=75, verbose=2, shuffle=True)

        return model 
    
    def s2s_predict(self, gmv_data, model):
        """Use s2s model to redict result here.

        Returns:
            The result used model to predict. Format of result
            is M x 1 array. M is the predict time length.

        Raises:
            IOError: An error occurred predict result using s2s model.
        """  

        test_X1 = gmv_data[-self.pre_time-self.input_len:
                           -self.pre_time].reshape(1, -1, self.input_dim)
        pre1 = model.predict(test_X1) 

        test_X2 = np.concatenate((test_X1[0][14:], pre1[0])).\
            reshape(1, -1, self.input_dim)
        pre2 = model.predict(test_X2)

        test_X3 = np.concatenate((test_X2[0][7:], pre1[0][0:7])).\
            reshape(1, -1, self.input_dim)
        pre3 = model.predict(test_X3)

        pre = np.concatenate((pre1[0], pre2[0], pre3[0][7:]))

        return pre
    
