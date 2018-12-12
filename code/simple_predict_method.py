# -*- coding:utf-8 -*-

import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt 
import statsmodels.api as sm
import pyramid as pm

# 简单加权指数平滑预测
# 可以直接调用SimpleExpSmoothing方法
def ses_method(s, a):
    s = s.values
    s2 = np.zeros(s.shape)
    s2[0] = s[0]
    for i in range(len(s)-1):
        s2[i+1] = a*s[i]+(1-a)*s2[i]

    return s2

# 非季节性+趋势的holt预测
def holt_method(s, pre_len, a, b):
    fit1 = Holt(np.asarray(s)).fit(smoothing_level=a, smoothing_slope=b)

    return fit1.forecast(pre_len)

# 季节性+趋势性的holt-winter预测
def holt_winter_method(s, pre_len, season_period, trend='add', seasonal='add'):
    fit1 = ExponentialSmoothing(np.asarray(s), seasonal_periods=season_period, 
                                trend=trend, seasonal=seasonal).fit()

    return fit1.forecast(pre_len)

# ETS方法，python中没有ETS方法，可以借助R相关的包进行调用，或者下载R进行预测

# auto.arima需要下载pyramid库
# 可参考https://www.alkaline-ml.com/pyramid/user_guide.html
def autoarima_method(s, pre_len, season_period):
    stepwise_fit = pm.auto_arima(np.asarray(s), start_p=1, start_q=1,
                             max_p=3, max_q=3, m=season_period,
                             start_P=0, seasonal=True,
                             d=1, D=1, trace=True,
                             error_action='ignore',  # don't want to know if an order does not work
                             suppress_warnings=True,  # don't want convergence warnings
                             stepwise=True) 

    return stepwise_fit.predict(pre_len)

# 季节性SARIMAX模型
def sarimax_method(s, pre_len, p, d, q, P, D, Q, m):
    fit1 = sm.tsa.statespace.SARIMAX(np.asarray(s), order=(p, d, q), 
                                     seasonal_order=(P,D,Q,m)).fit()    

    return fit1.predict(pre_len)