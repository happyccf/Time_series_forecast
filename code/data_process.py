# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 自相关和偏相关图，默认阶数为31阶
def draw_acf_pacf(ts, lag=31):
    f = plt.figure(figsize=(10, 5), facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags=lag, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=lag, ax=ax2)
    plt.show()

# 进行时序分解，model 0：加性分解 1：乘性分解
def decompose(ts, model=0):
    plt.figure(figsize=(10, 5))
    if model:
        sm.tsa.seasonal_decompose(ts, model="multiplicative").plot()
    else:
        sm.tsa.seasonal_decompose(ts).plot()
    plt.show()
