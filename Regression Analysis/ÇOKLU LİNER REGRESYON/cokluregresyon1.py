# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 02:34:32 2022

@author: yakou
"""

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score

boston = pd.read_csv('BostonHousing.csv')
boston.drop(["nox","rad"],axis=1, inplace=True)

X = boston.iloc[:, :-1].values
Y = boston.iloc[:, 11]

print (Y)

for col in boston.columns:
    en_yuksek_degerler = abs(boston.corr()[col]).nlargest(n=5)
    print(en_yuksek_degerler)
    for index, value in en_yuksek_degerler.items():
        if 1 > value >= 0.75:
            print(index,col,"DEĞİŞKENLERİ YÜKSEK KOROLASYONA SAHİP",value)
            """
            en yuksek korolosyona sahipleri çıkardık.
            """

"""
a = sns.heatmap(boston.corr(),cmap = 'BrBG', annot=True)
print(a)

a = boston.corr()
print(a)
"""
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,Y)

print(regressor.coef_)
print(regressor.intercept_)

Y_pred = regressor.predict(X)

print("REGRESYON SABİTİ...:",regressor.intercept_)
print("REGRESYON KATSAYILARI...:",regressor.coef_)
"""
print(Y_pred) # tahmin ettiği değerler eğitim verisine göre

print(Y)
"""
from sklearn.metrics import r2_score
print("R2 SCORE: ",r2_score(Y,Y_pred))

plt.plot(Y,label="GERÇEK DEĞERLER")
plt.plot(Y_pred,label="TAHMİN DEĞERLERİ")
plt.legend()

