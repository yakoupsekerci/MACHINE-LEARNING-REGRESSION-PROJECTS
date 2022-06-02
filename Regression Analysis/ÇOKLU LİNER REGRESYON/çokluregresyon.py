# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 03:02:47 2022

@author: yakou
"""

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score

dataset = pd.read_csv('data.csv')
"""
a = sns.heatmap(dataset.corr(),cmap = 'BrBG', annot=True)
print(a)
"""

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
"""
VERİ TABANINDAKİ KATAGORİK VERİYİ SAYISALLAŞTIRMAMIZ GEREKMEKTEDİR.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encode',OneHotEncoder(),[3])],remainder='passthrough')
X=ct.fit_transform(X)

X = X[:,1:]
print(X)

"""
yeni veriseti newyork = 2 column  california = 0 column florida = 1. column   
"""

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)

print("R^2:",r2_score(Y_test,Y_pred)) 
print(regressor.intercept_)
print(regressor.coef_)

plt.plot(Y_test,label="GERÇEK DEĞERLER")
plt.plot(Y_pred,label="TAHMİN DEĞERLERİ")
plt.legend()

print(Y_test)
print(Y_pred)









