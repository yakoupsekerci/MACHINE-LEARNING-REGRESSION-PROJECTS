# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 23:22:37 2022

@author: yakou
"""

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score
import seaborn as sns


wine = pd.read_csv('kirmizisarap.csv')
"""
plt.figure(figsize=(10,4))
sns.barplot(x="quality",y="fixed acidity",data=wine)
"""
"""
correlation = wine.corr()
plt.figure(figsize=(10,10))
a = sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8},cmap='Blues')
print(a)
"""

for col in wine.columns:
    en_yuksek_degerler = abs(wine.corr()[col]).nlargest(n=5)
    print(en_yuksek_degerler)
    for index, value in en_yuksek_degerler.items():
        if 1 > value >= 0.65:
            print(index,col,"DEĞİŞKENLERİ YÜKSEK KOROLASYONA SAHİP",value)
X = wine.iloc[:, :-1].values
Y = wine.iloc[:, 11]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)


"""
print(Y_pred) # tahmin ettiği değerler eğitim verisine göre

print(Y)
"""
from sklearn.metrics import r2_score
print("R2 SCORE: ",r2_score(Y_test,Y_pred))



plt.plot(Y_test,label="GERÇEK DEĞERLER")
plt.plot(Y_pred,label="TAHMİN DEĞERLERİ")
plt.legend()
print(X.info)


