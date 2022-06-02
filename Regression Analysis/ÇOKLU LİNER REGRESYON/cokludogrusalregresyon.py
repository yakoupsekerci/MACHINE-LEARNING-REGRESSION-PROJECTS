# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 02:52:09 2022

@author: yakou
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

data = pd.read_csv("emlak_fiyat_boyut_metro.csv")

x = data[['boyut','metro_yakinlik']]
y = data['fiyat']

plt.scatter(x['boyut'],y)
plt.xlabel('boyut',fontsize=15)
plt.ylabel('fiyat',fontsize=15)
plt.show()

plt.scatter(x['metro_yakinlik'],y)
plt.xlabel('metroya yakinlik',fontsize=15)
plt.ylabel('fiyat',fontsize=15)
plt.show()
        
 # korolasyona bak.
correlation = data.corr()
plt.figure(figsize=(10,10))
a = sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8},cmap='Blues')
print(a)




regressor = LinearRegression()
regressor.fit(x,y)
Y_pred = regressor.predict(x)


plt.plot(y,label="GERÇEK DEĞERLER")
plt.plot(Y_pred,label="TAHMİN DEĞERLERİ")
plt.legend()


from sklearn.metrics import r2_score
print("R2 SCORE: ",r2_score(y,Y_pred))

print("REGRESYON SABİTİ...:",regressor.intercept_)
print("REGRESYON KATSAYILARI...:",regressor.coef_)#0. İNDİS B0 1. İNDİS B1 DİR 2. DEĞİŞKEN OLDUĞU İÇİN X DEĞERİ 2 ELAMNLARI BİR DİZİ VERİR.


# 120 METREKARELİK VE METROYA 10 DAKKA MESAFADE BİR DAİRENİN TAHMİNİ FİYATINI BUL.

b = regressor.predict([[120,10]])
print(b)

# bir veri setini tahmin et dizi olarak bana sonuç ver.

new_data = pd.DataFrame({'boyut':[100,150,200],'metro_yakinlik':[30,10,5]})
print(new_data)
asd = regressor.predict(new_data)
print(asd)
new_data['Tahmini_Fiyat'] = regressor.predict(new_data)
print(new_data)
