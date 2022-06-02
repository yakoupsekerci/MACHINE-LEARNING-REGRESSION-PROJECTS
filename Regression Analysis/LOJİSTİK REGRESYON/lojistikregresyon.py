# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv('kredi_veriseti.csv')
print(dataset.head())

# katagorik veriyi sayılarla temsil edelim cinsiyet verisi katagorik bir veridir.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder(handle_unknown='ignore')
one_hot_cinsiyet = one_hot.fit_transform(dataset['cinsiyet'].values.reshape(-1,1)).toarray()
a = one_hot.categories_
one_hot_df = pd.DataFrame(one_hot_cinsiyet, columns=a)
one_hot_df2 = dataset.join(one_hot_df)
one_hot_df2.drop('cinsiyet',axis=1,inplace=True)
print(one_hot_df2)

X = one_hot_df2.drop('kredi',axis=1)
Y = one_hot_df2['kredi']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
sc = StandardScaler()
X_test = sc.fit_transform(X_test)
X_train = sc.fit_transform(X_train)


# modelleme aşaması

model = LogisticRegression(random_state=0)
model.fit(X_train, Y_train)

# tahmin aşamamsı
y_pred = model.predict(X_test)
print(y_pred)
print(Y_test.values)

from sklearn.metrics import  confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_true=Y_test,y_pred=y_pred))
print(accuracy_score(y_true=Y_test,y_pred=y_pred))


