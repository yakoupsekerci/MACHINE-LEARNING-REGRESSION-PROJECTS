# -*- coding: utf-8 -*-
"""
Created on Thu May 19 19:20:47 2022

@author: yakou
"""

import pickle 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



data = pd.read_csv("data/titanic_train.csv")

data = data.drop(['Name', 'PassengerId','Cabin','Ticket'], axis=1)

print(data.info())

for col in data.select_dtypes(include="int64").columns:
    print(col)
    print(data[col].unique())

data['Age'] = data['Age'].fillna(value=30)
data['Embarked'] = data['Embarked'].fillna(value="Q")

categorical_features=[feature for feature in data.columns if ((data[feature].dtypes=='O') & (feature not in ['Survived']))]
print(categorical_features)

data['Sex']=data['Sex'].map({'male':1,'female':0})

print(data)
Embarked = pd.get_dummies(data['Embarked'], prefix='Embarked', prefix_sep='_',drop_first=False)
data = pd.concat([data, Embarked],axis=1)
data = data.drop(['Embarked'], axis=1)

X = data.drop("Survived", axis=1)
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

regressor = LogisticRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

pickle.dump(regressor, open('model.pkl', 'wb'))

