# -*- coding: utf-8 -*-
"""
Created on Wed May  4 01:12:17 2022

@author: yakou
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns

train = pd.read_csv("titanic_train.csv")

print(train.info()) # kayıp değerlerimiz var yani boş değerlerimiz var.

# sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="viridis"); # kayıp verileri bu tabloda daha net görebiliriz.

#sns.countplot(x="Survived",hue="Sex",data=train, palette="RdBu_r");
# cinsiyeti erkek olanların hayatta kalması kadınlardan daha az.
# cinsiyeti erkek olanların ölüm olaranı kadınlardan daha fazla.

#sns.countplot(x="Survived", hue="Pclass", data=train);
# yolcu sınıfı 3. olanlar daha çok ölmüş

# plt.figure(figsize=(10,7))
# sns.distplot(train["Age"].dropna(),kde=False, bins=30);

#sns.countplot("SibSp",data=train)

#train["Fare"].hist(bins=40,figsize=(10,4));

plt.figure(figsize=(10,7))
sns.boxplot(x="Pclass",y="Age",data=train); #ortalamayla doldurmak için bakmamaız lazım.

def impute_age(col):
    Age = col[0]
    Pclass = col[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
    
train["Age"] = train[["Age","Pclass"]].apply(impute_age,axis=1)

# sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="viridis");

train.drop("Cabin",axis=1,inplace=True)
train.dropna(inplace=True)
#sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="viridis");

sex = pd.get_dummies(train["Sex"],drop_first=True)
embark = pd.get_dummies(train["Embarked"],drop_first=True)
train = pd.concat([train,sex,embark],axis=1)

train.drop(['Name','Sex','Embarked','Ticket','PassengerId'] ,axis=1,inplace=True)

print(train)

X = train.drop("Survived", axis=1)
Y = train["Survived"]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)
y_pred = logmodel.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test,y_pred))

b = logmodel.predict([[0,3,22,1,7.35,1,0,1]])
print(b)

print(X.columns)