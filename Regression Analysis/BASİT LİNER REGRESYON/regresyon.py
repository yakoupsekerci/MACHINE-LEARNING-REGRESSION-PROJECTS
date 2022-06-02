# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 03:10:59 2022

@author: yakou
"""
 

# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('veriseti1.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

hatalar = y_test - y_pred
hatalarinkaresi = np.square(hatalar)
r_2 = 1 - ((hatalarinkaresi).sum()/((y_test - y_test.mean())**2).sum()).sum()
print("Manuel R^2",r_2)

print("R^2:",r2_score(y_test,y_pred)) 
print("beta_0 = {}, beta_1 = {}".format(regressor.intercept_,regressor.coef_))

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red' ,label='gerçek değerler' ,marker='o')
plt.plot(X_train, regressor.predict(X_train), color = 'blue', label='regresyon' )
plt.legend()
plt.title('GEÇİRİLEN SÜRE VE SATIŞ (Test Verisi)')
plt.xlabel('GEÇİRİLEN SÜRE(DK)')
plt.ylabel('SATIŞ')
plt.show()

