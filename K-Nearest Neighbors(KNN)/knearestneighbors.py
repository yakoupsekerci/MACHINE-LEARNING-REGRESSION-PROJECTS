# -*- coding: utf-8 -*-
"""
Created on Wed May 11 18:22:32 2022

@author: yakou
"""

import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


df = pd.read_csv('iris.csv')

df['variety']=df['variety'].map({'Setosa':0,'Versicolor':1, 'Virginica':2})

X = df.drop("variety", axis=1)
y = df["variety"]

sns.pairplot(df,hue='variety')
plt.show()

sns.lmplot(x='petal.width',y='petal.length',data=df,hue='variety',fit_reg=False)
plt.xlabel('TAÇ YAPRAK GENİŞLİĞİ (CM)')
plt.ylabel('TAÇ YAPRAK UZUNLUGU (CM)')
plt.show() # sınıflandırma yapılabilir bir veri seti.

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)



k_range = range(1,26)
scores = {}
scores_list = []
for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train,y_train)
        y_pred=knn.predict(x_test)
        scores[k] = metrics.accuracy_score(y_test,y_pred)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))
print(scores)# en yakın knn değeri seçilir.

plt.plot(k_range,scores_list) # bide grafik olarak görelim.
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(y_test, y_pred))


#0 = setosa, 1=versicolor, 2=virginica
# let's make our prediction

new_data = pd.DataFrame({'sepal.length':[5.1,4.9,4.7,4.6],'sepal.width':[3.5,3,3.2,3.1],'petal.length':[1.4,1.4,1.3,1.5],'petal.width':[.2,.2,.2,.2]})
print(new_data)
knn.predict(new_data).round(1)
new_data['Prediction Variety'] = knn.predict(new_data)
print(new_data)

if y_pred.shape == y_test.shape:
    print('eşit')
    
sns.barplot(y_pred)
