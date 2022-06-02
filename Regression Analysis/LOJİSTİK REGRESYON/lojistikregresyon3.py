# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:50:13 2022

@author: yakou
"""

# modeli oluşturmak için gereken kitaplıkları içe aktaralım.

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font",size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid",color_codes=True)
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
import math
from sklearn import metrics
from sklearn.metrics import confusion_matrix

df = pd.read_csv('banking.csv')

for col in df.select_dtypes(include="object").columns:
    print(col)
    print(df[col].unique())
    
features_na = [features for features in df.columns if df[features].isnull().sum() > 0]
for feature in features_na:
    print(feature, np.round(df[feature].isnull().mean(), 4),  ' % missing values')
else:
    print("No missing value found")
    
categorical_features=[feature for feature in df.columns if ((df[feature].dtypes=='O') & (feature not in ['y']))]
print(categorical_features)

for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(df[feature].unique())))

a = sns.heatmap(df.corr(),cmap = 'Blues', annot=True, annot_kws={'size':12},fmt='.2f')
print(a)

plt.figure(figsize=(15,80), facecolor='white')
plotnumber =1
for categorical_feature in categorical_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.countplot(y=categorical_feature,data=df)
    plt.xlabel(categorical_feature)
    plt.title(categorical_feature)
    plotnumber+=1
plt.show()

# katagorik verileri sayısallaştırmamız lazım.

cat_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

job = pd.get_dummies(df['job'], prefix='job', prefix_sep='_',drop_first=False)
df = pd.concat([df, job],axis=1)
marital = pd.get_dummies(df['marital'], prefix='marital', prefix_sep='_',drop_first=True)
df = pd.concat([df, marital],axis=1)
education = pd.get_dummies(df['education'], prefix='education', prefix_sep='_',drop_first=True)
df = pd.concat([df, education],axis=1)
default = pd.get_dummies(df['default'], prefix='default', prefix_sep='_',drop_first=True)
df = pd.concat([df, default],axis=1)
housing = pd.get_dummies(df['housing'], prefix='housing', prefix_sep='_',drop_first=True)
df = pd.concat([df, housing],axis=1)
loan = pd.get_dummies(df['loan'], prefix='housing', prefix_sep='_',drop_first=True)
df = pd.concat([df, loan],axis=1)
contact = pd.get_dummies(df['contact'], prefix='contact', prefix_sep='_',drop_first=True)
df = pd.concat([df, contact],axis=1)
month = pd.get_dummies(df['month'], prefix='month', prefix_sep='_',drop_first=True)
df = pd.concat([df, month],axis=1)
day_of_week = pd.get_dummies(df['day_of_week'], prefix='day_of_week', prefix_sep='_',drop_first=True)
df = pd.concat([df, day_of_week],axis=1)
poutcome = pd.get_dummies(df['poutcome'], prefix='poutcome', prefix_sep='_',drop_first=True)
df = pd.concat([df, poutcome],axis=1)

print(df.columns)
#sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap="viridis"); #missing valuelerimiz yok.
#missing value yok.




#plt.figure(figsize=(10,7))
#sns.distplot(df["age"].dropna(),kde=False, bins=20);
#üyelerin çoğu 25 ile 40 yaş arasındadır.

#iş türüne göre abonelik yüzdesi

x,y = 'job','y'
df1 = df.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1)
g.ax.set_ylim(0,100)
g.fig.set_size_inches(20,10)

for p in g.ax.patches:
    txt = str(p.get_height().round(1)) + '%'
    txt_x = p.get_x()
    txt_y = p.get_height()
    g.ax.text(txt_x,txt_y,txt)
    
marital=['married','single','divorced','unknown']
for i in marital:
    dfmarried = df[df['marital']==i]
    labels = "subscribed","unsubscribed"
    sizes = (dfmarried['y'].value_counts()).to_list()
    explode = (0,0.1)
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=True,startangle=90)
    ax1=('equal')
    plt.title(i)
    plt.show()  

x,y = 'marital','y'
df1 = df.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1)
g.ax.set_ylim(0,100)
g.fig.set_size_inches(20,10)

for p in g.ax.patches:
    txt = str(p.get_height().round(1)) + '%'
    txt_x = p.get_x()
    txt_y = p.get_height()
    g.ax.text(txt_x,txt_y,txt)
  


print(df['month'].value_counts()) # kaç tane ayımız var verimizde ona bakalım... tüm aylar olmayabilir.

# string değerleri int yapalım...
X = df.drop(['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome','y'], axis=1)
Y = df["y"]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=101)
logmodel = LogisticRegression(max_iter=76000)
logit_model = sm.Logit(Y_train,X_train)
result = logit_model.fit()
print(result.summary2())
logmodel.fit(X_train,Y_train)
y_pred = logmodel.predict(X_test)

print('Logistic Regressin model accuracy...: {:.2f}'.format(logmodel.score(X_test,Y_test)))




