# -*- coding: utf-8 -*-
"""
Created on Sun May  8 01:34:41 2022

@author: yakou
"""

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
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('load.csv')
print(df.info())

for col in df.select_dtypes(include="object").columns:
    print(col)
    print(df[col].unique()) 

features_na = [features for features in df.columns if df[features].isnull().sum() > 0 ]
for feature in features_na:
    print(feature,np.round(df[feature].isnull().sum()),' tane missing value var')
else:
    print('NO MİSSİNG VALUE FOUND.')

x,y = 'Property_Area','Credit_History'
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
"""
Kayıp verilerle baş etmenin en iyi yolu hiç kayıp veriye sahip olmamaktır. 
"""
def impute_credit(col):
    credit_history = col[0]
    Property_Area = col[1]
    
    if pd.isnull(credit_history):
        if Property_Area == 'Urban':
            return 1
        elif Property_Area == 'Rural':
            return 1
        elif Property_Area == 'Semiurban':
            return 1
        else:
            return 1
    else:
        return credit_history
    
df["Credit_History"] = df[["Credit_History","Property_Area"]].apply(impute_credit,axis=1)

print(df.info())

x,y = 'Property_Area','Loan_Amount_Term'
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

def impute_amount(col):
    Loan_Amount_Term = col[0]
    Property_Area = col[1]
    
    if pd.isnull(Loan_Amount_Term):
        if Property_Area == 'Urban':
            return 360
        elif Property_Area == 'Rural':
            return 360
        elif Property_Area == 'Semiurban':
            return 360
        else:
            return 360
    else:
        return Loan_Amount_Term

df["Loan_Amount_Term"] = df[["Loan_Amount_Term","Property_Area"]].apply(impute_credit,axis=1)

categorical_features=[feature for feature in df.columns if ((df[feature].dtypes=='O') & (feature not in ['Loan_ID']))]
print(categorical_features)

plt.figure(figsize=(15,80), facecolor='white')
plotnumber =1
for categorical_feature in categorical_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.countplot(x=categorical_feature,data=df)
    plt.xlabel(categorical_feature)
    plt.title(categorical_feature)
    plotnumber+=1
plt.show()




"""
BİR DEĞİŞKEN İÇİN ÇİZDİRMEK.
"""

sns.countplot(x=df['Gender'],data=df)

missingcat = ['Gender','Married','Dependents','Education','Self_Employed']
counter = 0
a = []
siraliSozluk = []
for i in missingcat:
    columns = df[i]
    a = sorted(dict(columns.value_counts()).items(),key=lambda x: x[1], reverse=True)
#    print(a[0])
    siraliSozluk += a[0]
print(siraliSozluk)
encokdeger = list(siraliSozluk[::2])
print(encokdeger)    

def impute_amount(col):
    Loan_Amount_Term = col[0]
    Property_Area = col[1]
    
    if pd.isnull(Loan_Amount_Term):
        if Property_Area == 'Urban':
            return 360
        elif Property_Area == 'Rural':
            return 360
        elif Property_Area == 'Semiurban':
            return 360
        else:
            return 360
    else:
        return Loan_Amount_Term


missingcat = ['Gender','Married','Dependents','Education','Self_Employed']
tofiil = ['Male', 'Yes', '0', 'Graduate', 'No']

df['Gender'] = df['Gender'].fillna('Male')
df['Married'] = df['Married'].fillna('Yes')
df['Dependents'] = df['Dependents'].fillna('0')
df['Education'] = df['Education'].fillna('Graduate') 
df['Self_Employed'] = df['Self_Employed'].fillna('No')


df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())

print(df.isnull().sum())

df['Gender']=df['Gender'].map({'Male':1,'Female':0})
df['Married']=df['Married'].map({'Yes':1,'No':0})
df['Education']=df['Education'].map({'Graduate':1,'Not Graduate':0})
df['Self_Employed']=df['Self_Employed'].map({'Yes':1,'No':0})

lbl_encode = LabelEncoder()
add_columns = pd.get_dummies(df['Property_Area'],prefix='Property_Area')
df['Property_Area'] = lbl_encode.fit_transform(df['Property_Area'])
df['Property_Area_label'] = lbl_encode.fit_transform(df['Property_Area'])
df.drop(['Property_Area_label','Property_Area'], axis=1,inplace=True)
df = df.join(add_columns)
print(df.columns)



lbl_encode = LabelEncoder()
add_columns = pd.get_dummies(df['Dependents'],prefix='Dependents')
df['Dependents'] = lbl_encode.fit_transform(df['Dependents'])
df['Dependents_label'] = lbl_encode.fit_transform(df['Dependents'])
df.drop(['Dependents_label','Dependents'], axis=1,inplace=True)
df = df.join(add_columns)
print(df.columns)

df['Loan_Status']=df['Loan_Status'].map({'Y':1,'N':0})

categorical_features=[feature for feature in df.columns if ((df[feature].dtypes=='O') & (feature not in ['Loan_ID'] ))]
print(categorical_features)
df.drop('Loan_ID',axis=1,inplace=True)
print(df.info())

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

model = LogisticRegression()
model.fit(X_train,y_train)

lr_prediction = model.predict(X_test)
print('Logistic Regression accuracy = ', metrics.accuracy_score(lr_prediction,y_test))

logit_model = sm.Logit(y_train,X_train)
result = logit_model.fit()
print(result.summary())

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test,model.predict(X_test))
fpr, tpr, threshold = roc_curve(y_test,model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label='LOGİSTİC REGRESSİON (AREA = %0.2f)' % logit_roc_auc )
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE POSİTİVE RATE')
plt.ylabel('TRUE POSİTİVE RATE')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='LOWER RİGHT')
plt.savefig('log_ROC')
plt.show()
