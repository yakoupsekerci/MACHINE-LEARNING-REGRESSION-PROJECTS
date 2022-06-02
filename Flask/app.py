# -*- coding: utf-8 -*-
"""
Created on Thu May 19 19:28:18 2022

@author: yakou
"""

from flask import Flask
from flask import render_template
from flask import request
import pickle
import pandas as pd

app = Flask(__name__)

def setEmbarkedValues(value):
	if value == 'Embarked_Q':
		return 0, 1, 0
	elif value == 'Embarked_S':
		return 0, 0, 1
	else:
		return 1, 0, 0

def setSexValues(value):
	if value == 'male':
		return 1
	else:
		return 0

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Embarked_Q,Embarked_S,Embarked_C = setEmbarkedValues(request.form.get('Embarked'))
    Sex = setSexValues(request.form.get('Sex'))
    Pclass = request.form.get('Pclass')
    Age = request.form.get('Age')
    SibSp = request.form.get('SibSp')
    Parch = request.form.get('Parch')
    Fare = request.form.get('Fare')
    
    resultValues = pd.DataFrame({'Pclass':[Pclass],'Sex':[Sex],'Age':[Age],'SibSp':[SibSp],'Parch':[Parch],'Fare':[Fare],'Embarked_Q':[Embarked_Q],'Embarked_C':[Embarked_C],'Embarked_S':[Embarked_S]})
    model = pickle.load(open('model.pkl','rb'))
    predictionValues = model.predict(resultValues)
    return render_template('index.html',prediction_text = predictionValues)    

if __name__ == '__main__':
	app.run(debug=True)



