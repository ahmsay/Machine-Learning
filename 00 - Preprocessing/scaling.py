# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 17:05:36 2018

@author: Ahmet
"""

#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

# load data
datas = pd.read_csv("eksikveriler.csv")

# fill missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
age = datas.iloc[:,1:4].values
imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])

# encode categoric to numeric
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
country = datas.iloc[:,0:1].values
le = LabelEncoder()
country[:,0] = le.fit_transform(country[:,0])
ohe = OneHotEncoder(categorical_features="all")
country = ohe.fit_transform(country).toarray()

# reunion
result = pd.DataFrame(data = country, index = range(22), columns = ['fr', 'tr', 'us'])
result2 = pd.DataFrame(data = age, index = range(22), columns = ['boy', 'kilo', 'yas'])
gender = datas.iloc[:,-1:].values
result3 = pd.DataFrame(data = gender, index = range(22), columns=['cinsiyet'])
reunion = pd.concat([result, result2], axis=1)
#reunion2 = pd.concat([reunion, result3], axis=1)

# train and test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(reunion, result3, test_size=0.33, random_state=0)

# scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)