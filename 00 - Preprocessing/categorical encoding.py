#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv("eksikveriler.csv")
#print(datas)

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

age = datas.iloc[:,1:4].values
#print(age)

imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])
#print(age)

country = datas.iloc[:,0:1].values
print(country)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
country[:,0] = le.fit_transform(country[:,0])
print(country)

ohe = OneHotEncoder(categorical_features="all")
country = ohe.fit_transform(country).toarray()
print(country)

# labelencoder = verilen değeri bire bir sayıya çevirir
# onehotencoder = verilen değeri kolon bazlı sayıya çevirir (binary)