import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

# load data
datas = pd.read_csv("veriler.csv")
boyKiloYas = datas.iloc[:,1:4].values

# encode categoric to numeric
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
ulkeler = datas.iloc[:,0:1].values
le = LabelEncoder()
ulkeler[:,0] = le.fit_transform(ulkeler[:,0])
ohe = OneHotEncoder(categorical_features="all")
ulkeler = ohe.fit_transform(ulkeler).toarray()

c = datas.iloc[:,-1:].values
le = LabelEncoder()
c[:,0] = le.fit_transform(c[:,0])
ohe = OneHotEncoder(categorical_features="all")
c = ohe.fit_transform(c).toarray()

# reunion
nUlkeler = pd.DataFrame(data = ulkeler, index = range(22), columns = ['fr', 'tr', 'us'])
nBoyKiloYas = pd.DataFrame(data = boyKiloYas, index = range(22), columns = ['boy', 'kilo', 'yas'])
nCinsiyet = pd.DataFrame(data = c[:,:1], index = range(22), columns=['cinsiyet'])
ubky = pd.concat([nUlkeler, nBoyKiloYas], axis=1)
ubkyc = pd.concat([ubky, nCinsiyet], axis=1)

boy = ubkyc.iloc[:,3:4].values
sol = ubkyc.iloc[:,:3]
sag = ubkyc.iloc[:,4:]
ukyc = pd.concat([sol, sag], axis=1)

# train and test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(ukyc, boy, test_size=0.33, random_state=0)

# learn
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

# backward elimination
import statsmodels.formula.api as sm
#X = np.append(arr = np.ones((22,1)).astype(int), values = ukyc, axis=1)
X_list = ukyc.iloc[:,[0,1,2,3,4,5]].values
r = sm.OLS(endog = boy, exog = X_list).fit()
print(r.summary())