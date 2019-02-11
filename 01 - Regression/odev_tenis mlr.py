import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

# load data
datas = pd.read_csv("odev_tenis.csv")

# encode categoric to numeric
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
ohe = OneHotEncoder(categorical_features="all")

weather = datas.iloc[:,0:1].values
weather[:,0] = le.fit_transform(weather[:,0])
weather = ohe.fit_transform(weather).toarray()

tempHum = datas.iloc[:,[1,2]].values

datas2 = datas.apply(LabelEncoder().fit_transform)

# reunion
nWeather = pd.DataFrame(data = weather, index = range(14), columns = ['overcast', 'rainy', 'sunny'])
nTempHum = pd.DataFrame(data = tempHum, index = range(14), columns = ['temperature', 'humidity'])
nPlay = datas2.iloc[:,4:]
wth = pd.concat([nWeather, nTempHum], axis = 1)
wthwp = pd.concat([wth, nPlay], axis = 1)

# train and test
h = wthwp.iloc[:,4:5]
wtwp = wthwp.drop('humidity', 1)
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(wtwp, h, test_size=0.33, random_state=0)

# learn
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

# backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((14,1)).astype(int), values = wtwp, axis=1)
X_list = wtwp.iloc[:,[0,1,2,3,4]].values
r = sm.OLS(endog = h, exog = X_list).fit()
print(r.summary())