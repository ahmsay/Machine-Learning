#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load data
datas = pd.read_csv("maaslar.csv")

x = datas.iloc[:,1:2]
y = datas.iloc[:,2:]

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree = 4)
x_poly = pf.fit_transform(x)

from sklearn.linear_model import LinearRegression
lr2 = LinearRegression()
lr2.fit(x_poly, y)
pred = lr2.predict(x_poly)

plt.scatter(x,y, color="red")
plt.plot(x, pred, color="blue")

print(lr2.predict(pf.fit_transform(11)))