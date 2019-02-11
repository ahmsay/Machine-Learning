#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load data
datas = pd.read_csv("maaslar.csv")

x = datas.iloc[:,1:2]
y = datas.iloc[:,2:]

# scaling
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_scaled = sc1.fit_transform(x)
sc2 = StandardScaler()
y_scaled = sc2.fit_transform(y)

from sklearn.svm import SVR
svr = SVR(kernel = "rbf")
svr.fit(x_scaled, y_scaled)

plt.scatter(x_scaled, y_scaled, color="red")
plt.plot(x_scaled, svr.predict(x_scaled), color="blue")