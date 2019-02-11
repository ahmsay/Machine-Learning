#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load data
datas = pd.read_csv("maaslar.csv")

x = datas.iloc[:,1:2]
y = datas.iloc[:,2:]

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=10, random_state=0) # n_estimators : how many decision trees will be created
rfr.fit(x,y)

print(rfr.predict(6.6))
print(rfr.predict(11))
z = x + 0.5
k = x - 0.5

plt.scatter(x, y, color="red")
plt.plot(x, rfr.predict(x), color="blue")
plt.plot(x, rfr.predict(z), color="green")
plt.plot(x, rfr.predict(k), color="yellow")

from sklearn.metrics import r2_score
print(r2_score(y, rfr.predict(x)))