#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load data
datas = pd.read_csv("maaslar.csv")

x = datas.iloc[:,1:2]
y = datas.iloc[:,2:]

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state = 0)
dtr.fit(x,y)

print(dtr.predict(6.6))
print(dtr.predict(11))
z = x + 0.5
k = x - 0.4

plt.scatter(x, y, color="red")
plt.plot(x, dtr.predict(x), color="blue")
plt.plot(x, dtr.predict(z), color="green")
plt.plot(x, dtr.predict(k), color="yellow")