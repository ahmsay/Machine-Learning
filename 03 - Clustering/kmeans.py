# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 00:54:49 2018

@author: Ahmet
"""

#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv('musteriler.csv')
X = datas.iloc[:,3:].values

from sklearn.cluster import KMeans
km = KMeans(n_clusters = 3, init = 'k-means++', random_state = 123)
Y = km.fit_predict(X)
#print(km.cluster_centers_)
print(Y)
plt.scatter(X[Y == 0,0], X[Y == 0,1], s = 100, c='red')
plt.scatter(X[Y == 1,0], X[Y == 1,1], s = 100, c='blue')
plt.scatter(X[Y == 2,0], X[Y == 2,1], s = 100, c='green')
plt.show()


results = []
for i in range(1,11):
    km = KMeans(n_clusters = i, init = 'k-means++', random_state = 123)
    km.fit(X)
    results.append(km.inertia_)
    
plt.plot(range(1,11), results)
