# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 20:50:41 2018

@author: Ahmet
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("musteriler.csv")
x = data.iloc[:,3:].values

print("K-Means")
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 4, init = 'k-means++', random_state = 0)
y = km.fit_predict(x)
plt.scatter(x[y == 0,0], x[y == 0,1], s = 25, c='red')
plt.scatter(x[y == 1,0], x[y == 1,1], s = 25, c='blue')
plt.scatter(x[y == 2,0], x[y == 2,1], s = 25, c='green')
plt.scatter(x[y == 3,0], x[y == 3,1], s = 25, c='yellow')
plt.show()

print("Agglomerative")
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 4, affinity = "euclidean", linkage = "ward")
y = ac.fit_predict(x)
plt.scatter(x[y == 0,0], x[y == 0,1], s = 25, c='red')
plt.scatter(x[y == 1,0], x[y == 1,1], s = 25, c='blue')
plt.scatter(x[y == 2,0], x[y == 2,1], s = 25, c='green')
plt.scatter(x[y == 3,0], x[y == 3,1], s = 25, c='yellow')
plt.show()

print("Birch")
from sklearn.cluster import Birch
brc = Birch(branching_factor=50, n_clusters=4, threshold=0.5, compute_labels=True)
y = brc.fit_predict(x)
plt.scatter(x[y == 0,0], x[y == 0,1], s = 25, c='red')
plt.scatter(x[y == 1,0], x[y == 1,1], s = 25, c='blue')
plt.scatter(x[y == 2,0], x[y == 2,1], s = 25, c='green')
plt.scatter(x[y == 3,0], x[y == 3,1], s = 25, c='yellow')
plt.show()