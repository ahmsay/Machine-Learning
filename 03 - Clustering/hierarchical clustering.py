# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 13:50:55 2018

@author: Ahmet
"""

#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv('musteriler.csv')
X = datas.iloc[:,3:].values

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 3, affinity = "euclidean", linkage = "ward")
Y = ac.fit_predict(X)
print(Y)

plt.scatter(X[Y == 0,0], X[Y == 0,1], s = 100, c='red')
plt.scatter(X[Y == 1,0], X[Y == 1,1], s = 100, c='blue')
plt.scatter(X[Y == 2,0], X[Y == 2,1], s = 100, c='green')
plt.show()

import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = "ward"))
plt.show()