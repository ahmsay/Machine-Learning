# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 18:24:35 2018

@author: Ahmet
"""
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("sepet.csv", header = None)
t = []
for i in range(0,7501):
    t.append([str(data.values[i,j]) for j in range(0,20)])

from apyori import apriori
apr = apriori(t, min_support = 0.01, min_confidence = 0.2, min_lift = 3, min_length = 2)
print(list(apr))