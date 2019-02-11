# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 15:14:09 2018

@author: Ahmet
"""

#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

data = pd.read_csv("Ads_CTR_Optimisation.csv")

N = 10000
d = 10
summ = 0
selections = []
ones = [0] * d
zeros = [0] * d
for n in range(1,N):
    ad = 0
    max_th = 0
    for i in range(0,d):
        rasbeta = random.betavariate(ones[i] + 1, zeros[i] + 1)
        if rasbeta > max_th:
            max_th = rasbeta
            ad = i
    selections.append(ad)
    reward = data.values[n,ad]
    if reward == 1:
        ones[ad] = ones[ad] + 1
    else:
        zeros[ad] = zeros[ad] + 1
    summ = summ + reward
    
print(summ)
plt.hist(selections)
plt.show()