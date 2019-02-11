# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:31:16 2018

@author: Ahmet
"""

#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

data = pd.read_csv("Ads_CTR_Optimisation.csv")

N = 10000
d = 10
rewards = [0] * d
clicks = [0] * d
summ = 0
selections = []
for n in range(1,N):
    ad = 0
    max_ucb = 0
    for i in range(0,d):
        if (clicks[i] > 0):
            mean = rewards[i] / clicks[i]
            delta = math.sqrt(3/2 * math.log(n) / clicks[i])
            ucb = mean + delta
        else:
            ucb = N*10
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i
    selections.append(ad)
    clicks[ad] = clicks[ad] + 1
    reward = data.values[n,ad]
    rewards[ad] = rewards[ad] + reward
    summ = summ + reward
    
print(summ)
plt.hist(selections)
plt.show()