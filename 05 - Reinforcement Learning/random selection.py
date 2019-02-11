# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:15:50 2018

@author: Ahmet
"""

#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("Ads_CTR_Optimisation.csv")

import random
N = 10000
d = 10
summ = 0
selections = []
for n in range(0,N):
    ad = random.randrange(d)
    selections.append(ad)
    reward = data.values[n, ad]
    summ = summ + reward
    
plt.hist(selections)
plt.show()