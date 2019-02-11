#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv("eksikveriler.csv")
print(datas)

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

age = datas.iloc[:,1:4].values
print(age)

imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])
print(age)