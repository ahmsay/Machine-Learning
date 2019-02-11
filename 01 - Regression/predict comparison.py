import pandas as pd

# load data
datas = pd.read_csv("maaslar_yeni.csv")
#print(datas.corr())
x = datas.iloc[:,2:5]
y = datas.iloc[:,5:]

# scaling
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_scaled = sc1.fit_transform(x)
sc2 = StandardScaler()
y_scaled = sc2.fit_transform(y)

# training
# multilinear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x, y)
MLRpred = lr.predict(x)

# polynomial regression
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree = 6)
x_poly = pf.fit_transform(x)
lr2 = LinearRegression()
lr2.fit(x_poly, y)
PRpred = lr2.predict(x_poly)

# suppor vector regression
from sklearn.svm import SVR
svr = SVR(kernel = "rbf")
svr.fit(x_scaled, y_scaled)
SVRpred = svr.predict(x_scaled)

# decision tree
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state = 0)
dtr.fit(x,y)
DTRpred = dtr.predict(x)

# random forest regression
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=10, random_state=0)
rfr.fit(x,y)
RFRpred = rfr.predict(x)

# analysis
from sklearn.metrics import r2_score
print("R2 Scores")
print("Multi Linear Regression: ", r2_score(y, MLRpred))
print("Polynomial Regression: ", r2_score(y, PRpred))
print("Support Vector Regression: ", r2_score(y_scaled, SVRpred))
print("Decision Tree Regression: ", r2_score(y, DTRpred))
print("Random Forest Regression: ", r2_score(y, RFRpred))

"""
import statsmodels.api as sm
model = sm.OLS(PRpred, x)
print(model.fit().summary())
"""