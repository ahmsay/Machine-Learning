import pandas as pd

data = pd.read_csv("forestData.csv")

x = data.iloc[:,1:55]
y = data.iloc[:,55:]

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion="entropy")

from sklearn.model_selection import cross_val_score
success = cross_val_score(estimator = rfc, X = X_train, y = y_train.values.ravel(), cv = 4)
print(success.mean())
print(success.std())