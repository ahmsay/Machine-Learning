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
"""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
"""
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=300, criterion="entropy")
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

correct = 0
for i in range(cm[0].size):
    correct += cm[i][i]

print(correct, "/", x_test.shape[0])