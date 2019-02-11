#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

class Tester:
    def loadData(self):
        self.datas = pd.read_csv("veriler.csv")
        self.x = self.datas.iloc[:,1:4]
        self.y = self.datas.iloc[:,4:]

    def trainAndTest(self):
        from sklearn.cross_validation import train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.33, random_state=0)
    
    def scale(self):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.x_train)
        self.X_test = sc.transform(self.x_test)
    
    def lgr(self):
        from sklearn.linear_model import LogisticRegression
        lgr = LogisticRegression(random_state = 0)
        lgr.fit(self.X_train, self.y_train)
        self.y_pred = lgr.predict(self.X_test)
    
    def knn(self):
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
        knn.fit(self.X_train, self.y_train)
        self.y_pred = knn.predict(self.X_test)
    
    def svc(self):
        from sklearn.svm import SVC
        svc = SVC(kernel='rbf')
        svc.fit(self.X_train, self.y_train)
        self.y_pred = svc.predict(self.X_test)
        
    def nb(self):
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        gnb.fit(self.X_train, self.y_train)
        self.y_pred = gnb.predict(self.X_test)
        
    def dtc(self):
        from sklearn.tree import DecisionTreeClassifier
        dtc = DecisionTreeClassifier(criterion="entropy")
        dtc.fit(self.X_train, self.y_train)
        self.y_pred = dtc.predict(self.X_test)
        
    def rfc(self):
        from sklearn.ensemble import RandomForestClassifier
        rfc = RandomForestClassifier(n_estimators=5, criterion="entropy")
        rfc.fit(self.X_train, self.y_train)
        self.y_pred = rfc.predict(self.X_test)
    
    def confusionMatrix(self):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        print(cm)
        
tester = Tester()
tester.loadData()
tester.trainAndTest()
tester.scale()
tester.knn()
tester.confusionMatrix()
x = tester.x
y = tester.y