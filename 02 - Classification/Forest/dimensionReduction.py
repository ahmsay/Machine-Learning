import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

class Tester:
    def initialize(self):
        # data import
        data = pd.read_csv("forestData.csv")
        x = data.iloc[:,1:55].values
        y = data.iloc[:,55:].values

        # train test split
        from sklearn.cross_validation import train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.33, random_state=0)

        # scale
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.x_train)
        self.X_test = sc.transform(self.x_test)

    def randomForest(self, xtrain, xtest):
        rfc = RandomForestClassifier(n_estimators=300, criterion="entropy")
        rfc.fit(xtrain, self.y_train.ravel())
        y_pred = rfc.predict(xtest)
        self.analyze(y_pred)

    def analyze(self, pred):
        cm = confusion_matrix(self.y_test, pred)
        print(cm)
        correct = 0
        for i in range(cm[0].size):
            correct += cm[i][i]
        print(correct, "/", self.x_test.shape[0])
        
    def pca(self, nComponents):
        from sklearn.decomposition import PCA
        pca = PCA(n_components = nComponents)
        self.X_train_pca = pca.fit_transform(self.X_train)
        self.X_test_pca = pca.transform(self.X_test)
        
    def lda(self, nComponents):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(n_components = nComponents)
        self.X_train_lda = lda.fit_transform(self.X_train, self.y_train.ravel())
        self.X_test_lda = lda.transform(self.X_test)
        
tester = Tester()
tester.initialize()
print("Without dimension reduction:")
tester.randomForest(tester.X_train, tester.X_test)
print("With PCA:")
tester.pca(27)
tester.randomForest(tester.X_train_pca, tester.X_test_pca)
print("With LDA:")
tester.lda(27)
tester.randomForest(tester.X_train_lda, tester.X_test_lda)