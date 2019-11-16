import pandas as pd

class Tester:
    def loadData(self):
        self.heroes = pd.read_csv("heroes.csv")
        self.criterias = self.heroes[["HERO", "STR25", "INT25", "AGI25", "CLASS"]]
        
    def trainAndTest(self):
        from sklearn.cross_validation import train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.criterias[["HERO", "STR25", "INT25", "AGI25"]], self.criterias[["CLASS"]], test_size=0.33) # we don't need the hero name for train and test, we'll use it later for the analysis
        
    def scale(self):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.x_train.iloc[:,1:]) # hero name is not included in scaling (obviously)
        self.X_test = sc.transform(self.x_test.iloc[:,1:])
        
    def nb(self):
        from sklearn.naive_bayes import GaussianNB # gaussian naive bayes is one of the best algorithms for this dataset (the other one is SVC (kernel ='poly', degree = 1))
        gnb = GaussianNB()
        gnb.fit(self.X_train, self.y_train)
        self.y_pred = gnb.predict(self.X_test)
        
    def analyze(self):
        print("--- Results ---")
        print("\nConfusion matrix:")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.y_test, self.y_pred) # compare actual results and predicted results with confussion matrix
        print(cm)
        self.analysis = pd.concat([self.x_test.iloc[:, 0:1], self.y_test], axis = 1).reset_index(drop=True) # concatenate the hero name and the actual results
        self.predDF = pd.DataFrame(data = self.y_pred, index = range(self.y_pred.size), columns = ['Prediction'])
        self.analysis = pd.concat([self.analysis, self.predDF], axis = 1) # add the predicted results to the table
        print("\nIncorrect prediction(s):")
        for index, row in self.analysis.iterrows():
            if row["CLASS"] != row["Prediction"]:
                print(row["HERO"], "=>", "Actual:", row["CLASS"], ", Predicted:", row["Prediction"])
        print("\nNumber of incorrect predictions out of", self.y_pred.size, "heroes:", (self.predDF.values != self.y_test.values).sum())

tester = Tester()
tester.loadData()
tester.trainAndTest()
tester.scale()
tester.nb()
tester.analyze()