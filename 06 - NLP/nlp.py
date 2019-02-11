import pandas as pd

reviews = pd.read_csv("Restaurant_Reviews.csv")

import re
import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

results = []
for i in range(reviews.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', reviews['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    results.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
x = cv.fit_transform(results).toarray()
y = reviews.iloc[:,1].values

from sklearn.feature_selection import SelectKBest, chi2
X_new = SelectKBest(chi2, k=1000).fit_transform(x, y)

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state = 0)

from sklearn.naive_bayes import GaussianNB      
gnb = GaussianNB()
gnb.fit(x_train, y_train)  
y_pred = gnb.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)