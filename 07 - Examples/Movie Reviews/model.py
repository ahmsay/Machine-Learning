import numpy as np
import re
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
nltk.download('stopwords')
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense

results_p = []

positive = open('positive.txt', 'r', encoding="utf8")
for review in positive:
    review = re.sub('[^a-zA-ZçğıöşüÇĞİÖŞÜ]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('turkish'))]
    review = ' '.join(review)
    results_p.append(review)

results_n = []

negative = open('negative.txt', 'r', encoding="utf8")
for review in negative:
    review = re.sub('[^a-zA-ZçğıöşüÇĞİÖŞÜ]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('turkish'))]
    review = ' '.join(review)
    results_n.append(review)

results = results_p + results_n

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(results).toarray()

y_p = np.ones((730,))
y_n = np.zeros((730,))
y = np.append(y_p, y_n, axis=0)

from sklearn.feature_selection import SelectKBest, chi2
skb = SelectKBest(chi2, k=5000)
X_new = skb.fit_transform(x, y)

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

nn = Sequential()
nn.add(Dense(2500, kernel_initializer = "uniform", activation = "relu", input_dim = 5000))
nn.add(Dense(2500, kernel_initializer = "uniform", activation = "relu"))
nn.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
nn.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])
nn.fit(X_train, y_train, epochs = 30)
y_pred = nn.predict(X_test)
y_pred = y_pred > 0.5

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy:", (cm[0][0] + cm[1][1]) / sum(map(sum, cm)))

"""
new_reviews = ['berbat ötesi bir filmdi.']
new_results = []
for review in new_reviews:
    review = re.sub('[^a-zA-ZçğıöşüÇĞİÖŞÜ]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('turkish'))]
    review = ' '.join(review)
    new_results.append(review)
x_2 = cv.transform(new_results).toarray()
x_2 = skb.transform(x_2)
x_2 = sc.transform(x_2)
y_pred2 = nn.predict(x_2)
y_pred2 = y_pred2 > 0.5
"""