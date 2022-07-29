import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

df =pandas.read_csv("../data/TextFiles/smsspamcollection.tsv", sep="\t")
df.head()
df.isnull().sum()
df["label"].value_counts()


X = df["message"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Plain frequency count vectoriser:
count_vect = CountVectorizer()
# Fit and transform separately:
count_vect.fit(X_train)
X_train_counts = count_vect.transform(X_train)
# Fit and transform in one step:
X_train_counts = count_vect.fit_transform(X_train)


# Convert freq vectoriser to a TF-IDF vectoriser:
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# Freq count vectoriser and TF-IDF transformer all in one:
tfidf_vectoriser = TfidfVectorizer()
X_train_tfidf = tfidf_vectoriser.fit_transform(X_train)


# Pass the TF-IDF transformed data to a linear classifier:
svc = LinearSVC()
svc.fit(X_train_tfidf, y_train)


# Serialise all steps into a single pipeline:
text_classifier = Pipeline([("tfidf", TfidfVectorizer()), ("svc", LinearSVC())])
text_classifier.fit(X_train, y_train)
predictions = text_classifier.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
accuracy_score(y_test, predictions)


# Test classifier manually:
text_classifier.predict(["Hi. How are you doing today?"])
text_classifier.predict(["Congratulations! You've been selected as a winner. TEXT WON to 32453."])
text_classifier.predict(["Hey, are you home? One of your packages was delivered to my house."])
