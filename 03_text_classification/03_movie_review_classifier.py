import numpy
import pandas

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

df = pandas.read_csv("../data/TextFiles/moviereviews.tsv", sep="\t")
df.head()
len(df)

# Remove missing reviews:
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
len(df)

# Remove blank reviews:
blanks_idx = []
for idx, label, review in df.itertuples():
    if review.isspace():
        blanks_idx.append(idx)
df.drop(blanks_idx, inplace=True)
len(df)


# Create train and test sets:
X = df["review"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Create classifier pipeline:
movie_classifier = Pipeline([
    ("tfidf", TfidfVectorizer()), 
    ("linear_svc", LinearSVC())
    ])
movie_classifier.fit(X_train, y_train)
predictions = movie_classifier.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
accuracy_score(y_test, predictions)

