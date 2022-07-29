import numpy
import pandas
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.svm import SVC

df = pandas.read_csv("../data/TextFiles/smsspamcollection.tsv", sep="\t")

# Scope data:
df.head()
df.isnull().sum()
len(df)
df["label"].unique()
df["label"].value_counts()


# See text length distribution:
pyplot.xscale("log")
bins = 1.15**(numpy.arange(0,50))
pyplot.hist(df[df["label"] == "ham"]["length"], bins=bins, alpha=0.8)
pyplot.hist(df[df["label"] == "spam"]["length"], bins=bins, alpha=0.8)
pyplot.legend(("ham", "spam"))
pyplot.show()

# Create training and test data objects:
X = df[["length", "punct"]]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Logistic Regression model:
lr_model = LogisticRegression(solver="lbfgs")
lr_model.fit(X_train, y_train)
predictions = lr_model.predict(X_test)
df_metrics = pandas.DataFrame(metrics.confusion_matrix(y_test, predictions), index=["ham", "spam"], columns=["ham", "spam"])
df_metrics
print(metrics.classification_report(y_test, predictions))
print(metrics.accuracy_score(y_test, predictions))

# Train a Naives Bayes model:
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
predictions = nb_model.predict(X_test)
df_metrics = pandas.DataFrame(metrics.confusion_matrix(y_test, predictions), index=["ham", "spam"], columns=["ham", "spam"])
df_metrics
print(metrics.classification_report(y_test, predictions))
print(metrics.accuracy_score(y_test, predictions))

# Train a SVM model:
svc_model = SVC(gamma="auto")  
svc_model.fit(X_train, y_train)
predictions = svc_model.predict(X_test)
df_metrics = pandas.DataFrame(metrics.confusion_matrix(y_test, predictions), index=["ham", "spam"], columns=["ham", "spam"])
df_metrics
print(metrics.classification_report(y_test, predictions))
print(metrics.accuracy_score(y_test, predictions))

