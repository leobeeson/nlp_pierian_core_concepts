from curses.ascii import isspace
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


sid = SentimentIntensityAnalyzer()


df = pandas.read_csv("../data/TextFiles/amazonreviews.tsv", sep="\t")
df.head()
df["label"].value_counts()

# Remove NA:
df.dropna(inplace=True)

# Remove empty or blank reviews:
blanks = []
for idx, label, review in df.itertuples():
    if isinstance(review, str):
        if review.isspace():
            blanks.append(idx)

sid.polarity_scores(df.iloc[0]["review"])


df["scores"] = df["review"].apply(lambda review: sid.polarity_scores(review))
df.head()

df["compound"] = df["scores"].apply(lambda score: score["compound"])
df.head()

df["compound_label"] = df["compound"].apply(lambda compound: "pos" if compound >= 0 else "neg")
df.head()

accuracy_score(df["label"], df["compound_label"])
print(classification_report(df["label"], df["compound_label"]))
 


