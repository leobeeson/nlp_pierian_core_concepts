from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas


df = pandas.read_csv("../data/TextFiles/moviereviews.tsv", sep="\t")
df.head()

df.dropna(inplace=True)

blanks = []
for idx, label, review in df.itertuples():
    if isinstance(review, str):
        if review.isspace():
            blanks.append(idx)

df.drop(blanks,inplace=True)
df["label"].value_counts()


sid = SentimentIntensityAnalyzer()

df["scores"] = df["review"].apply(lambda review: sid.polarity_scores(review))

df["compound"] = df["scores"].apply(lambda score: score["compound"])
df.head()

df["compound_label"] = df["compound"].apply(lambda compound: "pos" if compound >= 0 else "neg")
df.head()

accuracy_score(df["label"], df["compound_label"])
print(classification_report(df["label"], df["compound_label"]))
print(confusion_matrix(df["label"], df["compound_label"]))
