from random import sample
import nltk
# nltk.download("vader_lexicon") # Only needs to be done once.
from nltk.sentiment.vader import SentimentIntensityAnalyzer


sid = SentimentIntensityAnalyzer()

sample_review = "This is a good movie"
sid.polarity_scores(sample_review)

sample_review = "This was the best, most awesome movie EVER MADE!!!"
sid.polarity_scores(sample_review)

sample_review = "This was the WORST movie that has ever disgraced the screen."
sid.polarity_scores(sample_review)
