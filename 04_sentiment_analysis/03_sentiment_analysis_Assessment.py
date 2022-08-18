from nltk.sentiment.vader import SentimentIntensityAnalyzer


sid = SentimentIntensityAnalyzer()
review = "Doctor Strange Multiverse of Madness is overall an enjoyable movie. Benedict Cumberbatch and Elizabeth Olsen do a very good job, and it was nice to see some of the other MCU characters woven into the story. It is very fast paced, dynamic, and with interesting twists. However, it was permeated with Disney's woke agenda, with unnecessary elements of racism, sexual orientations, and satanization of motherhood."
sid.polarity_scores(review)


def review_rating(review: str) -> dict:
    vader_score = sid.polarity_scores(review)
    _ = vader_score.pop("compound")
    max_sentiment = max(vader_score, key=lambda key: vader_score[key])
    if max_sentiment == "neg":
        return "Negative"
    elif max_sentiment == "neu":
        return "Neutral"
    elif max_sentiment == "pos":
        return "Positive"

review_rating(review)