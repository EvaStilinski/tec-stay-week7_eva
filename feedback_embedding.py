import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('vader_lexicon')

# Sample customer feedback
feedbacks = [
    "The product is great, I love it!",
    "The customer service was terrible.",
    "Shipping was fast and efficient.",
    "I had a bad experience with the product quality.",
    "The product arrived broken and unusable.",
    "The product was not as described.",
    "I don't recommend buying this product.",
    "I like the colors and the design."
]

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Analyze sentiments for each word
for i, feedback in enumerate(feedbacks):
    words = word_tokenize(feedback)  # Tokenize words for the current feedback

    sentiment_value = 1
    for word in words:
            sentiment_scores = sid.polarity_scores(word)
            print(sentiment_scores['compound'])
            if sentiment_scores['compound'] > 0:
                sentiment_value = sentiment_value * 2
            elif sentiment_scores['compound'] < 0:
                sentiment_value = sentiment_value * -1
    sentiment = "positive" if sentiment_value > 1 else "negative" if sentiment_value < 0 else "neutral"
    print(f"Sentence: {feedback}, Sentiment: {sentiment}")

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")