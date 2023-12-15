import hopsworks
from datetime import datetime
import pandas as pd
from transformers import pipeline

def format_sentiment(sentiment):
    if sentiment['label'] == 'NEGATIVE':
        return sentiment['score'] * -1
    else:
        return sentiment['score']

def get_sentiment_value(news_object, sentiment_pipeline):
    return format_sentiment(sentiment_pipeline(news_object['title'])[0])

def main():
    # Get today's date
    today = datetime.now().date()

    # Get news articles
    project = hopsworks.login()
    fs = project.get_feature_store()
    news_fg = fs.get_feature_group(name="news_articles", version=1)
    news_df = news_fg.read()

    # Only keep news articles from today
    news_df = news_df[pd.to_datetime(news_df['pubdate']).dt.date == today]

    # Add sentiments for articles
    sentiment_pipeline = pipeline("sentiment-analysis")
    news_df['sentiment'] = news_df.apply(get_sentiment_value, sentiment_pipeline=sentiment_pipeline, axis=1)

    # Calculate today's average sentiment
    avg_sentiment = news_df['sentiment'].mean()

    # Find today's most positive article
    most_positive = news_df.loc[news_df['sentiment'].idxmax()]

if __name__ == "__main__":
    main()
