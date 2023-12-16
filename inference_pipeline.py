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
    news_fg = fs.get_feature_group(name="news_articles", version=4)
    news_df = news_fg.read()

    # Only keep news articles from today
    news_df = news_df[news_df['pubdate'] == today]
    print(f"Len of df: {len(news_df)}")

    # Add sentiments for articles
    sentiment_pipeline = pipeline("sentiment-analysis")
    news_df['sentiment'] = news_df.apply(get_sentiment_value, sentiment_pipeline=sentiment_pipeline, axis=1)

    # Calculate today's average sentiment
    avg_sentiment = news_df['sentiment'].mean()

    # Find today's most positive article
    most_positive = news_df.loc[news_df['sentiment'].idxmax()]
    most_positive = pd.DataFrame(most_positive).T
    most_positive['sentiment'] = most_positive['sentiment'].astype('float')



    # Put most positive article and average sentiment of today in feature group
    articles_monitoring_fg = fs.get_or_create_feature_group(
        name="articles_most_positive",
        version=1,
        primary_key=['pubdate'],
        description="Today's most positive article and average rating"
    )

    print(f'sentiment: {most_positive["sentiment"]}')
    most_positive['avg_sentiment'] = news_df['sentiment'].mean()

    articles_monitoring_fg.insert(most_positive, write_options={"wait_for_job": False})

    # Put predictions for each of today's articles in feature group
    articles_predictions_fg = fs.get_or_create_feature_group(
        name="articles_predictions",
        version=1,
        primary_key=['article_id'],
        description="Sentiment ratings of articles"
    )
    prediction_df = news_df.filter(['article_id', 'pubdate', 'sentiment'], axis=1)
    articles_predictions_fg.insert(prediction_df, write_options={"wait_for_job": False})

if __name__ == "__main__":
    main()
