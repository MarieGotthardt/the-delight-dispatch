import hopsworks
from datetime import datetime
import pandas as pd
from transformers import pipeline
from openai import OpenAI
import requests
import os


def format_sentiment(sentiment):
    if sentiment['label'] == 'NEGATIVE':
        return sentiment['score'] * -1
    else:
        return sentiment['score']

def get_sentiment_value(news_object, sentiment_pipeline):
    return format_sentiment(sentiment_pipeline(news_object['title'])[0])

# Function to save an image from a URL
def save_image_from_url(image_url, file_path):
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)

def main():
    # Get today's date
    today = datetime.now().strftime('%Y-%m-%d')

    # Get news articles
    project = hopsworks.login()
    dataset_api = project.get_dataset_api()
    fs = project.get_feature_store()
    news_fg = fs.get_feature_group(name="news_articles", version=5)
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
        version=2,
        primary_key=['article_id'],
        description="Today's most positive article and average rating"
    )

    print(f'sentiment: {most_positive["sentiment"]}')
    most_positive['avg_sentiment'] = news_df['sentiment'].mean()

    articles_monitoring_fg.insert(most_positive, write_options={"wait_for_job": False})

    # Put predictions for each of today's articles in feature group
    articles_predictions_fg = fs.get_or_create_feature_group(
        name="articles_predictions",
        version=2,
        primary_key=['article_id'],
        description="Sentiment ratings of articles"
    )
    prediction_df = news_df.filter(['article_id', 'pubdate', 'sentiment'], axis=1)
    articles_predictions_fg.insert(prediction_df, write_options={"wait_for_job": False})

    # Create image today's most positive article and upload to Hopsworks
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = "Create a simple news article drawing for the headline: " + most_positive['title']
    response = client.images.generate(model="dall-e-3", prompt=prompt, size="1024x1024", quality="standard", n=1)
    image_url = response.data[0].url
    save_image_from_url(image_url, './news_image.png')
    dataset_api.upload("./news_image.png", "Resources/images", overwrite=True)


if __name__ == "__main__":
    main()
