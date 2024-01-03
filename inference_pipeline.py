import hopsworks
from datetime import datetime
import pandas as pd
from transformers import pipeline
from openai import OpenAI
import requests
import os
from ast import literal_eval
from graph_generation import *
from article_summarization import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager

mpl.rcParams['text.color'] = 'black'
mpl.rc('font',family='Courier New')



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
    news_fg = fs.get_feature_group(name="news_articles", version=6)
    news_df = news_fg.read()

    # Check data types of category and country
    print(f'type of country: {type(news_df["country"].iloc[0])}')
    print(f'type of category: {type(news_df["category"].iloc[0])}')

    # Only keep news articles from today
    news_df = news_df[news_df['pubdate'] == today]

    # Add sentiments for articles
    sentiment_pipeline = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")
    news_df['sentiment'] = news_df.apply(get_sentiment_value, sentiment_pipeline=sentiment_pipeline, axis=1)

    # Calculate today's average sentiment
    avg_sentiment = news_df['sentiment'].mean()

    # Find today's most positive article
    most_positive = news_df.loc[news_df['sentiment'].idxmax()]
    most_positive = pd.DataFrame(most_positive).T
    most_positive['sentiment'] = most_positive['sentiment'].astype('float')

    # Cast the category and the country columns
    # most_positive["country"] = most_positive["country"].apply(literal_eval)
    # most_positive["category"] = most_positive["category"].apply(literal_eval)

    # Create timelines for average sentiment and most positive sentiment
    most_positive_date, most_positive_sentiment, average_sentiment = get_sentiment_history()
    most_positive_date = np.append(most_positive_date, most_positive["pubdate"].values)
    most_positive_sentiment = np.append(most_positive_sentiment, most_positive["sentiment"].values)
    average_sentiment = np.append(average_sentiment, avg_sentiment)

    # Plot most positive sentiment timeline and upload it to hopsworks
    plot_most_positive_timeline(most_positive_sentiment, most_positive_date, n=5)
    dataset_api.upload("./most_positive_timeline.png", "Resources/images", overwrite=True)

    # Plot average sentiment timeline and upload it to hopsworks
    plot_average_sentiment_timeline(average_sentiment, most_positive_date, n=5)
    dataset_api.upload("./average_sentiment_timeline.png", "Resources/images", overwrite=True)

    """
    # Summarize article content
    try:
        most_positive['content'] = most_positive.apply(summarize_article, axis=1)
    except Exception as e: # if summarization fails, leave content as it is
        print("Article could not be summarized")
        print(e)
    
    # Put most positive article and average sentiment of today in feature group
    articles_monitoring_fg = fs.get_or_create_feature_group(
        name="articles_most_positive",
        version=3,
        primary_key=['article_id'],
        description="Today's most positive article and average rating"
    )

    print(f'sentiment: {most_positive["sentiment"]}')
    most_positive['avg_sentiment'] = avg_sentiment

    articles_monitoring_fg.insert(most_positive, write_options={"wait_for_job": False})

    # Put predictions for each of today's articles in feature group
    articles_predictions_fg = fs.get_or_create_feature_group(
        name="articles_predictions",
        version=3,
        primary_key=['article_id'],
        description="Sentiment ratings of articles"
    )
    prediction_df = news_df.filter(['article_id', 'pubdate', 'sentiment'], axis=1)
    articles_predictions_fg.insert(prediction_df, write_options={"wait_for_job": False})
    
    # Create image today's most positive article and upload to Hopsworks
    try:
        OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = "Create a simple and purely visual illustration of: " + most_positive.iloc[0]['title']
        response = client.images.generate(model="dall-e-3", prompt=prompt, size="1024x1024", quality="standard", n=1)
        image_url = response.data[0].url
        save_image_from_url(image_url, './news_image.png')
        dataset_api.upload("./news_image.png", "Resources/images", overwrite=True)
    except Exception as e: # API did not allow new image to be created (app will use a default image instead)
        print("OpenAI could not generate an image")
        print("Headline: " + most_positive.iloc[0]['title'])
        print(e)

    """
if __name__ == "__main__":
    main()
