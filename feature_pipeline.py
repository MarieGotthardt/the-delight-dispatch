import hopsworks
from newsdataapi import NewsDataApiClient
from datetime import datetime
import pandas as pd
import os

def main():
    project = hopsworks.login()
    fs = project.get_feature_store()

    # temporary key access method (add an API secret to repo)
    #with open("./NEWS_DATA_KEY.txt", "r") as file:
    #    api_key = file.read()
    api_key= os.environ["NEWS_DATA_KEY"]

    # Create a NewsData API client
    client = NewsDataApiClient(apikey=api_key)

    # Today's date ----- is this needed/helpful anywhere?
    today = datetime.now().strftime('%Y-%m-%d')

    # Collect articles from many countries in a list
    all_articles = []
    country_codes = ['us', 'ca', 'ie', 'gb', 'au', 'nz'] # https://newsdata.io/news-sources
    language = 'en'

    # Loop over countries
    for country_code in country_codes:
        # Fetch the news
        response = client.news_api(country=country_code, language=language)

        # Process the response
        if 'status' in response and response['status'] == 'success':
            articles = response['results']
            # Process the articles as needed
            for article in articles:
                all_articles.append(article)
        else:
            print("Failed to retrieve data:", response.get('message', 'Unknown Error'))

    # Put all the articles in a dataframe
    news_df = pd.DataFrame(all_articles)

    # Remove columns that often have null values
    news_df = news_df[['article_id', 'title', 'link', 'description', 'content', 'pubDate', 'source_id', 'country', 'category', 'language']]

    # Remove rows that still have a null value somewhere
    news_df = news_df.dropna()

    # Reformat date such that the time is dropped and just the date is kept
    news_df['pubDate'] = pd.to_datetime(news_df['pubDate']).dt.date.astype("string")

    # Reformat data types of category and country to string
    news_df['country'] = news_df['country'].astype("string")
    news_df['category'] = news_df['category'].astype("string")

    # Put articles in feature store
    news_fg = fs.get_or_create_feature_group(
        name="news_articles",
        version=6,
        primary_key=['article_id'],
        description="News articles dataset"
    )
    
    news_fg.insert(news_df)

if __name__ == "__main__":
    main()
