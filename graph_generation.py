import hopsworks
from datetime import datetime
import hopsworks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Functions for Graph Generation
"""


# Get historical data
def get_sentiment_history():
    project = hopsworks.login()
    fs = project.get_feature_store()

    most_positive_fg = fs.get_feature_group(name="articles_most_positive", version=3)
    most_positive_df = most_positive_fg.read()
    most_positive_sentiment = most_positive_df["sentiment"].values
    most_positive_date = most_positive_df["pubdate"].values
    average_sentiment = most_positive_df["avg_sentiment"].values
    return most_positive_date, most_positive_sentiment, average_sentiment

# Plot historical data for the most positive sentiment
def plot_most_positive_timeline(most_positive_sentiment, most_positive_date, n):
    if len(most_positive_date) > n:
        most_positive_date = most_positive_date[-n:]
        most_positive_sentiment = most_positive_sentiment[-n:]
    else:
        n = len(most_positive_date)

    plt.plot(most_positive_date, most_positive_sentiment, '--co')
    plt.xlabel("Date")
    plt.ylabel("Sentiment Rating")
    plt.ylim(-1, 1.1)
    plt.title(f"Sentiment Ratings of Most Positive Articles for the Past {n} Days ")
    plt.savefig('./most_positive_timeline.png')
    plt.show()


# Plot historical data for the average sentiment
def plot_average_sentiment_timeline(average_sentiment, most_positive_date, n):
    if len(most_positive_date) > n:
        most_positive_date = most_positive_date[-n:]
        average_sentiment = average_sentiment[-n:]
    else:
        n = len(most_positive_date)

    plt.plot(most_positive_date, average_sentiment, '--om')
    plt.xlabel("Date")
    plt.ylabel("Sentiment Rating")
    plt.ylim(-1, 1.1)
    plt.title(f"Average Sentiment Ratings of the Past {n} Days ")
    plt.savefig('./average_sentiment_timeline.png')
    plt.show()