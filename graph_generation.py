import hopsworks
from datetime import datetime
import hopsworks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.color'] = 'black'
#mpl.rc('font',family='Courier New')
mpl.rcParams['font.family'] = 'Courier New'

def get_sentiment_history():
    project = hopsworks.login()
    fs = project.get_feature_store()

    most_positive_fg = fs.get_feature_group(name="articles_most_positive", version=3)
    most_positive_df = most_positive_fg.read()
    most_positive_sentiment = most_positive_df["sentiment"].values
    most_positive_date = most_positive_df["pubdate"].values
    average_sentiment = most_positive_df["avg_sentiment"].values
    return most_positive_date, most_positive_sentiment, average_sentiment


def plot_most_positive_timeline(most_positive_sentiment, most_positive_date, n):
    if len(most_positive_date) > n:
        most_positive_date = most_positive_date[-n:]
        most_positive_sentiment = most_positive_sentiment[-n:]
    else:
        n = len(most_positive_date)
        
    plt.rcParams['font.family'] = 'Courier New'
    plt.plot(most_positive_date, most_positive_sentiment, '--o', color="teal", label="Most Positive Sentiment")
    plt.xlabel("Date", fontsize=12)
    plt.xticks(fontsize=10)
    plt.ylabel("Sentiment Rating", fontsize=12)
    plt.ylim(-1, 1.1)
    plt.yticks(fontsize=10)
    plt.axhline(y=0, xmin=0, xmax=len(most_positive_date), color='gray', linestyle='--', label="Neutral Sentiment")
    plt.title(f"Sentiment Ratings of Most Positive Articles \n for the Past {n} Days ", fontsize=12)
    plt.legend(loc="lower right")
    plt.savefig('./most_positive_timeline.png')
    plt.show()
    plt.close()


# Plot historical data for the average sentiment
def plot_average_sentiment_timeline(average_sentiment, most_positive_date, n):
    if len(most_positive_date) > n:
        most_positive_date = most_positive_date[-n:]
        average_sentiment = average_sentiment[-n:]
    else:
        n = len(most_positive_date)

    plt.plot(most_positive_date, average_sentiment, '--o', color="darkred", label="Average Sentiment")
    plt.xlabel("Date", fontname="Courier New", fontsize=12)
    plt.xticks(fontname="Courier New", fontsize=10)
    plt.ylabel("Sentiment Rating", fontname="Courier New", fontsize=12)
    plt.ylim(-1, 1.1)
    plt.yticks(fontname="Courier New", fontsize=10)
    plt.axhline(y=0, xmin=0, xmax=len(most_positive_date), color='gray', linestyle='--', label="Neutral Sentiment")
    plt.title(f"Average Sentiment Ratings of the Past {n} Days ", fontname="Courier New", fontsize=12)
    plt.legend(loc="lower right")
    plt.savefig('./average_sentiment_timeline.png')
    plt.show()
    plt.close()
