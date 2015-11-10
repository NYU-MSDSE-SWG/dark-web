import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from utils import read_forum_json


def to_daily_active_counts(df):
    """Get daily activity counts for each user

    Args:
        df (DataFrame): data frame with one post per row, containing info like
        author and date.

    Returns:
        DataFrame: data frame where each row represents a user and each
            column represents a date, values are counts of posts from that user
            on that date
    """
    df['time'] = pd.to_datetime(df.created_at, unit='ms')
    df['date'] = df.time.apply(lambda x: x.date())

    author_date = df.groupby(['author_id', 'date']).size()
    author_date = pd.DataFrame(author_date, columns=['count']).reset_index()

    data = author_date.pivot('author_id', 'date', 'count').fillna(0)
    return data


def aggregate_weekly(df):
    """Get weekly aggregated activity counts per user based on daily counts

    Args:
        df (DataFrame): data frame where each row represents a user and each
            column represents a date, values are counts of posts from that user
            on that date

    Returns:
        DataFrame: similar data frame, but aggregated weekly instead of daily
    """
    col_names = df.columns
    if len(col_names) == 0:
        raise ValueError

    index = 0
    week_start = col_names[0]
    sums = {}
    start_idx = 0

    while index < len(col_names):
        if (col_names[index] - week_start).days < 7:
            pass
        else:
            sums[week_start.date()] = df.iloc[:, start_idx:index].sum(1)
            week_start = col_names[index]
            start_idx = index
        index += 1

    return pd.DataFrame(sums)


def plot_author(author_week_df, author_id):
    author_week_df.loc[author_id].plot()
    plt.title('Weekly activity count for author ID {}'.format(author_id))
    plt.xlabel('time')
    plt.ylabel('count')


def main():
    df = read_forum_json('json/levergunscommunity.com.json')
    author_date_df = to_daily_active_counts(df)
    author_week_df = aggregate_weekly(author_date_df)
    plot_author(author_week_df, '1002')
