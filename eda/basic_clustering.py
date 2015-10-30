import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from utils import read_forum_json


def to_daily_active_counts(df):
    df['time'] = pd.to_datetime(df.created_at, unit='ms')
    df['date'] = df.time.apply(lambda x: x.date())

    author_date = df.groupby(['author_id', 'date']).size()
    author_date = pd.DataFrame(author_date, columns=['count']).reset_index()

    rows = {}
    for author in author_date.author_id.unique():
        tmp = author_date[author_date.author_id == author]
        row = pd.Series(tmp['count'])
        row.index = tmp.date
        rows[author] = row

    data = pd.DataFrame(rows).T.fillna(0)
    return data


def transform_data(data, trans_type='log'):
    """Transform data

    Args:
        data (DataFrame): input data
        trans_type (str): available types
            'identity': identical to input
            'log': add 1 and take log
            'row_norm': normalize by sum of rows
            'log_diff': apply 'log' and then take diff between days

    Returns:
        transformed data
    """
    if trans_type == 'identity':
        return data
    elif trans_type == 'log':
        return (data + 1).applymap(np.log)
    elif trans_type == 'row_norm':
        sums = data.sum(1)
        return data.apply(lambda x: x / sums)
    elif trans_type == 'log_diff':
        data_log = (data + 1).applymap(np.log)
        data_log_diff = data_log.diff(axis=1)
        return data_log_diff.drop(data_log_diff.columns[0], axis=1)


def make_plot(data, labels, label):
    plt.figure(figsize=(12, 8))
    for row in data[labels == label].index:
        plt.plot(data.loc[row])


def main():
    df = read_forum_json('json/levergunscommunity.com.json')
    data = to_daily_active_counts(df)
    transformed = transform_data(data)
    clustering = SpectralClustering()
    labels = clustering.fit_predict(transformed)
    print "Cluster sizes:", np.bincount(labels)
    make_plot(data, labels, 7)


if __name__ == '__main__':
    main()
