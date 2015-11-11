import pandas as pd
import matplotlib.pyplot as plt
from gensim.models.ldamodel import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, utils
from utils import read_forum_json, ADDITIONAL_STOPWORDS


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
    df = df.copy()
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


def to_daily_agg_contents(df):
    """Aggregate contents produced by the same author daily"""
    df = df.copy()
    df['time'] = pd.to_datetime(df.created_at, unit='ms')
    df['date'] = df.time.apply(lambda x: x.date())

    author_date = df.groupby(['author_id', 'date'])['content'].sum()
    author_date = pd.DataFrame(author_date, columns=['content']).reset_index()

    data = author_date.pivot('author_id', 'date', 'content').fillna('')
    return data


def plot_author(author_week_df, author_id):
    author_week_df.loc[author_id].plot()
    plt.title('Weekly activity count for author ID {}'.format(author_id))
    plt.xlabel('time')
    plt.ylabel('count')


def generate_corpus_weekly(series, tokenizer=None):
    """Generate corpus and dictionary from a series of documents"""
    if not tokenizer:
        tokenizer = utils.simple_preprocess

    documents = series.values
    texts = [tokenizer(doc) for doc in documents]
    texts = [[w.lower() for w in doc
              if (w.lower() not in STOPWORDS and
                  w.lower() not in ADDITIONAL_STOPWORDS)]
             for doc in texts]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus, dictionary


def lda_weekly(series, n_topics=5):
    """Do LDA on a column"""
    corpus, dictionary = generate_corpus_weekly(series)
    lda = LdaModel(corpus, num_topics=n_topics, id2word=dictionary)
    return lda, corpus, dictionary


def weekly_keywords(col):
    """Extract keywords for each account in a column (week)"""
    lda, corpus = lda_weekly(col)
    str_to_corpus = {a: b for a, b in zip(col.values, corpus)}
    new_col = {key: ('' if not value else get_keywords_from_lda(
        lda, str_to_corpus[value]))
               for key, value in col.to_dict().items()}
    return pd.Series(new_col)


def get_keywords_from_lda(lda, doc, n_keywords=10):
    keywords = []
    prob_step = 1.0 / n_keywords
    topic_dist = lda[doc]
    for topic, prob in topic_dist:
        n_words = int(prob / prob_step)
        if n_words > 0:
            prob_word_pairs = lda.show_topic(topic)[:n_words]
            words = [pair[1] for pair in prob_word_pairs]
            keywords.extend(words)
    return ' '.join(keywords)


def main():
    # get weekly counts for each author and plot one
    df = read_forum_json('json/levergunscommunity.com.json')
    author_date_df = to_daily_active_counts(df)
    author_week_df = aggregate_weekly(author_date_df)
    plot_author(author_week_df, '1002')

    # Get weekly contents produced for each author and do lda for that week
    author_date_content = to_daily_agg_contents(df)
    author_week_content = aggregate_weekly(author_date_content)
    lda, corpus, dictionary = lda_weekly(author_week_content.iloc[:, 50])
    # infer topic for the first document which corresponds to the first row
    print lda[corpus[0]]

    # Get keywords for each author in each week by doing LDA
    author_week_keywords = author_week_content.apply(weekly_keywords, axis=0)
