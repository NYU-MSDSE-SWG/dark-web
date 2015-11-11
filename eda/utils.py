import json
import pandas as pd
from gensim import corpora, utils
from gensim.parsing.preprocessing import STOPWORDS


ADDITIONAL_STOPWORDS = set(['photobucket', 'http', 'com', 'gif', 'jpg',
                            'image', 'images', 'www', 'albums', 'smilies'])


def read_forum_json(fpath):
    with open(fpath, 'rb') as f:
        data = [json.loads(line) for line in f.readlines()]
    for d in data:
        for key, value in d['author'].items():
            if not key.startswith('author'):
                key = 'author_' + key
            d[key] = value
        del d['author']
    df = pd.DataFrame([pd.Series(d) for d in data])
    return df


def generate_corpus(df, tokenizer=None):
    if not tokenizer:
        tokenizer = utils.simple_preprocess

    documents = df.content.values
    texts = [tokenizer(doc) for doc in documents]
    texts = [[w for w in doc if w not in STOPWORDS] for doc in texts]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus, dictionary
