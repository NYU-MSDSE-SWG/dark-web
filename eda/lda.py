from gensim.models.ldamulticore import LdaMulticore
from utils import read_forum_json, generate_corpus


def main():
    df = read_forum_json('json/levergunscommunity.com.json')
    corpus, dictionary = generate_corpus(df)
    lda = LdaMulticore(corpus, num_topics=20, id2word=dictionary, workers=3)
    lda.print_topics(num_topics=20, num_words=20)


if __name__ == '__main__':
    main()
