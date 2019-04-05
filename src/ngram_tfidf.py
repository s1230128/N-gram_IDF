import pickle
import nltk
from collections import Counter
from math import log
import pprint
from nltk.corpus import stopwords
from inflector import Inflector


def ngram_tfidf(text, ngrams=[1,2,3,4,5,6,7,8,9,10]):
    # data
    words = nltk.word_tokenize(text.lower())
    # stopwords
    stops = set(stopwords.words('english'))
    # total number of documents
    N = 4379810

    ng_tf = dict()
    ng_tfidf = dict()
    for n in ngrams:
        # load of sdf
        path = '../pickle/ngram_sdf/' + str(n) + '-gram.pickle'
        with open(path, 'rb') as f:
            sdf = pickle.load(f)

        terms = [words[i:i+n] for i in range(len(words) - n)]
        terms = [t for t in terms if all(w not in t for w in stops)]     # stopwords filter
        terms = [[Inflector().singularize(w) for w in t] for t in terms] # singularize
        terms = [' '.join(t) for t in terms]
        terms = [t for t in terms if t in sdf]

        # tf
        tf = Counter(terms)
        tfidf = dict()
        # tfidf
        for t in tf.keys():
            tfidf[t] = tf[t] * log(N / sdf[t])

        ng_tf[n] = tf
        ng_tfidf[n] = tfidf

    return ng_tf, ng_tfidf


def main():
    with open('../data/wiki/computer.txt', 'r') as f:
        text = f.read()

    ng_tf, ng_tfidf = ngram_tfidf(text)

    with open('../pickle/concepts.pickle', 'wb') as f:
        pickle.dump(ng_tfidf, f)


if __name__ == '__main__': main()
