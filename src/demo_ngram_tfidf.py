import pickle


with open('../pickle/concepts.pickle', 'rb') as f:
    ngram_concepts = pickle.load(f)

concepts = []
for n_cs in ngram_concepts.values():
    concepts.extend(list(n_cs.items()))
concepts.sort(reverse=True, key=lambda t: t[1])

n_print = 50

print(' ---------------------------------- ')
print('|  tfidf    |  n-gram term         |')
print('|-----------|----------------------|')
for c, v in concepts[:n_print]: print('|  {:>7.3f}  |  {:<18}  |'.format(v, c))
print(' ---------------------------------- ')
