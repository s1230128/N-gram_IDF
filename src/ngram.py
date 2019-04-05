import csv
import pickle


# N = 4379810
#  1  3375423
#  2 12398428
#  3 12106697
#  4  6017528
#  5  2668298
#  6  1271350
#  7   710802
#  8   464302
#  9   335280
# 10   262671

print('Loading ... ')
with open('../data/ngram', newline="") as f:
    # 書き込みと同じ設定（delimiterとquotechar）を指定します
    reader = csv.reader(f, delimiter="\t", quotechar='"')

    ngram_sdf = dict()
    for id, l, gtf, df, sdf, term in reader:
        if l not in ngram_sdf:
            ngram_sdf[l] = dict()
        term = term.strip()
        ngram_sdf[l][term] = int(sdf)

print('Saving ...')
for i in ngram_sdf.keys():
    path = '../pickle/ngram_sdf/' + str(i) + '-gram.pickle'
    with open(path, 'wb') as f:
        pickle.dump(ngram_sdf[i], f)
