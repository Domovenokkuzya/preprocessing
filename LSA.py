import pandas as pd

import pickle
from gensim.corpora import Dictionary
from gensim import models, similarities
from gensim.models import Phrases
from gensim.models.phrases import Phraser

with open("D:/data/preprocessing.txt", "rb") as fp:  # Unpickling
    docs = pickle.load(fp)

bigram = Phrases(docs, min_count=10, threshold=2, delimiter=',')

bigram_phraser = Phraser(bigram) # Для биграмм, если собираетесь использовать униграммы, замените bigram на docs

bigram_token = []
for sent in docs:
    bigram_token.append(bigram_phraser[sent])

dict_ = Dictionary(bigram_token)

print(dict_)

corpus = [dict_.doc2bow(doc) for doc in docs]

lsi = models.LsiModel(corpus, id2word=dict_, num_topics=62)

lsi.save("lsa_model_bigram_minten")

courseoutcome = docs[2112]

covec = dict_.doc2bow(courseoutcome)

index = similarities.MatrixSimilarity(lsi[corpus])
lsivec = lsi[covec]

sims = index[lsivec]

copomap = []
for i, sim in enumerate(sims):
    copomap.append({
        'po': i + 1,
        'similarity': sim
    }
    )

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
cdf = pd.DataFrame(copomap)
cdf['similarity'] = cdf['similarity']
num = 1
num_5059 = 0
num_6069 = 0
num_7079 = 0
num_8089 = 0
num_90 = 0
for ind in cdf['similarity']:
    if ind > 0.79:
        print(num, ind)
    if 0.5 < ind < 0.59:
        num_5059 += 1
    if 0.6 < ind < 0.69:
        num_6069 += 1
    if 0.7 < ind < 0.79:
        num_7079 += 1
    if 0.8 < ind < 0.89:
        num_8089 += 1
    if ind > 0.9:
        num_90 += 1
    num += 1

print('В диапазоне 0.5:0.59:', num_5059, 'В диапазоне 0.6:0.69:', num_6069, 'В диапазоне 0.7:0.79:', num_7079,
      'В диапазоне 0.8:0.89:', num_8089, 'В диапазоне 0.9:1:', num_90)
