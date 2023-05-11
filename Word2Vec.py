import pandas as pd
from gensim.models import Word2Vec
import pickle
import numpy as np
from scipy import spatial

with open("D:/data/preprocessing.txt", "rb") as fp:  # Unpickling
    docs = pickle.load(fp)

model = Word2Vec(sentences=docs, vector_size=62, window=5, min_count=25, workers=4)

index2word_set = set(model.wv.index_to_key)


def avg_feature_vector(sentence, model, num_features, index2word_set):
    feature_vec = np.zeros((num_features,), dtype='float32')
    n_words = 0
    for word in sentence:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model.wv[word])
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec


file = open(r"C:/Users/user/Downloads/1.txt", "r", encoding="utf-8")
txt = file.read()

copomap = []
s1_afv = avg_feature_vector(txt.split(), model=model, num_features=62, index2word_set=index2word_set)
for i, doc in enumerate(docs):
    s2_afv = avg_feature_vector(docs[i], model=model, num_features=62, index2word_set=index2word_set)
    sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
    copomap.append({
        'po': i + 1,
        'similarity': sim
    })

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
cdf = pd.DataFrame(copomap)

num = 1
num_5059 = 0
num_6069 = 0
num_7079 = 0
num_8089 = 0
num_90 = 0
for ind in cdf['similarity']:
    if 0.84 < ind < 1.00:
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
