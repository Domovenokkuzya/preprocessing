import pickle

import gensim
import numpy as np


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


w2v_model = gensim.models.Word2Vec.load(
    r"C:\Users\user\PycharmProjects\diplom_new\diplom\database\models\w2v_model")
index2word_set = set(w2v_model.wv.index_to_key)

LR_model = pickle.load(open(r'C:\Users\user\PycharmProjects\diplom_new\diplom\database\models\model.pkl', 'rb'))

file = open(r"C:\Users\user\PycharmProjects\diplom_new\diplom\database\txt\Automatisation.txt", "r", encoding="utf-8")
txt = file.read()

X_test_vect_avg = []

s_afv = avg_feature_vector(txt.split(), model=w2v_model, num_features=62, index2word_set=index2word_set)
X_test_vect_avg.append(s_afv)

answer = LR_model.predict(X_test_vect_avg)
print(answer, len(X_test_vect_avg))

