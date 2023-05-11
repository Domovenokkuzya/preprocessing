import pickle

import gensim
import numpy as np
from nltk import FreqDist

from functionList import frequent_words

lst = []

# w2v_model = gensim.models.Word2Vec.load(
#     r"C:\Users\user\PycharmProjects\diplom_new\diplom\database\models\w2v_model")
# index2word_set = set(w2v_model.wv.index_to_key)

file = open(r"C:\Users\user\PycharmProjects\diplom_new\diplom\database\txt\Automatisation.txt", "r", encoding="utf-8")
txt = file.read()

txt = txt.split()

frequent_words(txt, lst)

fdist = FreqDist(lst)

print(fdist.get('примеров'))

# for word in fdist.elements():
#     if word in index2word_set:
#         print(w2v_model.wv[word])

