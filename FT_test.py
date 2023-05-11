import gensim
import numpy as np
from scipy import spatial

ft_model = gensim.models.FastText.load(
    r"C:\Users\user\PycharmProjects\diplom_new\diplom\database\models\ft_model")


def avg_feature_vector(sentence, model, num_features):
    feature_vec = np.zeros((num_features,), dtype='float32')
    n_words = 0
    for word in sentence:
            n_words += 1
            feature_vec = np.add(feature_vec, model.wv[word])
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

file = open(r"C:\Users\user\PycharmProjects\diplom_new\diplom\database\txt\История1.txt", "r", encoding="utf-8")
txt1 = file.read()

file = open(r"C:\Users\user\PycharmProjects\diplom_new\diplom\database\txt\Automatisation.txt", "r", encoding="utf-8")
txt2 = file.read()

t1 = "Математика физика информатика"
t2 = "История искусство русский язык"

s1_afv = avg_feature_vector(txt1.split(), model=ft_model, num_features=62)
s2_afv = avg_feature_vector(txt2.split(), model=ft_model, num_features=62)


sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)

print(sim)

