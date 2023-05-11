# import modules
import pickle
import time
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

df = pd.read_pickle('D:/data/preprocessing.txt')

df1 = pd.read_pickle('D:/data/rubricks.txt')

# get the locations
X = df
y = df1

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

size = 62

# Train the word2vec model
w2v_model = Word2Vec(X_train,
                     vector_size=size,
                     window=500,
                     min_count=35,
                     workers=4)


# w2v_model.save("w2v_model")

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


X_train_vect_avg = []
X_test_vect_avg = []
index2word_set = set(w2v_model.wv.index_to_key)

for v in X_train:
    s_afv = avg_feature_vector(v, model=w2v_model, num_features=size, index2word_set=index2word_set)
    X_train_vect_avg.append(s_afv)

for v in X_test:
    s_afv = avg_feature_vector(v, model=w2v_model, num_features=size, index2word_set=index2word_set)
    X_test_vect_avg.append(s_afv)

# Instantiate and fit a basic Random Forest model on top of the vectors
from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier()
rf.fit(X_train_vect_avg, y_train)

# Use the trained model to make predictions on the test data
y_pred = rf.predict(X_test_vect_avg)

from sklearn.metrics import precision_score, recall_score, classification_report

precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print(classification_report(y_test, y_pred, labels=np.unique(y_pred)))


from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression


lr = LogisticRegression(C=2.0, max_iter=1000)
lr.fit(X_train_vect_avg, y_train)

# Use the trained model to make predictions on the test data
y_pred = lr.predict(X_test_vect_avg)

precision1 = precision_score(y_test, y_pred, average='macro')
recall1 = recall_score(y_test, y_pred, average='macro')

print(classification_report(y_test, y_pred, labels=np.unique(y_pred)))


# pickle.dump(lr, open('model.pkl', 'wb'))

from sklearn.neighbors import KNeighborsClassifier


kn = KNeighborsClassifier()
kn.fit(X_train_vect_avg, y_train)

# Use the trained model to make predictions on the test data
y_pred = kn.predict(X_test_vect_avg)

precision2 = precision_score(y_test, y_pred, average='macro')
recall2 = recall_score(y_test, y_pred, average='macro')

print(classification_report(y_test, y_pred, labels=np.unique(y_pred), zero_division=0))

