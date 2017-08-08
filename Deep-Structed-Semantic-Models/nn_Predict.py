#-*- coding: utf-8 -*-
''' Import theano and numpy '''
import theano
import numpy as np

''' Import keras to build a DL model '''
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Activation
import cPickle as Pickle
from keras.layers import Activation, Input

''' Import other class '''
import evaluate
import ProcDoc
from collections import defaultdict

WORD_DEPTH = 51253
K = 300 # Dimensionality of the projetion layer. See section 3.1.
L = 128 # Dimensionality of latent semantic space. See section 3.1.
J = 3 # Number of random unclicked documents serving as negative examples for a query. See section 3.

def normalize(lsa_model):
    for idx, m_vec in enumerate(lsa_model):
        lsa_model[idx] /= np.sqrt((m_vec ** 2).sum(axis = 0))
    return lsa_model


# Input tensors holding the query, positive (clicked) document, and negative (unclicked) documents.
# The first dimension is None because the queries and documents can vary in length.
input_layer = Input(shape = (WORD_DEPTH,))

# Latent Semantic Model
# projection high dimension to low.
proj = Dense(K, name="proj_1", activation="tanh")(input_layer)
proj_2 = Dense(K, name="proj_2", activation="tanh")(proj)
sem = Dense(L, name="sem", activation = "tanh")(proj_2)

model = Model(inputs=input_layer, outputs=sem)

print 'Building a model whose optimizer=adadelta, activation function=softmax'

#model.load_weights("NN_Model/DSSM_WEIGHTS_SD.h5", by_name=True)
test = load_model("NN_Model/DSSM_TD_54shuffle.h5")

with open("model/UM/query_model.pkl", "rb") as f: query_model = Pickle.load(f)
with open("model/UM/query_list.pkl", "rb") as f:  query_list = Pickle.load(f)
with open("model/UM/doc_model.pkl", "rb") as f: doc_model = Pickle.load(f)
with open("model/UM/doc_list.pkl", "rb") as f:  doc_list = Pickle.load(f)
with open("../Information-retrieval/Corpus/qry_train_set.pkl", "rb") as f: qry_train_set = Pickle.load(f)


qry_vec = []
d1_vec = []
d2_vec = []
d3_vec = []
d4_vec = []
x_train = []
qry_mid = len(qry_train_set) / 2
for idx, trn in enumerate(qry_train_set[qry_mid:qry_mid + 1000]):
    qry_idx = query_list.index(trn[0])
    qry_vec.append(query_model[qry_idx])
    [d1_n, d2_n, d3_n, d4_n] = trn[1:]
    d1_vec.append(doc_model[doc_list.index(d1_n)])
    d2_vec.append(doc_model[doc_list.index(d2_n)])
    d3_vec.append(doc_model[doc_list.index(d3_n)])
    d4_vec.append(doc_model[doc_list.index(d4_n)])


qry_vec = np.vstack(qry_vec)
d1_vec = np.vstack(d1_vec)
d2_vec = np.vstack(d2_vec)
d3_vec = np.vstack(d3_vec)
d4_vec = np.vstack(d4_vec)

result = test.predict([qry_vec, d2_vec, d1_vec, d3_vec, d4_vec])
print result.sum(axis = 0)

'''
background = ProcDoc.read_background()

query_lsa = np.array(model.predict_on_batch(query_model))
doc_lsa = np.array(model.predict_on_batch(doc_model))

print query_lsa.shape
print doc_lsa.shape

query_lsa = normalize(query_lsa)
doc_lsa = normalize(doc_lsa)

query_result = np.dot(query_lsa, doc_lsa.T)
result = np.argsort(-query_result, axis = 1)
evl = evaluate.evaluate_model(True)
query_docs_ranking = defaultdict(dict)

for q_idx in xrange(len(query_list)):
    docs_ranking = []
    for doc_idx in result[q_idx]:
        docs_ranking.append(doc_list[doc_idx])
        query_docs_ranking[query_list[q_idx]] = docs_ranking

mAP = evl.mean_average_precision(query_docs_ranking)
print mAP
'''
