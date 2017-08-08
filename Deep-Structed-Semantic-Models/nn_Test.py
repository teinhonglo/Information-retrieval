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
from collections import defaultdict
import os
import ProcDoc

WORD_DEPTH = 51253
K = 300 # Dimensionality of the projetion layer. See section 3.1.
L = 128 # Dimensionality of latent semantic space. See section 3.1.
J = 3 # Number of random unclicked documents serving as negative examples for a query. See section 3.

def normalize(lsa_model):
    for idx, m_vec in enumerate(lsa_model):
        lsa_model[idx] /= np.sqrt((m_vec ** 2).sum(axis = 0))

    return lsa_model

# Input tensors holding the query, positive (clicked) document, and negative (unclicked) documents.
input_layer = Input(shape = (WORD_DEPTH,))
doc_input_layer = Input(shape = (WORD_DEPTH,))

# Latent Semantic Model
# projection high dimension to low.

proj = Dense(K, name="proj_1", activation="tanh")(input_layer)
proj_2 = Dense(K, name="proj_2", activation="tanh")(proj)
sem = Dense(L, name="sem", activation = "tanh")(proj_2)
nn_model = Model(input=input_layer, output=sem)

'''
doc_proj = Dense(K, name="d_proj_1", activation="tanh")(doc_input_layer)
doc_proj_2 = Dense(K, name="d_proj_2", activation="tanh")(doc_proj)
doc_sem = Dense(L, name="d_sem", activation = "tanh")(doc_proj_2)
doc_nn = Model(inputs=doc_input_layer, outputs=doc_sem)
'''

print 'Building a model whose optimizer=adadelta, activation function=softmax'

filepath = "Epochs/with_gamma_single_TF_Spk"
filename = []

#doc_nn.load_weights("NN_Model/DSSM_WEIGHTS_TD_54_double_gamma_shuffle.h5", by_name=True)

emb_model = load_model("NN_Model/RLE_SWLM_E_S.h5")
with open("model/UM/test_query_model_short.pkl", "rb") as file:query_model = Pickle.load(file)
with open("model/UM/test_query_list_short.pkl", "rb") as file:query_list = Pickle.load(file)
with open("model/test_query_model_short.pkl", "rb") as file:query_TF = Pickle.load(file)
with open("model/log_doc_model_s.pkl", "rb") as file:doc_model = Pickle.load(file)
with open("model/doc_list_s.pkl", "rb") as file:doc_list = Pickle.load(file)

bg_md = ProcDoc.read_background_dict()
query_model = 0.5 * query_model + 0.5 *np.array(emb_model.predict_on_batch(query_model))
query_model *= query_TF.sum(axis = 1).reshape(16, 1) * 10
#doc_model = doc_model


max_mAP = ["", 0]
evl = evaluate.evaluate_model(False)

for dir_item in os.listdir(filepath):
    # join dir path and file name
    dir_item_path = os.path.join(filepath, dir_item)
    # check whether a file exists before read
    if os.path.isfile(dir_item_path):
        dir_item_path = "Epochs/with_gamma_single_TF_Spk/54_log_gamma_shuffle_spk_weights-30-0.11.hdf5"
        #if dir_item_path.find("log") == -1: continue
        print dir_item_path
        nn_model.load_weights(dir_item_path, by_name=True)

        query_lsa = np.array(nn_model.predict_on_batch(query_model))
        doc_lsa = np.array(nn_model.predict_on_batch(doc_model))

        print query_lsa.shape
        print doc_lsa.shape

        query_lsa = normalize(query_lsa)
        doc_lsa = normalize(doc_lsa)

        # cosine similarity
        query_result = np.dot(query_lsa, doc_lsa.T)
        result = np.argsort(-query_result, axis = 1)
        query_docs_ranking = defaultdict(dict)

        for q_idx in range(len(query_list)):
            docs_ranking = []
            for doc_idx in result[q_idx]:
                docs_ranking.append(doc_list[doc_idx])
            query_docs_ranking[query_list[q_idx]] = docs_ranking

        mAP = evl.mean_average_precision(query_docs_ranking)
        print mAP
        if mAP > max_mAP[1]:
            max_mAP = [dir_item_path, mAP]
    break
print max_mAP
