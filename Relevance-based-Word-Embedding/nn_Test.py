#-*- coding: utf-8 -*-
''' Import theano and numpy '''
import theano
import numpy as np

''' Import keras to build a DL model '''
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
import cPickle as Pickle

optimizer = ["Adagrad", "Adam", "Nadam"]
losses = ["categorical_crossentropy", "kullback_leibler_divergence"]

print 'Building a model whose optimizer=adam, activation function=softmax'
model = load_model("NN_Model/RLE_SSWLM_S.h5")
query_model = Pickle.load(open("test_query_model.pkl", "rb"))
output = np.array(model.predict_on_batch(query_model))
Pickle.dump(output, open("NN_result/SSWLM_RLE_S.pkl", "wb"), True)

'''
print 'Building a model whose optimizer=adam, activation function=softmax'
model = load_model("RPE.h5")
query_model = Pickle.load(open("test_query_model.pkl", "rb"))
output = np.array(model.predict(query_model))
Pickle.dump(output, open("query_relevance_model_RPE.pkl", "wb"), True)
'''
