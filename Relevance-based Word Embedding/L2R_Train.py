#-*- coding: utf-8 -*-
''' Import theano and numpy '''
import numpy as np
import cPickle as Pickle
np.random.seed(1331)

import theano
import cPickle as pickle

''' Import keras to build a DL model '''
from keras.layers import Dense, Dropout, Input, Lambda, merge, Embedding, LSTM, Convolution1D
from keras.layers.core import Reshape, Dense, Activation, Dropout
from keras.models import Sequential, Model
from keras import backend as K

model_path = "../Corpus/model/TDT2/UM/"
optimizer = ["Adagrad" , "Adam", "Nadam"]
losses = ["categorical_crossentropy", "kullback_leibler_divergence"]

''' custom loss function'''
def hinge_loss(y_true, y_pred):
    epsilon = 1.
    return K.mean(K.maximum(0, epsilon-(K.sign(y_true) * y_pred)), axis=-1 )

def kl_distance(vects):
    qry_vec, doc_vec = vects
    qry_vec = K.clip(qry_vec, K.epsilon(), 1)
    doc_vec = K.clip(doc_vec, K.epsilon(), 1)
    return K.sum(qry_vec * K.log(doc_vec), axis=-1)

def relative_distance(vects):
    x, y = vects
    return x - y

def kl_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def rel_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

with open(model_path + "query_model.pkl", "rb") as file: query_model = pickle.load(file)
X_train = query_model
with open("obj_func/TDT2/rel_supervised_swlm_entropy.pkl", "rb") as file: query_relevance = pickle.load(file)
Y_train = np.random.rand(800, 1)#np.sum(query_relevance, axis = 1)
print Y_train.shape
#print Y_train.shape

''' set the size of mini-batch and number of epochs'''
batch_size = 16 
epochs = 55
vocabulary_size = 51253
embedding_dimensionality = 350


print 'Building a model whose optimizer=adam, activation function=softmax'
#input_layer = Dense(embedding_dimensionality, input_dim = vocabulary_size, name="input_layer")
emb_layer = Dense(embedding_dimensionality, activation="linear", name="emb_layer")
rep_layer = Dense(vocabulary_size, activation="softmax", name="rep_layer")

qry = Input(shape=(vocabulary_size,), name="qry_input")
doc_plus = Input(shape=(vocabulary_size,), name="doc_plus_input")
doc_minus = Input(shape=(vocabulary_size,), name="doc_minus_input")

qry_emb = emb_layer(qry)
doc_emb_plus = emb_layer(doc_plus)
doc_emb_minus = emb_layer(doc_minus)

qry_rep = rep_layer(qry_emb)
doc_rep_plus = rep_layer(doc_emb_plus)
doc_rep_minus = rep_layer(doc_emb_minus)

plus_score = Lambda(kl_distance, output_shape=kl_shape, name="qry_plus")([qry_rep, doc_rep_plus])
minus_score = Lambda(kl_distance, output_shape=kl_shape, name="qry_minus")([qry_rep, doc_rep_minus])

predicts = Lambda(relative_distance, output_shape=rel_shape)([plus_score, minus_score])

model = Model(inputs=[qry, doc_plus, doc_minus], outputs=predicts)
model.summary()

''' Setting optimizer as Adam '''
from keras.optimizers import Adam, SGD
model.compile(loss= hinge_loss,
              optimizer='Adam')

from keras.utils import plot_model
plot_model(model, to_file='model.png')

''' Fit models and use validation_split=0.1 '''
history_adam = model.fit([X_train, X_train, X_train], Y_train,
			batch_size=batch_size,
			epochs=epochs,
			verbose=1,
			shuffle=True,
			validation_split=0
			#callbacks = [modelCheckpoint]
			# earlyStopping callbacks
			#callbacks = [earlyStopping]
			)

''' Create a HDF5 file '''							
#model.save('NN_Model/TDT2/RLE_SSWLM_E.h5')
