from __future__ import absolute_import
from __future__ import print_function
import numpy as np

np.random.seed(1337)

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import Adam
from keras import backend as K

def hinge_loss(y_true, y_pred):
	epsilon = 1.
	K.maximum(0, K.sign(y_true) * y_pred)

def relative_distance(vects):
    x, y = vects
    return x - y

def baseModel(input_dim):
    rep_layer1 = Input(shape = (input_dim,), name="rep_layer1")
    rep_layer2 = Input(shape = (input_dim,), name="rep_layer2")
    h1 = Dense(128, activation="relu")
    #d1 = Dropout(0.2)
    h2 = Dense(128, activation="relu")
    h3 = Dense(128, activation="sigmoid")
    encoded_a = h3(h2(h1(rep_layer1)))
    encoded_b = h3(h2(h1(rep_layer2)))
    predicts = Lambda(relative_distance, output_shape=(1,))([encoded_a, encoded_b])
    # Define a trainable model linking the the predicts
    model = Model(input=[rep_layer1, rep_layer2], output=predicts)
    return model

def embModel(input_dim, vocabulary_size + 1):
    rep_layer1 = Input(shape = (input_dim,), name="rep_layer1")
    rep_layer2 = Input(shape = (input_dim,), name="rep_layer2")
    word_emb = TimeDistributed(Embedding(input_dim=vocabulary_size, output_dim=word_rep, input_length=maxlen))
    sum = Dense(1)
    h1 = Dense(128, activation="relu")
    h2 = Dense(128, activation="relu")
    h3 = Dense(128, activation="sigmoid")
    encoded_a = h3(h2(h1(sum(word_emb(rep_layer1)))))
    encoded_b = h3(h2(h1(sum(word_emb(rep_layer2)))))
    predicts = predicts = Lambda(relative_distance, output_shape=(1,))([encoded_a, encoded_b])
    model = Model(input=[input_layer], output=predicts)
    return model    

    