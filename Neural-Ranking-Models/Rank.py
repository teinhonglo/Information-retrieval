from __future__ import absolute_import
from __future__ import print_function
import numpy as np

np.random.seed(1337)

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import Adam
from keras import backend as K

def baseModel(input_dim):
    input_layer = Input(shape = (input_dim,), name="input_layer")
    h1 = Dense(128, activation="relu")(input_layer)
    #d1 = Dropout(0.2)
    h2 = Dense(128, activation="relu")(h1)
    h3 = Dense(128, activation="relu")(h2)
    predicts = Dense(1), activation='tanh')(h3)
    model = Model(input=[input_layer], output=predicts)
    return seq

def embModel(input_dim, vocabulary_size + 1):
    input_layer = Input(shape = (input_dim,), name="input_layer")
    word_emb = TimeDistributed(Embedding(input_dim=vocabulary_size, output_dim=word_rep, input_length=maxlen))(input_dim)
    emb_sum = Dense(1)(word_emb)
    h1 = Dense(128, activation="relu")(emb_sum)
    h2 = Dense(128, activation="relu")(h1)
    h3 = Dense(128, activation="relu")(h2)
    predicts = Dense(1), activation='tanh')(h3)
    model = Model(input=[input_layer], output=predicts)
    return model
