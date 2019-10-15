from __future__ import absolute_import
from __future__ import print_function
import numpy as np

np.random.seed(1337)

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, TimeDistributed, Embedding, Flatten, Convolution1D, Reshape
from keras.optimizers import Adam
from keras import backend as K

def baseModel(input_dim):
    input_layer = Input(shape = (input_dim,), name="input_layer")
    h1 = Dense(128, activation="relu")(input_layer)
    #d1 = Dropout(0.2)
    h2 = Dense(128, activation="relu")(h1)
    h3 = Dense(128, activation="relu")(h2)
    predicts = Dense(1, activation='tanh')(h3)
    model = Model(input=[input_layer], output=predicts)
    return model

def embModel(input_dim, word_rep, vocabulary_size = 51253):
    input_layer = Input(shape = (input_dim,), name="input_layer")
    word_emb = Embedding(input_dim=vocabulary_size + 1, output_dim=word_rep, input_length=input_dim)(input_layer)
    #word_emb = Reshape((input_dim, word_rep, 1))(word_emb)
    emb_sum = Convolution1D(filters=1, kernel_size=input_dim, strides=1, padding='valid', name="weight_sum_")(word_emb)
    emb_sum = Flatten()(emb_sum)
    h1 = Dense(128, activation="relu")(emb_sum)
    h2 = Dense(128, activation="relu")(h1)
    h3 = Dense(128, activation="relu")(h2)
    predicts = Dense(1, activation='tanh')(h3)
    model = Model(input=[input_layer], output=predicts)
    return model

if __name__ == "__main__":
    batch_size = 16    
    nb_epoch = 50
    validation_split = 0.1
    input_dim = 50

    data_a = np.random.rand(100, input_dim)*3
    data_b = np.random.rand(100, input_dim)*3
    labels = np.sum(data_a - data_b, axis=1)
    
    #model = baseModel(input_dim)
    model = embModel(input_dim, 64)
    model.summary()
    model.compile(optimizer='Adam',
              loss= 'hinge',
              metrics=['accuracy'])
'''    
    model.fit([data_a], labels,
                        batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        verbose=1,
                        shuffle=True,
                        validation_split=validation_split
                        )    
'''
