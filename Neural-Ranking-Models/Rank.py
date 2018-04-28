from __future__ import absolute_import
from __future__ import print_function
import numpy as np

np.random.seed(1337)

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, TimeDistributed, Embedding, Flatten, Convolution2D, Reshape
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

def embModel(input_dim, word_rep, vocabulary_size = 51253):
    rep_layer1 = Input(shape = (input_dim,), name="rep_layer1")
    rep_layer2 = Input(shape = (input_dim,), name="rep_layer2")
    word_emb = Embedding(input_dim=vocabulary_size + 1, output_dim=word_rep, input_length=input_dim)
    emb_sum = Convolution2D(filters=1, kernel_size=(input_dim, 1), strides=(input_dim, 1), padding='same', name="weight_sum_")
    h1 = Dense(128, activation="relu")
    h2 = Dense(128, activation="relu")
    h3 = Dense(128, activation="sigmoid")
    
    emb_a = word_emb(rep_layer1)
    emb_a = Reshape((input_dim, word_rep, 1))(emb_a)
    emb_sum_a = emb_sum(emb_a)
    emb_sum_a = Flatten()(emb_sum_a)
    
    emb_b = word_emb(rep_layer2)
    emb_b = Reshape((input_dim, word_rep, 1))(emb_b)
    emb_sum_b = emb_sum(emb_b)
    emb_sum_b = Flatten()(emb_sum_b)
    
    encoded_a = h3(h2(h1(emb_sum_a)))
    encoded_b = h3(h2(h1(emb_sum_b)))
    
    predicts = predicts = Lambda(relative_distance, output_shape=(1,))([encoded_a, encoded_b])
    model = Model(input=[rep_layer1, rep_layer2], output=predicts)
    return model    

if __name__ == "__main__":
    batch_size = 16    
    nb_epoch = 50
    validation_split = 0.1
    input_dim = 50

    data_a = np.random.rand(100, input_dim)*3
    data_b = np.random.rand(100, input_dim)*3
    labels = np.sum(data_a - data_b, axis=1)
    
    model = baseModel(input_dim)
    #model = embModel(input_dim, 64)
    model.summary()
    model.compile(optimizer='Adam',
              loss= 'hinge',
              metrics=['accuracy'])
    
    model.fit([data_a, data_b], labels,
                batch_size=batch_size,
                nb_epoch=nb_epoch,
                verbose=1,
                shuffle=True,
                validation_split=validation_split)

    