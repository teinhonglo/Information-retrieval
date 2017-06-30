from __future__ import absolute_import
from __future__ import print_function
import numpy as np

np.random.seed(1337)

from keras.datasets import mnist
from keras.models import Sequential, Model
import keras
from keras.layers import Dense, Dropout, Input, Lambda, merge
from keras.optimizers import Adam
from keras import backend as K

def hinge_loss(y_true, y_pred):
    epsilon = 1.
    return K.mean(K.maximum(0, epsilon-(K.sign(y_true) * y_pred)),axis=-1 )

def relative_distance(vects):
    x, y = vects
    return x - y

def base_network(input_dim):
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,)))
    seq.add(Dropout(0.2))
    seq.add(Dense(256, activation='sigmoid'))
    seq.add(Dropout(0.2))
    seq.add(Dense(128, activation='sigmoid'))
    seq.add(Dropout(0.2))
    seq.add(Dense(1, activation='sigmoid'))
    return seq
    
data_a = np.random.rand(100,50)*3
data_b = np.random.rand(100,50)*(-3)
labels =np.asarray([1]*100)

batch_size = 16    
nb_epoch = 50
validation_split = 0.1
input_dim = 50

rep_vect1 = Input(shape=(input_dim,))
rep_vect2 = Input(shape=(input_dim,))    

model = base_network(input_dim)    
encoded_a = model(rep_vect1)
encoded_b = model(rep_vect2)

# Concatenate the two vectors
#merged_vector = merge([encoded_a, encoded_b], mode='concat')

predicts = Lambda(relative_distance,
                  output_shape=(1,))([encoded_a, encoded_b])
                  
# Define a trainable model linking the the predicts
model = Model(input=[rep_vect1, rep_vect2], output=predicts)

model.compile(optimizer='Adam',
              loss= hinge_loss,
              metrics=['accuracy'])
model.summary()


history_adam = model.fit([data_a, data_b], labels,
                        batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        verbose=1,
                        shuffle=True,
                        validation_split=validation_split
                           )
                        
#model.save('rank_function.h5')                                
