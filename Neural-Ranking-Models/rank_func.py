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

def absolute_distance(vects):
    x, y = vects
    return x - y

def base_network(input_dim):
	seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,))
    seq.add(Dropout(0.2))
    seq.add(Dense(256, activation='relu'))
    seq.add(Dropout(0.2))
	seq.add(Dense(128, activation='relu'))
	seq.add(Dropout(0.2))
    seq.add(Dense(1, activation='tanh'))
    return seq

	
batch_size = 16	
nb_epoch = 50
validation_split = 0.1

rep_vect1 = Input(shape=(input_dim,))
rep_vect2 = Input(shape=(input_dim,))	

model = base_network(input_dim)	
encoded_a = model(rep_vect1)
encoded_b = model(rep_vect2)

# Concatenate the two vectors
merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

predicts = Lambda(absolute_distance,
                  output_shape=eucl_dist_output_shape)([encoded_a, encoded_b])
				  
# Define a trainable model linking the the predicts
model = Model(inputs=[rep_vect1, rep_vect1], outputs=predicts)

model.compile(optimizer='Adam',
              loss=hinge_loss,
              metrics=['accuracy'])

history_adam = model.fit([data_a, data_b], labels,
						batch_size=batch_size,
						nb_epoch=nb_epoch,
						verbose=1,
						shuffle=True,
						validation_split=validation_split
                   		)
						
''' Create a HDF5 file '''							
model.save('rank_function.h5')								