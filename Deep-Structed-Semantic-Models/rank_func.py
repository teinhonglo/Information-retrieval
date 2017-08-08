from __future__ import absolute_import
from __future__ import print_function
import numpy as np

np.random.seed(1337)

from keras.models import Sequential, Model
import theano
from keras.engine.topology import Layer
from keras.layers import Dense, Dropout, Input, Lambda, merge, Embedding, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras import backend as K
from keras.engine.topology import Layer

def softmax_reg(weight_matrix):
    ndim = K.ndim(weight_matrix)
    if ndim == 2:
        return K.softmax(weight_matrix)
    elif ndim > 2:
        e = K.exp(weight_matrix - K.max(weight_matrix, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')

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
    seq.add(Dense(256, activation='tanh'))
    seq.add(Dropout(0.2))
    seq.add(Dense(128, activation='tanh'))
    seq.add(Dropout(0.2))
    seq.add(Dense(1, activation='tanh'))
    return seq
    
batch_size = 16    
nb_epoch = 50
validation_split = 0.1
input_dim = 50

data_a = np.random.rand(100, input_dim) * 3
data_b = np.random.rand(100, input_dim) * (-3)
'''
emb = input_representation(51253, 64, 50)
emb.compile('rmsprop', 'mse')
emb.summary()
output_array = emb.predict(data_a)
print(output_array.shape)


count =0
for layer in emb.layers:
    if count == 0: 
        count += 1
        continue
    print("Layer")
    weights = layer.get_weights()[0]
    print(len(weights))
    print(weights)
    print(np.array(weights).shape)
    count += 1
'''

labels = np.sum(data_a - data_b, axis = 1)
labels = labels.astype(np.float32)


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
              loss=hinge_loss,
              metrics=['accuracy'])
model.summary()

'''
history_adam = model.fit([data_a, data_b], labels,
                        batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        verbose=1,
                        shuffle=True,
                        validation_split=0.1
                           )

'''
from keras.utils import plot_model
plot_model(model, to_file='model.png')
'''
from keras.utils.visualize_util import plot
plot(model, to_file='model.png')                        
#model.save('rank_function.h5')                                
'''

