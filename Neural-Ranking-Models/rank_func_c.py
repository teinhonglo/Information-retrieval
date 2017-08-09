import numpy as np

np.random.seed(1337)

from keras.models import Sequential, Model
import theano
from keras.engine.topology import Layer
from keras.layers import Dense, Dropout, Input, Lambda, merge, Embedding, LSTM, Convolution1D
from keras.layers.core import Reshape
from keras import backend as K
from keras.engine.topology import Layer

''' weight softmax function '''
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

''' custom loss function'''
def hinge_loss(y_true, y_pred):
    epsilon = 1.
    return K.mean(K.maximum(0, epsilon-(K.sign(y_true) * y_pred)), axis=-1 )

''' calculate the dist '''
def relative_distance(vects):
    x, y = vects
    print x, y
    return x - y

def dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

# Network Parameter
batch_size = 16    
epochs = 50
validation_split = 0.1
vocab_size = 51253
embed_dim = 64
input_dim = 50

# Number of the input(query and document)
qry_vec = Input(shape=(input_dim,), name = "qry_input") 
doc_vec = Input(shape=(input_dim,), name = "doc_input") 

# Embeddings Layer
emb = Embedding(vocab_size, embed_dim, input_length = input_dim, name="emb_layer")
with_gamma = Convolution1D(1, 1, padding = "same", input_shape = (input_dim, embed_dim), activation = "linear", use_bias = False)
seq_layer = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]), name="seq_layer")
# emb_rsp = Reshape((input_dim, embed_dim), name = "emb_reshape")
# seq_layer = Dense(1)

# Base Network
h1 = Dense(128, input_shape=(input_dim,), name = "h1")
h1_dp = Dropout(0.2)
h2 = Dense(256, activation='relu', name = "h2")
h2_dp = Dropout(0.2)
h3 = Dense(128, activation='relu', name = "h3")
h3_dp = Dropout(0.2)
h4_rk = Dense(1, activation='sigmoid', name = "h4_score")

# Embedding and Pojection
qry_rep = seq_layer(with_gamma(emb(qry_vec)))
doc_rep = seq_layer(with_gamma(emb(doc_vec)))

qry_h1 = h1_dp(h1(qry_rep))
doc_h1 = h1_dp(h1(doc_rep))

qry_h2 = h2_dp(h2(qry_h1))
doc_h2 = h2_dp(h2(doc_h1))

qry_h3 = h3_dp(h3(qry_h2))
doc_h3 = h3_dp(h3(doc_h2))

encoded_qry = h4_rk(qry_h3)
encoded_doc = h4_rk(doc_h3)

# Concatenate the two vectors
predicts = Lambda(relative_distance,
                  output_shape=dist_output_shape)([encoded_qry, encoded_doc])
                  
# Define a trainable model linking the the predicts
model = Model(inputs=[qry_vec, doc_vec], outputs=predicts)

model.compile(optimizer='Adam',
              loss=hinge_loss,
              metrics=['accuracy'])

model.summary()

# Preprocess data
data_a = np.random.rand(100, input_dim) * 3
data_b = np.random.rand(100, input_dim) * (-3)

da_sub_db = np.sum(data_a - data_b, axis = 1)
ab_min = np.min(da_sub_db, axis = 0)
ab_max = np.max(da_sub_db, axis = 0)
labels = (da_sub_db - ab_min) / (ab_max - ab_min)

from keras.utils import plot_model
plot_model(model, to_file='model.png')

history_adam = model.fit([data_a, data_b], labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        shuffle=True,
                        validation_split=validation_split)
                        
#model.save('rank_function.h5')
