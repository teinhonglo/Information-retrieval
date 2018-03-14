import numpy as np
np.random.seed(5566)
from keras import backend as K
from keras.layers import Activation, Input, ZeroPadding2D
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalMaxPooling2D
from keras.layers.merge import concatenate, dot
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects

def log_activation(x):
    return K.log(x)

get_custom_objects().update({'log_activation': Activation(log_activation)})

def create_model(MAX_QRY_LENGTH = 50, MAX_DOC_LENGTH = 2900, NUM_OF_FEATS = 10, PSGS_SIZE = [(50, 1)], NUM_OF_FILTERS = 5, tau = 1):
    alpha_size = len(PSGS_SIZE)
    psgMat = Input(shape = (MAX_QRY_LENGTH, MAX_DOC_LENGTH, 1,), name="passage")
    homoMat = Input(shape = (NUM_OF_FEATS, ), name="h_feats")
    # Convolution2D, Meaning pooling and Max pooling.
    # Conv2D, Mean pooling, Max pooling
    M, K, r = [], [], []
    for idx, PSG_SIZE in enumerate(PSGS_SIZE):
        tau = PSG_SIZE[0] / 2
        pool_size = (MAX_QRY_LENGTH - PSG_SIZE[0]) / tau + 1
        # Convolution
        m_1 = Convolution2D(filters=NUM_OF_FILTERS, kernel_size=PSG_SIZE, strides=tau, padding='valid', name="pConv2D_" + str(idx))(psgMat)
        M.append(m_1)
        # Mean pooling
        k_1 = AveragePooling2D(pool_size=(pool_size, 1), strides=1, name="pAvePool_" + str(idx))(M[idx])
        K.append(k_1)
        # Max Pooling
        r_1 = GlobalMaxPooling2D(name="pMaxPool_" + str(idx))(K[idx])
        r.append(r_1)
    concat_r = concatenate(r)
    # Fusion Matrix and predict relevance
    # get h(q, d)
    # MLP(DENSE(len(r(q,d))))
    phi_h = Dense(alpha_size, activation="softmax", name="TrainMat")(homoMat)
    dot_prod = dot([concat_r, phi_h], axes = 1, name="rel_dot")
    # tanh(dot(r.transpose * h))
    #pred = Activation("tanh", name="activation_tanh")(dot_prod)
    pred = Dense(1, activation="sigmoid", name="activation_sigmoid")(dot_prod)
    
    # We now have everything we need to define our model.
    model = Model(inputs = [psgMat, homoMat], outputs = pred)
    model.summary()
    '''
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
    '''
    return model

if __name__ == "__main__":
    MAX_QRY_LENGTH = 1794
    MAX_DOC_LENGTH = 2907
    NUM_OF_FEATS = 10
    PSGS_SIZE = [(16, 1), (150, 1)]
    NUM_OF_FILTERS = 1
    batch_size = 4
    tau = 1
    X = np.random.rand(batch_size, MAX_QRY_LENGTH, MAX_DOC_LENGTH, 1)
    X1 = np.random.rand(batch_size, NUM_OF_FEATS)
    y = np.random.rand(batch_size)
    model = create_model(MAX_QRY_LENGTH, MAX_DOC_LENGTH, NUM_OF_FEATS, PSGS_SIZE, NUM_OF_FILTERS, tau)
    model.compile(loss= 'kullback_leibler_divergence',    optimizer='Nadam',    metrics=['accuracy'])
    
    model.fit([X, X1], y, 
            batch_size=batch_size, 
            epochs=10,    
            verbose=1,    
            shuffle=True)
    
