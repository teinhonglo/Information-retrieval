import numpy as np
from keras import backend as K
from keras.layers import Activation, Input
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot
from keras.models import Model

def create_model(MAX_QRY_LENGTH = None, MAX_DOC_LENGTH = None, NUM_OF_FEATURE = 10, PSG_SIZE = 10, NUM_OF_FILTERS = 10):
	pool_size = (MAX_QRY_LENGTH * MAX_DOC_LENGTH) / PSG_SIZE
	alpha_size = NUM_OF_FILTERS
    psgMat = Input(shape = (None, MAX_QRY_LENGTH, MAX_DOC_LENGTH))
	heterMat = Input(shape = (None, NUM_OF_FEATURE))
	# Convolution2D, Meaning pooling and Max pooling.
	# Conv2D, Mean pooling, Max pooling
	M = Conv2D(filters=NUM_OF_FILTERS, kernel_size=PSG_SIZE, strides=(1, 1), padding='valid')(psgMat)
	K = AveragePooling1D(pool_size=pool_size, strides=1, padding='valid')(M)
	r = GlobalMaxPooling1D(pool_size=pool_size, strides=None, padding='valid')(K)
	
	# Fusion Matrix and predict relevance
	# get h(q, d)
	# MLP(DENSE(len(r(q,d))))
	phi_h = Dense(alpha_size, activation="softmax")(heterMat)
	dot_prod = dot([K.transpose(r), phi_h], axes = 1, name="rel_dot")
	# tanh(dot(r.transpose * h))
	pred = Activation("tanh")(dot_prod)
    # We now have everything we need to define our model.
    model = Model(inputs = [psgMat, heterMat], outputs = pred)
    model.compile(optimizer = "adadelta", loss = "categorical_crossentropy", metrics=["accuracy"])

    model.summary()
    '''
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
    '''
    return model

if __name__ == "__main__":
    create_model()