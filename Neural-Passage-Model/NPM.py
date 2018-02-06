import numpy as np
from keras import backend as K
from keras.layers import Activation, Input
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot
from keras.models import Model

def create_model(MAX_QRY_LENGTH = None, MAX_DOC_LENGTH = None, NUM_OF_FEATURE = 10, PSG_Size = 10, NUM_OF_FILTERS = 10):
    PsgMat = Input(shape = (None, MAX_QRY_LENGTH, MAX_DOC_LENGTH))
	HeterMat = Input(shape = (None, NUM_OF_FEATURE))
	# Convolution2D, Meaning pooling and Max pooling
	# get r(q, d)
	# Conv2D
	# Mean pooling
	# Max pooling
	
	# Fusion Matrix and predict relevance
	# get h(q, d)
	# MLP(DENSE(len(r(q,d))))
	# tanh(dot(r.transpose * h))
	pred = Activation("tanh")(doc_prod)
    # We now have everything we need to define our model.
    model = Model(inputs = [PsgMat, HeterMat], outputs = pred)
    model.compile(optimizer = "adadelta", loss = "categorical_crossentropy", metrics=["accuracy"])

    model.summary()
    '''
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
    '''
    return model

if __name__ == "__main__":
    create_model()