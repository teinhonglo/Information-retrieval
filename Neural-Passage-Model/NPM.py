import numpy as np
np.random.seed(5566)
from keras import backend as K
from keras.layers import Activation, Input, ZeroPadding2D
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalMaxPooling2D
from keras.layers.merge import concatenate, dot
from keras.models import Model

def create_model(MAX_QRY_LENGTH = 50, MAX_DOC_LENGTH = 2900, NUM_OF_FEATURE = 10, PSG_SIZE = 50, NUM_OF_FILTERS = 5, tau = 1):
	alpha_size = NUM_OF_FILTERS
	psgMat = Input(shape = (MAX_QRY_LENGTH, MAX_DOC_LENGTH, 1,), name="passage")
	heterMat = Input(shape = (NUM_OF_FEATURE, ), name="h_feats")
	# Convolution2D, Meaning pooling and Max pooling.
	# Conv2D, Mean pooling, Max pooling
    #psgMat_ZP  = ZeroPadding2D()(psgMat)
	M = Convolution2D(filters=NUM_OF_FILTERS, kernel_size=(1, PSG_SIZE), strides=tau, padding='valid', name="pConv2D")(psgMat)
	K = AveragePooling2D(pool_size=(MAX_QRY_LENGTH, 1), strides=1, name="pAvePool")(M)
	r = GlobalMaxPooling2D(name="pMaxPool")(K)
	# Fusion Matrix and predict relevance
	# get h(q, d)
	# MLP(DENSE(len(r(q,d))))
	phi_h = Dense(alpha_size, activation="softmax", name="TrainMat")(heterMat)
	dot_prod = dot([r, phi_h], axes = 1, name="rel_dot")
	# tanh(dot(r.transpose * h))
	pred = Activation("tanh", name="activation_tanh")(dot_prod)
	
	# We now have everything we need to define our model.
	model = Model(inputs = [psgMat, heterMat], outputs = pred)
	model.summary()
	'''
	from keras.utils import plot_model
	plot_model(model, to_file='model.png')
	'''
	return model

if __name__ == "__main__":
    data = []
    model = create_model(44, 300, 15, 50)
