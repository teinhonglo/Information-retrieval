import numpy as np
np.random.seed(5566)
from keras import backend as K
from keras.layers import Activation, Input, LSTM, RepeatVector
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalMaxPooling2D
from keras.layers.merge import concatenate, dot
from keras import backend as K

def ctc_lambda_func(y_pred, labels, input_length, label_length):
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def create_model(MAX_INPUT_LENGTH = 50, INPUT_DIM = 1, LSTM_SIZE = 100, BOTTLENECK_FEATURE= 100):
	inputs = Input(shape=(MAX_INPUT_LENGTH, INPUT_DIM), name="input_layer")
	encoded = LSTM(LSTM_SIZE)(inputs)

	decoded = RepeatVector(MAX_INPUT_LENGTH)(encoded)
	decoded = LSTM(INPUT_DIM, return_sequences=True)(decoded)

	sequence_autoencoder = Model(inputs, decoded)
	#encoder = Model(inputs, encoded)
	labels = Input(name='the_labels', shape=[MAX_INPUT_LENGTH], dtype='float32')
	input_length = Input(name='input_length', shape=[1], dtype='int64')
	label_length = Input(name='label_length', shape=[1], dtype='int64')
	loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([decoded, labels, input_length, label_length])
	sequence_autoencoder.summary()
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
	model = create_model(MAX_QRY_LENGTH)
	model.compile(loss= 'kullback_leibler_divergence',	optimizer='Nadam',	metrics=['accuracy'])
	
	model.fit([X, X1], y, 
			batch_size=batch_size, 
			epochs=10,	
			verbose=1,	
			shuffle=True)
	
