import numpy as np
np.random.seed(5566)
from keras import backend as K
from keras.layers import Activation, Input, LSTM, RepeatVector
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalMaxPooling2D
from keras.layers.merge import concatenate, dot
from keras.models import Model
from keras import backend as K

class CharacterTable(object):
	"""Given a set of characters:
	+ Encode them to a one hot integer representation
	+ Decode the one hot integer representation to their character output
	+ Decode a vector of probabilities to their character output
	"""
	def __init__(self, chars):
		"""Initialize character table.
		# Arguments
			chars: Characters that can appear in the input.
		"""
		self.chars = sorted(set(chars))
		self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
		self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

	def encode(self, C, num_rows):
		"""One hot encode given string C.
		# Arguments
			num_rows: Number of rows in the returned one hot encoding. This is
				used to keep the # of rows for each data the same.
		"""
		x = np.zeros((num_rows, len(self.chars)))
		for i, c in enumerate(C):
			x[i, self.char_indices[c]] = 1
		return x

	def decode(self, x, calc_argmax=True):
		if calc_argmax:
			x = x.argmax(axis=-1)
		return ''.join(self.indices_char[x] for x in x)

def create_model(MAX_INPUT_LENGTH = 50, INPUT_DIM = 1, LSTM_SIZE = 100, BOTTLENECK_FEATURE= 100):
	input_tensor = Input(shape=(MAX_INPUT_LENGTH, INPUT_DIM), name="input_layer")
	encoded = LSTM(LSTM_SIZE)(input_tensor)
	# base model
	decoded = RepeatVector(MAX_INPUT_LENGTH)(encoded)
	decoded = LSTM(INPUT_DIM, return_sequences=True)(decoded)
	sequence_autoencoder = Model(inputs=input_tensor, output=decoded)
	model.summary()
	'''
	from keras.utils import plot_model
	plot_model(model, to_file='model.png')
	'''
	return model

if __name__ == "__main__":
	MAX_INPUT_LENGTH = 1794
	INPUT_DIM = 1
	LSTM_SIZE = 100
	BOTTLENECK_FEATURE= 100
	model = create_model(MAX_INPUT_LENGTH, INPUT_DIM, LSTM_SIZE, BOTTLENECK_FEATURE)
	model.fit([X, ], y, 
			batch_size=batch_size, 
			epochs=10,	
			verbose=1,	
			shuffle=True)
	
	
