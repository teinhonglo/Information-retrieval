# -*- coding: utf-8 -*-
'''An implementation of sequence to sequence learning for autoencoder
'''
from __future__ import print_function
import numpy as np
np.random.seed(5566)
import os

from keras.models import Model
from keras.layers import Masking, Bidirectional
from keras import layers
from six.moves import range
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
				  inter_op_parallelism_threads = 1, intra_op_parallelism_threads = 1))

				  
class CharacterTable(object):
	"""Given a set of characters:
	+ Encode them to a one hot integer representation
	+ Decode the one hot integer representation to their character output
	+ Decode a vector of probabilities to their character output
	"""
	def __init__(self, chars, encode_length = 16):
		"""Initialize character table.
		# Arguments
			chars: Characters that can appear in the input.
		"""
		en_format = '{0:0'+str(encode_length)+'b}'
		self.en_length = encode_length
		self.chars = chars
		self.char_indices = dict((str(c), en_format.format(i)) for i, c in enumerate(self.chars))
		self.indices_char = dict((en_format.format(i), str(c)) for i, c in enumerate(self.chars))

	def encode(self, C, num_rows):
		"""One hot encode given string C.
		# Arguments
			num_rows: Number of rows in the returned binary encoding. 
			This is used to keep the # of rows for each data the same.
		"""
		en_length = self.en_length
		x = np.zeros((num_rows, en_length))
		for i, c in enumerate(C):
			# lookup table
			bin_encode = self.char_indices[c]
			# convert binary string to numpy
			bin_encode = [b for b in bin_encode]
			bin_encode = ','.join(bin_encode)
			b_e = np.fromstring(bin_encode, sep=",")
			x[i] = b_e
		return x

	def decode(self, x):
		q = ""
		# mapping posteria to class
		x = (x > 0.5) * 1
		for a in x:
			# convert numpy to binary string.
			b = np.copy(a).tolist()
			b = ''.join(str(b1) for b1 in b)
			# lookup table
			if b in self.indices_char:
				q += str(self.indices_char[b]) + " "
			else:	
				q += "O "
		return q


class colors:
	ok = '\033[92m'
	fail = '\033[91m'
	close = '\033[0m'

# Parameters for the model and dataset.
TRAINING_SIZE = 800
INVERT = True

# Maximum length of input.
MAXLEN = 7

# All the numbers, plus sign and space for padding.
encode_length = 16
chars = list(range(51253 + 1))
ctable = CharacterTable(chars, encode_length)

questions = []
expected = []
seen = set()
print('Generating data...')
while len(questions) < TRAINING_SIZE:
	a = ''
	for i in range(np.random.randint(1, MAXLEN)):
		a += str(np.random.choice(chars)) + ' '
	
	# Skip any addition questions we've already seen
	key = a   
	
	if key in seen:
		continue
	
	seen.add(key)
	# Pad the data with spaces such that it is always MAXLEN.
	q = a.split()
	for x in xrange(MAXLEN - len(q)):
		q.insert(0, '0')
	questions.append(q)
	expected.append(q)
	print(str(len(questions)) + '/' + str(TRAINING_SIZE), end = '\r')
print('Total addition questions:', len(questions))

print('Vectorization...')

x = np.zeros((len(questions), MAXLEN, encode_length), dtype=np.bool)
y = np.zeros((len(expected), MAXLEN, encode_length), dtype=np.bool)
for i, sentence in enumerate(questions):
	x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
	y[i] = ctable.encode(sentence, MAXLEN)

# Shuffle (x, y) in unison as the later parts of x will almost all be larger
# digits.
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over.
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

# Try replacing GRU, or SimpleRNN.
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 50
LAYERS = 1

print('Build model...')
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).
input_tensor = layers.Input(shape=(MAXLEN, encode_length))
hid_layer = RNN(HIDDEN_SIZE, return_sequences=True, activation='linear')(input_tensor)
#hid_layer = layers.Dense(HIDDEN_SIZE)(hid_layer)
#hid_layer = layers.RepeatVector(MAXLEN)(hid_layer)

# As the decoder RNN's input, repeatedly provide with the last hidden state of
# RNN for each time step.
#model.add(layers.RepeatVector(MAXLEN))
# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(LAYERS):
	# By setting return_sequences to True, return not only the last output but
	# all the outputs so far in the form of (num_samples, timesteps,
	# output_dim). This is necessary as TimeDistributed in the below expects
	# the first dimension to be the timesteps.
	hid_layer = RNN(HIDDEN_SIZE, activation='sigmoid', return_sequences=True)(hid_layer)

# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
linear_mapping = layers.TimeDistributed(layers.Dense(encode_length))(hid_layer)
pred = layers.Activation('sigmoid')(linear_mapping)

model = Model(inputs=input_tensor, outputs=pred)
model.compile(loss='binary_crossentropy',
			  optimizer='Adam',
			  metrics=['accuracy'])
model.summary()
from keras.utils import plot_model
plot_model(model, to_file='model.png')

# Train the model each generation and show predictions against the validation
# dataset.
with tf.device('/gpu:0'):
	for iteration in range(1, 200):
		print()
		print('-' * 50)
		print('Iteration', iteration)
		model.fit(x_train, y_train,
				  batch_size=BATCH_SIZE,
				  epochs=1,
				  validation_data=(x_val, y_val))
		# Select 10 samples from the validation set at random so we can visualize
		# errors.
		for i in range(10):
			ind = np.random.randint(0, len(x_val))
			rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
			preds = model.predict(rowx, verbose=0)
			q = ctable.decode(rowx[0])
			correct = ctable.decode(rowy[0])
			guess = ctable.decode(preds[0])
			print('Q', q, end=' ')
			print('T', correct, end=' ')
			if correct == guess:
				print(colors.ok + '☑' + colors.close, end=' ')
			else:
				print(colors.fail + '☒' + colors.close, end=' ')
			print(guess)
