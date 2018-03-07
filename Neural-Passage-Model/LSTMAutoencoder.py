# -*- coding: utf-8 -*-
'''An implementation of sequence to sequence learning for autoencoder
'''
from __future__ import print_function
import numpy as np
np.random.seed(5566)
import os
import sys
sys.path.append("../Tools")
import ProcDoc
import WER

from keras.models import Model
from keras.layers import Masking, Bidirectional
from keras import layers
from six.moves import range
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
#				  inter_op_parallelism_threads = 1, intra_op_parallelism_threads = 1))


class colors:
	ok = '\033[92m'
	fail = '\033[91m'
	close = '\033[0m'				  
				  
class CharacterTable(object):
	"""Given a set of characters:
	+ Encode them to a binary representation
	+ Decode the binary representation to their character output
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
		
def create_model(MAXLEN, LAYERS, ENCODE_LENGTH, HIDDEN_SIZE):
	RNN = layers.GRU
	print('Build model...')
	# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
	# Note: In a situation where your input sequences have a variable length,
	# use input_shape=(None, num_feature).
	input_tensor = layers.Input(shape=(MAXLEN, ENCODE_LENGTH))
	mask_layer = layers.Masking(mask_value=0.0)(input_tensor)
	hid_layer = RNN(HIDDEN_SIZE, activation='relu')(mask_layer)
	hid_layer = layers.Dense(HIDDEN_SIZE)(hid_layer)
	hid_layer = layers.RepeatVector(MAXLEN)(hid_layer)

	# As the decoder RNN's input, repeatedly provide with the last hidden state of
	# RNN for each time step.
	#model.add(layers.RepeatVector(MAXLEN))
	# The decoder RNN could be multiple layers stacked or a single layer.
	for _ in range(LAYERS):
		# By setting return_sequences to True
		hid_layer = RNN(HIDDEN_SIZE, activation='tanh', return_sequences=True)(hid_layer)

	# Apply a dense layer to the every temporal slice of an input. For each of step
	# of the output sequence, decide which character should be chosen.
	hid_layer = layers.Dense(HIDDEN_SIZE)(hid_layer)
	#linear_mapping = layers.TimeDistributed(layers.Dense(ENCODE_LENGTH))(hid_layer)
	linear_mapping = layers.Dense(ENCODE_LENGTH)(hid_layer)
	pred = layers.Activation('sigmoid')(linear_mapping)

	model = Model(inputs=input_tensor, outputs=pred)
	model.compile(loss='binary_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])
	model.summary()
	return model

if __name__ == "__main__":
	MAXLEN = 1794
	HIDDEN_SIZE = 32
	LAYERS = 1
	TRAINING_SIZE = 800
	EPOCHS = 200
	BATCH_SIZE = 50
	# All the numbers, plus sign and space for padding.
	VOCAB_SIZE = 51253
	corpus = "TDT2"
	ENCODE_LENGTH = len('{0:016b}'.format(VOCAB_SIZE))
	qry_path = "../Corpus/" + corpus + "/Train/XinTrainQryTDT2/QUERY_WDID_NEW"
	res_pos = True 
	str2int = True 
	# 
	chars = list(range(VOCAB_SIZE + 1))
	ctable = CharacterTable(chars, ENCODE_LENGTH)
	qry = ProcDoc.read_file(qry_path)
	qry = ProcDoc.query_preprocess(qry, res_pos, str2int)
	TRAINING_SIZE = len(qry.keys())
	questions = []
	expected = []
	count = 0
	print('Generating data...')
	for q_name, q_cont in qry.items():
		#a = ' '.join(str(np.random.choice(chars)) for i in range(np.random.randint(1, MAXLEN)))
		# Pad the data with spaces such that it is always MAXLEN.
		q = [str(e + 1) for e in q_cont]
		for x in xrange(MAXLEN - len(q)):
			q.insert(0, '0')
		#print(q)	
		questions.append(q)
		count += 1
		print(str(count) + "/" + str(TRAINING_SIZE), end='\r')
		#raw_input()
	print('Total addition questions:', len(questions))

	print('Vectorization...')
	x = np.zeros((len(questions), MAXLEN, ENCODE_LENGTH), dtype=np.bool)
	for i, sentence in enumerate(questions):
		x[i] = ctable.encode(sentence, MAXLEN)

	# Shuffle (x, y) in unison as the later parts of x will almost all be larger
	# digits.
	indices = np.arange(len(x))
	np.random.shuffle(indices)
	x = x[indices]

	# Explicitly set apart 10% for validation data that we never train over.
	split_at = len(x) - len(x) // 10
	(x_train, x_val) = x[:split_at], x[split_at:]

	print('Training Data:')
	print(x_train.shape)

	print('Validation Data:')
	print(x_val.shape)
	
	model = create_model(MAXLEN, LAYERS, ENCODE_LENGTH, HIDDEN_SIZE)
	# Train the model each generation and show predictions against the validation
	# dataset.
	with tf.device('/gpu:1'):
		for iteration in range(1, EPOCHS):
			print()
			print('-' * 50)
			print('Iteration', iteration)
			model.fit(x_train, x_train,
					  batch_size=BATCH_SIZE,
					  epochs=1,
					  validation_data=(x_val, x_val))
			# Select 10 samples from the validation set at random so we can visualize
			# errors.
			for i in range(10):
				ind = np.random.randint(0, len(x_val))
				rowx = x_val[np.array([ind])]
				preds = model.predict(rowx, verbose=0)
				correct = ctable.decode(rowx[0])
				guess = ctable.decode(preds[0])
				#print('T', correct, end=' ')
				if correct == guess:
					print(colors.ok + '☑' + colors.close, end=' ')
				else:
					print(colors.fail + '☒' + colors.close, end=' ')
				#print(WER.wer(correct.split(), guess.split()))
	