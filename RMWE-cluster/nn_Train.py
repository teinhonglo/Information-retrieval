#-*- coding: utf-8 -*-
''' Import theano and numpy '''
import numpy as np
np.random.seed(1331)
import theano
''' Import keras to build a DL model '''
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

def create_model:
	def __init__(self, batch_size = 16, epochs = 55, vocabulary_size = 51253, embedding_dimensionality = 350)
		''' set the size of mini-batch and number of epochs'''
		self.batch_size = batch_size
		self.epochs = epochs
		self.vocabulary_size = vocabulary_size
		self.embedding_dimensionality = embedding_dimensionality
		self.model = based_model()
		
	def based_model(self):
		print 'Building a model whose optimizer=adam, activation function=softmax'
		vocabulary_size = self.vocabulary_size
		embedding_dimensionality = self.embedding_dimensionality
		
		model = Sequential()
		model.add(Dense(embedding_dimensionality, input_dim = vocabulary_size))
		model.add(Dense(vocabulary_size))
		model.add(Activation('softmax'))
		from keras.optimizers import Adam, SGD
		''' Setting optimizer as Adam '''
		model.compile(loss= 'kullback_leibler_divergence',
						optimizer='Adam',
						metrics=['accuracy'])
		model.summary()
		return model

	def train(self, X_train, Y_train):
		X_train = X_train
		Y_train = Y_train
		batch_size = self.batch_size
		epochs = self.epochs
		''' Fit models and use validation_split=0.1 '''
		history_adam = model.fit(X_train, Y_train,
								batch_size=batch_size,
								epochs=epochs,
								verbose=1,
								shuffle=True,
								validation_split=0.1)
	def save(self, Corpus, obj_func)							
		''' Create a HDF5 file '''							
		model.save("NN_Model/"+Corpus+"/RLE_" + obj_func + ".h5")