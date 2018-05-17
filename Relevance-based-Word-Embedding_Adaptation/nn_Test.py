#-*- coding: utf-8 -*-
''' Import theano and numpy '''
import theano
import numpy as np

''' Import keras to build a DL model '''
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
import cPickle as Pickle

class model:
	# Initial model
	def __init__(self, load_model_name, load_query_model_name):
		self.model = load_model(load_model_name)
		with open(load_query_model_name, "rb") as file: self.query_model = Pickle.load(file)
	# Train on batch
	def train(self, train_objective, epoch = 10):
		print 'Building a model whose optimizer=Nadam, activation function=softmax'
		model = self.model
		X = self.query_model
		Y = train_objective
		for i in xrange(epoch):
			model.train_on_batch(X, Y)

		self.model = model
	# predict value
	def predict(self, query_model):
		model = self.model
		X = query_model
		output = np.array(model.predict(X))
		return output





























