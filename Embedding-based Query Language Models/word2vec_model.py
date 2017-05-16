import os
from scipy.spatial.distance import cosine
import numpy as np
from math import exp
import cPickle as Pickle
from collections import defaultdict
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

class word2vec_model():
	def __init__(self, alpha = 50, c = 0.7):
		self.word2vec = self.readWord2VecModel()
		self.vocabulary_length = len(self.word2vec.vocab)
		first_word = self.word2vec.vocab.keys()[0]
		self.word_vec_length = len(self.word2vec[first_word])
		self.alpha = alpha
		self.c = c
		self.mean_vector = self.calcMeanVec()
		
	def readWord2VecModel(self):
		word2vec = []
		with open("../Corpus/word2vec.pickle", "rb") as file:
			word2vec = Pickle.load(file)
		word2vec = word2vec.wv
		return word2vec
	
	def calcMeanVec(self):
		w2v = self.word2vec
		w2v_vocab = w2v.vocab
		mean_vector = np.zeros(self.word_vec_length)
		length = self.vocabulary_length
		for word in w2v_vocab:
			mean_vector += w2v[word]
		mean_vector /= self.vocabulary_length
		return mean_vector

	def sumOfTotalSimilarity(self, cur_set, collection):
		# avoid memory error
		total_similarity ={}
		
		for word, cur_word_vec in cur_set.items():
			total_similarity[word] = 0
			word_sq_vec = np.array(collection.values())
			cosine_vectors = (cur_word_vec * word_sq_vec).sum(axis = 1)
			for cosine_result in cosine_vectors:
				total_similarity[word] += self.sigmoid(cosine_result)
		return total_similarity		
	
	def getWordSimilarity(self, w1_vec, w2_vec):
		cosine_result = (w1_vec* w2_vec).sum(axis = 0)
		return self.sigmoid(cosine_result)
	
	def sigmoid(self, x):
		gamma = -self.alpha * (x - self.c)
		return 1 / (1 + exp(gamma))
		
	def setAlpha(self, a):
		self.alpha = a
		
	def getWord2Vec(self):
		return self.word2vec	
	
	def getMeanVec(self):
		return self.mean_vector	
	

