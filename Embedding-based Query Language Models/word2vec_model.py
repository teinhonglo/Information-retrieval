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
		'''
		# avoid memory error
		word_pointer = 0
		total_similarity ={}
		
		for word, cur_word_vec in cur_set.items():
			total_similarity[word] = 0
			word_sq_vec = np.array(collection.values())
			result = (cur_word_vec * word_sq_vec).sum(axis = 0)
			for r in result:
				total_similarity[word] += self.sigmoid(r)
			word_pointer += 1
			print word_pointer
		return total_similarity		
		'''
		
		word_list = cur_set.keys()
		word_pointer = 0
		total_similarity = {}
		cur_set_val = np.array(cur_set.values())
		collection_val = np.array(collection.values())
		# cross product
		cosine_result = np.dot(cur_set_val, collection_val.T)
		
		for word_cosine_vector in cosine_result:
			current_word_similiary = 0
			for cosine_similiary in word_cosine_vector:
				current_word_similiary += self.sigmoid(cosine_similiary)
			total_similarity[word_list[word_pointer]] = current_word_similiary
			word_pointer += 1
			print word_pointer
			
		return total_similarity
		
	
	def getWordSimilarity(self, w1_vec, w2_vec):
		word2vec = self.word2vec
		return self.sigmoid((w1_vec * w2_vec).sum(axis = 0))
	
	def sigmoid(self, x):
		gamma = -self.alpha * (x - self.c)
		# overflow
		if gamma < 0:
			return 1 - 1 / (1 + exp(gamma))
		else:
			return 1 / (1 + exp(-gamma))
		
	def setAlpha(self, a):
		self.alpha = a
		
	def getWord2Vec(self):
		return self.word2vec	
	
	def getMeanVec(self):
		return self.mean_vector	
	

