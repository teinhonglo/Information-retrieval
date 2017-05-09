import os
from scipy.spatial.distance import cosine
import numpy as np
from math import exp
import cPickle as Pickle
from collections import defaultdict
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

class word2vec_model():
	def __init__(self, vocabulary_length = 22738, word_vec_length = 100, alpha = 50, c = 0.7):
		self.word2vec = self.readWord2VecModel()
		self.vocabulary_length = vocabulary_length
		self.word_vec_length = word_vec_length
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
	
	def getMeanVec(self):
		return self.mean_vector

	def sumOfTotalSimiliary(self, cur_word_vec, collection):
		total_similiary = 0
		for word_sq, word_sq_vec in collection.items():
			total_similiary += self.sigmoid(1 - cosine(cur_word_vec, word_sq_vec))
		return total_similiary
	
	def getWordSimilarity(self, w1_vec, w2_vec):
		word2vec = self.word2vec
		return self.sigmoid(1 - cosine(w1_vec, w2_vec))
		
	def sigmoid(self, x):
		return 1 / (1 + exp(-self.alpha * (x - self.c)))
	
	def getWord2Vec(self):
		return self.word2vec
	

