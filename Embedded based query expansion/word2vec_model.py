import os
from sklearn.metrics.pairwise import cosine_similarity
from math import exp
import cPickle as Pickle

class word2vec_model():
	def __init__(self, vocabulary_length = 22738, alpha = 50, c = 0.7):
		self.word2vec = self.readWord2VecModel()
		self.vocabulary_length = vocabulary_length
		self.alpha = alpha
		self.c = c
		
	def readWord2VecModel(self):
		word2vec = []
		with open("../Corpus/word2vec.pickle", "rb") as file:
			word2vec = Pickle.load(file)
		word2vec = word2vec.wv
		return word2vec

	def sumOftotalSimiliary(self,cur_word, collection):
		word2vec = self.word2vec
		total_similiary = 0
		for word_sq in collection:
			total_similiary += self.sigmoid(word2vec.similarity(cur_word, word_sq))
		return total_similiary
	
	def getWordSimilarity(self, w1, w2):
		word2vec = self.word2vec
		return self.sigmoid(word2vec.similarity(w1, w2))
		
	def sigmoid(self, x):
		return 1 / (1 + exp(-self.alpha * (x - self.c)))
	

