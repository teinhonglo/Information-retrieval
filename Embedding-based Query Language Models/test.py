import word2vec_model
import cPickle as Pickle
from scipy.spatial.distance import cosine
import numpy as np
import timeit

word_model = word2vec_model.word2vec_model()
word_vec = word_model.getWord2Vec()
word = word_vec["3478"]
w_vec = word / np.sqrt((word ** 2).sum(axis = 0))
word *= -1
inv_vec = word / np.sqrt((word ** 2).sum(axis = 0))
print inv_vec
print 1-cosine(w_vec, inv_vec)
print (w_vec * inv_vec).sum(axis = 0)

print word_model.getWordSimilarity(w_vec, inv_vec)
stop = timeit.default_timer()


'''
with open("query_model_prev.pkl", "rb") as file:
	query_model = Pickle.load(file)
#print (query_model)	

query_model_aft = {}
with open("query_model_aft.pkl", "rb") as file:
	query_model_aft = Pickle.load(file)
#print (query_model_aft)

for q_key, q_wordcount in query_model.items():
	for word, count in q_wordcount.items():
		if query_model[q_key][word] != query_model_aft[q_key][word]:
			print q_key, word
			print query_model[q_key][word], query_model_aft[q_key][word]
			print "False"
			
doc_model = {}
with open("doc_unigram_prev.pkl", "rb") as file:
	doc_model = Pickle.load(file)
#print (query_model)	

doc_model_aft = {}
with open("doc_unigram_prev.pkl", "rb") as file:
	doc_model_aft = Pickle.load(file)
#print (query_model_aft)

for d_key, d_wordcount in doc_model.items():
	for word, count in d_wordcount.items():
		if doc_model[d_key][word] != doc_model_aft[d_key][word]:
			print d_key, word
			print "False"			
'''			