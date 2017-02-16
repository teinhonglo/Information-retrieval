import ProcDoc
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from math import log, sqrt
import numpy as np
import timeit

start = timeit.default_timer()

documents = ProcDoc.read_doc()
texts = [[word for word in document.lower().split()] for document in documents]
total_docs = len(texts) * 1.0

term_freq = []
doc_freq = {}
for text in texts:
	cur_term_freq = {}
	for token in text:
		if token in cur_term_freq:
			cur_term_freq[token] += 1
		else:	
			cur_term_freq[token] = 1
			if token in doc_freq:
				doc_freq[token] += 1
			else:
				doc_freq[token] = 1
	term_freq.append(cur_term_freq)

tfidf = []	
for doc_tf in term_freq:
	doc_tfidf = {}
	for term, tf in doc_tf.items():
		idf = log(1 + total_docs / doc_freq[term])
		doc_tfidf[term] = tf / idf	
	tfidf.append(doc_tfidf)	
	
_tfidf = []
for doc_tfidf in tfidf:
	vector = []
	for token in doc_freq.keys():
		if token in doc_tfidf:
			vector.append(doc_tfidf[token])
		else:	
			vector.append(0)
	_tfidf.append(vector)

'''
_tfidf = my_cosine_similarity.run(_tfidf)
print _tfidf
'''
_tfidf = sparse.csr_matrix(_tfidf)
	
similarities = cosine_similarity(_tfidf)
print('pairwise sparse output:\n {}\n'.format(similarities))
print similarities.shape
threshold = np.array(int(total_docs) * [0])
weight = np.array(int(total_docs) * [1])
print ((np.array(similarities) > threshold) * weight).sum(axis = 0)

stop = timeit.default_timer()
print stop - start
