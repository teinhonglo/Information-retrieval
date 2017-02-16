import ProcDoc
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from math import log, sqrt
import numpy as np

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
_tfidf = np.array(_tfidf)
cosine_sim = []
for doc_tfidf in _tfidf:
	cosine_sim.append((doc_tfidf * _tfidf).sum(axis = 1) / (np.sqrt((doc_tfidf ** 2).sum(axis = 0)) * np.sqrt((_tfidf ** 2).sum(axis = 1))))
	
print cosine_sim
'''
_tfidf = sparse.csr_matrix(_tfidf)
	
similarities = cosine_similarity(_tfidf)
print('pairwise sparse output:\n {}\n'.format(similarities))
print similarities.shape