import ProcDoc
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from math import log, sqrt
from multiprocessing import Process, Queue, freeze_support
import numpy as np
import timeit

def my_cosine_similarity(output, interative, _tfidf):
	cosine_sim = []
	for doc_tfidf in interative:
		cosine_sim.append((doc_tfidf * _tfidf).sum(axis = 1) / (np.sqrt((doc_tfidf ** 2).sum(axis = 0)) * np.sqrt((_tfidf ** 2).sum(axis = 1))))
	output.put(cosine_sim)

def main():
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
	
	_tfidf = np.array(_tfidf)
	
	
	output = Queue()
	pipeline = [_tfidf[:len(_tfidf) * 1/ 4], _tfidf[len(_tfidf) * 1/ 4:len(_tfidf) * 2/ 4], _tfidf[len(_tfidf) * 2/ 4:len(_tfidf) * 3/ 4], _tfidf[len(_tfidf) * 3/ 4:]]
		
	processes = [Process(target=my_cosine_similarity, args=(output, x_func, _tfidf)) for x_func in pipeline]
		
	for p in processes:
		p.start()
		
	result = [output.get() for p in processes]
	result.sort()
	cosine_sim = []
	
	for r in results:
		cosine_sim += r[1]
		
	cosine_sim = sparse.csr_matrix(cosine_sim)
	print cosine_sim
	return cosine_sim	
	

if __name__ == '__main__':
	start = timeit.default_timer()
	freeze_support() # Optional under circumstances described in docs
	main()
	stop = timeit.default_timer()	
	print stop - start
	'''
	for doc_tfidf in _tfidf:
		cosine_sim.append((doc_tfidf * _tfidf).sum(axis = 1) / (np.sqrt((doc_tfidf ** 2).sum(axis = 0)) * np.sqrt((_tfidf ** 2).sum(axis = 1))))
		
	print cosine_sim
	
	_tfidf = sparse.csr_matrix(_tfidf)
		
	similarities = cosine_similarity(_tfidf)
	print('pairwise sparse output:\n {}\n'.format(similarities))
	print similarities.shape	
	'''