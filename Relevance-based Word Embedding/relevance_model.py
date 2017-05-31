'''
	query_docs_ranking 	{q_key:[d_key...], ...}
	query_list  		[q_key, ....]
	query_model 		[[query_unigram], ....]
	doc_list			[d_key, .....]
	doc_model			[[doc_unigram]]
'''

import cPickle as pickle
import numpy as np

topM = 10
vocabulary_size = 51253

''' load data'''
query_list = pickle.load(open("query_list.pkl", "rb"))
query_model = pickle.load(open("query_model.pkl", "rb"))
doc_list = pickle.load(open("doc_list.pkl", "rb"))
doc_model = pickle.load(open("doc_model.pkl", "rb"))

''' relevace model '''
for q_idx, q_key in enumerate(query_list):
	q_vec = query_model[q_idx]
	# relevance top N document
	q_t_d = np.zeros(topM)
	w_d = np.zeros(vocabulary_size)
	for rank_idx, doc_key in enumerate(query_docs_ranking[q_key][:topM]):
		doc_idx = np.where(doc_list == doc_key)
		doc_vec = doc_model[doc_idx]
		# probability of query term 
		q_non_zero, = np.where(q_vec != 0)
		q_t_d[rank_idx] += 1
		for q_t in doc_vec(q_non_zero):
			q_t_d[rank_idx] *= q_t	
		w_d += doc_vec * q_t_d[rank_idx]
	# relevance model
	w_d /= q_t_d.sum(axis = 0)
	query_model[q_idx] = w_d
	