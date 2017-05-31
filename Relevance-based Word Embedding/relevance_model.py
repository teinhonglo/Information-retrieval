'''
	query_docs_ranking 	{q_key:[d_key...], ...}
	query_list  		[q_key, ....]
	query_model 		[[query_unigram], ....]
	doc_list			[d_key, .....]
	doc_model			[[doc_unigram]]
'''

import cPickle as pickle
import numpy as np



query_list = pickle.load(open("query_list.pkl", "rb"))
query_model = pickle.load(open("query_model.pkl", "rb"))
doc_list = pickle.load(open("doc_list.pkl", "rb"))
doc_model = pickle.load(open("doc_model.pkl", "rb"))

''' relevace model '''
for q_idx, q_vec in enumerate(query_model):
	q_key = query_list[q_idx]
	