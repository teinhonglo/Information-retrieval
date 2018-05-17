'''	type	
	query_docs_ranking 		dict		{q_key:[d_key...], ...}
	query_list  			list		[q_key, ....]
	query_model 			numpy		[[query_unigram], ....]
	doc_list				list		[d_key, .....]
	doc_model				numpy		[[doc_unigram]]
	HMMTraingSetDict		
'''

import cPickle as pickle
import numpy as np
import ProcDoc
from math import exp
import copy

topM = 9


''' load data '''
'''
with open("query_list.pkl", "rb") as file:
	query_list = pickle.load(file)
with open("query_model.pkl", "rb") as file:	
	query_model = pickle.load(file)
with open("query_ranking_list.pkl", "rb") as file:
	query_docs_ranking = pickle.load(file)	
'''	
class RM_FB:
	def __init__(self, query_model, isSpoken = False):
		smoothing = 0.0
		with open("test_query_list.pkl", "rb") as file:
			self.query_list = pickle.load(file)
		with open("doc_list.pkl", "rb") as file:
			self.doc_list = pickle.load(file)
		if isSpoken:	
			with open("doc_model_wc_s.pkl", "rb") as file:doc_model = pickle.load(file)
		else:
			with open("doc_model_wc.pkl", "rb") as file: doc_model = pickle.load(file)
					
		background_model =  ProcDoc.read_background_dict()	
		self.query_model = copy.deepcopy(query_model)
		self.vocabulary_size = 51253
		self.doc_model = copy.deepcopy(doc_model)
		self.doc_model = doc_model
		self.background_model = background_model
		
	def PRF(self, query_docs_ranking, topM):
		query_model = copy.deepcopy(self.query_model)
		query_list = self.query_list
		doc_model = self.doc_model
		doc_list = self.doc_list
		background_model = self.background_model
		vocabulary_size = self.vocabulary_size
		''' relevace model '''
		for q_idx, q_key in enumerate(query_list):
			q_vec = query_model[q_idx]
			# relevance top M document
			q_t_d = np.zeros(len(query_docs_ranking[q_key][:topM]))
			w_d = np.zeros(vocabulary_size)
			for rank_idx, doc_key in enumerate(query_docs_ranking[q_key][:topM]):
				doc_idx = doc_list.index(doc_key)
				doc_vec = doc_model[doc_idx]
				w_d += doc_vec
			# unigram model
			w_d /= w_d.sum(axis = 0)
			query_model[q_idx] = np.copy(w_d)
		return query_model

	def smoothing(self, ori_md, smth_md, sm_lambda, isBG):
		aft_model = copy.deepcopy(ori_md)
		for ori_idx, ori_vec in enumerate(aft_model):
			if isBG:
				aft_model[ori_idx] = (1 - sm_lambda) * ori_vec + sm_lambda * smth_md[ori_idx]
			else:
				aft_model[ori_idx] = (1 - sm_lambda) * ori_vec + sm_lambda * smth_md
				
		return aft_model
