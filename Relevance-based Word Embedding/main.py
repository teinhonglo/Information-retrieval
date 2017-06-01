import operator
import numpy as np
import ProcDoc
from collections import defaultdict
from math import log
import cPickle as Pickle
import evaluate
rel_qry_lambda = 0.1
qry_lambda = 0.1
doc_lambda = 0.8

query_model = Pickle.load(open("query_model.pkl", "rb"))
query_list = Pickle.load(open("query_list.pkl", "rb"))
print query_model.shape

doc_model = Pickle.load(open("doc_model.pkl", "rb"))
doc_list = Pickle.load(open("doc_list.pkl", "rb"))
print doc_model.shape

background_model = ProcDoc.read_background_dict()
print background_model.shape

eval = evaluate.evaluate_model()

''' document smoothing '''
for doc_idx in range(doc_model.shape[0]):
	doc_vec = doc_model[doc_idx]
	doc_model[doc_idx] = (1 - doc_lambda) * doc_vec + doc_lambda * background_model

mAP_list = []
total_mAP_list = []
query_rel_list = []
query_bg_list = []	
doc_model = np.log(doc_model)
for smoothing in np.linspace(0.1, 0.9, num = 9):
	rel_query_model = Pickle.load(open("relevance_model_"+str(smoothing)+".pkl", "rb"))
	for rel_qry_lambda in np.linspace(0, 1., num=11):
		''' query smoothing '''	
		query_model = Pickle.load(open("query_model.pkl", "rb"))
		for qry_idx in range(query_model.shape[0]):
			qry_vec = query_model[qry_idx]
			query_model[qry_idx] = (1 - rel_qry_lambda) * qry_vec + rel_qry_lambda * rel_query_model[qry_idx]
			#query_model[qry_idx] = (1 - qry_lambda) * qry_vec + qry_lambda * background_model
		result = np.argsort(-np.dot(query_model, doc_model.T), axis = 1)
		query_docs_ranking = {}
		''' speedup '''
		for q_idx in range(len(query_list)):
			docs_ranking = []
			for doc_idx in result[q_idx]:
				docs_ranking.append(doc_list[doc_idx])
			query_docs_ranking[query_list[q_idx]] = docs_ranking
		
		''' query 
		for query_key, query_vec in  zip(query_list, query_model):
			print len(query_docs_ranking.keys())
			query_result = np.argsort(-(query_vec * doc_model).sum(axis = 1))
			docs_ranking = []
			for doc_idx in query_result:
				docs_ranking.append(doc_list[doc_idx])
				query_docs_ranking[query_key] = docs_ranking
			
		mAP = eval.mean_average_precision(query_docs_ranking)	
		print mAP, qry_lambda, rel_qry_lambda
		'''
		mAP = eval.mean_average_precision(query_docs_ranking)	
		mAP_list.append(mAP)
		print mAP, smoothing, rel_qry_lambda
	total_mAP_list.append(max(mAP_list))
	print 
print max(total_mAP_list), np.argmax(np.array(total_mAP_list))	
