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

query_model = Pickle.load(open("test_query_model.pkl", "rb"))
rel_query_model = Pickle.load(open("query_relevance_model_RLE.pkl", "rb"))
query_list = Pickle.load(open("test_query_list.pkl", "rb"))
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
query_rel_list = []
query_bg_list = []	
doc_model = np.log(doc_model)	

''' query smoothing '''	
for qry_idx in range(query_model.shape[0]):
	qry_vec = query_model[qry_idx]
	query_model[qry_idx] = (1 - rel_qry_lambda) * qry_vec + rel_qry_lambda * rel_query_model[qry_idx]
	#query_model[qry_idx] = (1 - qry_lambda) * qry_vec + qry_lambda * background_model
			
''' query '''	
query_docs_ranking = {}
for query_key, query_vec in  zip(query_list, query_model):
	query_result = np.argsort(-(query_vec * doc_model).sum(axis = 1))
	docs_ranking = []
	for doc_idx in query_result:
		docs_ranking.append(doc_list[doc_idx])
		query_docs_ranking[query_key] = docs_ranking

mAP = eval.mean_average_precision(query_docs_ranking)	
print mAP, qry_lambda, rel_qry_lambda