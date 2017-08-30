import operator
import numpy as np
import ProcDoc
from collections import defaultdict
from math import log
import cPickle as Pickle
import evaluate

rel_qry_lambda = 0.1
qry_lambda = 0.1
doc_lambda = 0.5

optimizer = ["Adagrad", "Nadam", "Adam"]
losses = ["k", "c"]


with open("test_query_model.pkl", "rb") as file: query_model = Pickle.load(file)
with open("test_query_list.pkl", "rb") as file:	query_list = Pickle.load(file)
print query_model.shape

with open("doc_model.pkl", "rb") as file: doc_model = Pickle.load(file)
with open("doc_list.pkl", "rb") as file: doc_list = Pickle.load(file)
print doc_model.shape

#with open("RM_S_RLE.pkl", "rb") as file : rel_query_model = Pickle.load(file)
with open("NN_result/SSWLM_S_RLE.pkl", "rb") as file : rel_query_model = Pickle.load(file)
print rel_query_model.shape

background_model = ProcDoc.read_background_dict()
print background_model.shape

evl = evaluate.evaluate_model(False)

''' document smoothing '''
for doc_idx in xrange(doc_model.shape[0]):
	doc_vec = doc_model[doc_idx]
	doc_model[doc_idx] = (1 - doc_lambda) * doc_vec + doc_lambda * background_model

mAP_list = []
query_rel_list = []
query_bg_list = []	
doc_model = np.log(doc_model)


for rel_qry_lambda in np.linspace(0, 1., num=11):
	''' query smoothing '''	
	with open("test_query_model.pkl", "rb") as file: query_model = Pickle.load(file)
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
	mAP = evl.mean_average_precision(query_docs_ranking)	
	print mAP
with open('eswlm_ranking_result.pkl', "wb") as file :	Pickle.dump(query_docs_ranking, file, True)

