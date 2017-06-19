import operator
import numpy as np
import ProcDoc
from collections import defaultdict
from math import log
import cPickle as Pickle
import evaluate
import relevance_model

rel_qry_lambda = 0.7
qry_lambda = 0.1
doc_lambda = 0.8

def search(query_model, query_list, doc_model, doc_list):
	result = np.argsort(-np.dot(query_model, doc_model.T), axis = 1)
	query_docs_ranking = {}
	''' speedup '''
	for q_idx in range(len(query_list)):
		docs_ranking = []
		for doc_idx in result[q_idx]:
			docs_ranking.append(doc_list[doc_idx])
		query_docs_ranking[query_list[q_idx]] = docs_ranking
	return query_docs_ranking


with open("test_query_model.pkl", "rb") as file: query_model = Pickle.load(file)
with open("test_query_list.pkl", "rb") as file:	query_list = Pickle.load(file)
print query_model.shape

with open("doc_model.pkl", "rb") as file: doc_model = Pickle.load(file)
with open("doc_list.pkl", "rb") as file: doc_list = Pickle.load(file)
print doc_model.shape

with open("SRM_RLE.pkl", "rb") as file : rel_query_model = Pickle.load(file)
#with open("query_relevance_model_RLE.pkl", "rb") as file : rel_query_model = Pickle.load(file)

background_model = ProcDoc.read_background_dict()
print background_model.shape

eval = evaluate.evaluate_model()

''' document smoothing '''
for doc_idx in range(doc_model.shape[0]):
	doc_vec = doc_model[doc_idx]
	doc_model[doc_idx] = (1 - doc_lambda) * doc_vec + doc_lambda * background_model

mAP_list = []
doc_model = np.log(doc_model)


''' query smoothing '''	
for qry_idx in range(query_model.shape[0]):
	qry_vec = query_model[qry_idx]
	query_model[qry_idx] = (1 - rel_qry_lambda) * qry_vec + rel_qry_lambda * rel_query_model[qry_idx]

rm_feedback = relevance_model.RM_FB(query_model)

query_docs_ranking = search(query_model, query_list, doc_model, doc_list)
mAP = eval.mean_average_precision(query_docs_ranking)	
print mAP	

''' feedback '''
rm_feedback_list = []
for step in np.linspace(1, 15., num=15):
	rm_feedback_list.append(rm_feedback.PRF(query_docs_ranking, int(step)))

fb_query_model = list(query_model)	
for step, relevance_model in enumerate(rm_feedback_list):
	print step
	for fb_lambda in np.linspace(0, 1., num=11):
		''' query smoothing '''	
		for qry_idx in range(query_model.shape[0]):
			qry_vec = query_model[qry_idx]
			fb_query_model[qry_idx] = (1 - fb_lambda) * qry_vec + fb_lambda * rel_query_model[qry_idx]
		query_docs_ranking = search(fb_query_model, query_list, doc_model, doc_list)
		mAP = eval.mean_average_precision(query_docs_ranking)	
		print fb_lambda, mAP

#with open("query_ranking_list.pkl", "wb") as file: Pickle.dump(query_docs_ranking, file, True)	