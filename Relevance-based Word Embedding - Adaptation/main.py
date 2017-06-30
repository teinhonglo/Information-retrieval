import operator
import numpy as np
from collections import defaultdict
from math import log
import cPickle as Pickle
import copy
import evaluate
import relevance_model
import unigram_model
import ProcDoc
#import nn_Test

rel_qry_lambda = 0.5
qry_lambda = 0.1
doc_lambda = 0.5
#nn_model = nn_Test.model("RLE_RM.h5", "test_query_model.pkl")

def search(query_model, query_list, doc_model, doc_list):
	result = np.argsort(-np.dot(query_model, doc_model.T), axis = 1)
	query_docs_ranking = {}
	''' speedup '''
	for q_idx in xrange(len(query_list)):
		docs_ranking = []
		for doc_idx in result[q_idx]:
			docs_ranking.append(doc_list[doc_idx])
		query_docs_ranking[query_list[q_idx]] = docs_ranking
	return query_docs_ranking

def smoothing(ori_md, smth_md, sm_lambda, isBG):
	aft_model = copy.deepcopy(ori_md)
	for ori_idx, ori_vec in enumerate(aft_model):
		if isBG:
			aft_model[ori_idx] = (1 - sm_lambda) * ori_vec + sm_lambda * smth_md[ori_idx]
		else:
			aft_model[ori_idx] = (1 - sm_lambda) * ori_vec + sm_lambda * smth_md
			
	return aft_model

with open("test_query_model.pkl", "rb") as file: query_model = Pickle.load(file)
with open("test_query_list.pkl", "rb") as file:	query_list = Pickle.load(file)
print query_model.shape

with open("doc_model.pkl", "rb") as file: doc_model = Pickle.load(file)
with open("doc_list.pkl", "rb") as file: doc_list = Pickle.load(file)
print doc_model.shape

with open("NN_Result/SRM_S_RLE.pkl", "rb") as file : rel_query_model = Pickle.load(file)
#with open("query_relevance_model_RLE.pkl", "rb") as file : rel_query_model = Pickle.load(file)

background_model = ProcDoc.read_background_dict()
print background_model.shape

evl = evaluate.evaluate_model()

''' document smoothing 
	smoothing parameter 0.8
'''
doc_model = smoothing(doc_model, background_model, doc_lambda, False)


mAP_list = []
doc_model = np.log(doc_model)

''' query smoothing
	smoothing parameter 0.7
'''	
query_model = smoothing(query_model, rel_query_model, rel_qry_lambda, True)

# one-shot retrieval	
query_docs_ranking = search(query_model, query_list, doc_model, doc_list)
with open("t1.pkl", "wb") as f : Pickle.dump(query_model, f, True)
with open("t2.pkl", "wb") as f : Pickle.dump(doc_model, f, True)

mAP = evl.mean_average_precision(query_docs_ranking)	
print mAP

''' feedback 
	best: 1 document
'''

rm_feedback = relevance_model.RM_FB(query_model)
rm_feedback_list = []
fb_docs = 1
for step in np.linspace(1, 15., num=15):
	rm_feedback_list.append(rm_feedback.PRF(query_docs_ranking, int(step)))

with open("test_query_model.pkl", "rb") as file: query_model = Pickle.load(file)
fb_query_model = copy.deepcopy(query_model)

alpha = 0.25
beta = 0.25
gamma = 0.25
delta = 0.25

for step, fb_relevance_model in enumerate(rm_feedback_list):
	print step + 1
	mAP_list = []
	# smoothing parameter 0.7
	for fb_lambda in [np.linspace(0, 1., num=11)[7]]:
		''' query smoothing '''	
		for q_idx, q_vec in enumerate(query_model):
#			fb_query_model[q_idx] = alpha * query_model[q_idx] + beta * background_model + gamma * rel_query_model[q_idx] + delta * fb_relevance_model[q_idx] 
		''' mean average precision '''
		query_docs_ranking = search(fb_query_model, query_list, doc_model, doc_list)
		mAP = evl.mean_average_precision(query_docs_ranking)	
		mAP_list.append(mAP)
		print fb_lambda, mAP
	print max(mAP_list)
'''
nn_model.train(rm_feedback_list[0], 10)
with open("test_query_model.pkl", "rb") as file: query_model = Pickle.load(file)

rel_query_model = nn_model.predict(query_model)

query_model = smoothing(query_model, rel_query_model, rel_qry_lambda, True)

# retrieval	
query_docs_ranking = search(query_model, query_list, doc_model, doc_list)
mAP = evl.mean_average_precision(query_docs_ranking)	
print mAP	
'''
