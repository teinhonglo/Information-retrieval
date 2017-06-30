import theano
import theano.tensor as T
from theano.tensor import _shared
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

optimizer = ["Adagrad", "Nadam", "Adam"]
losses = ["k", "c"]

def calculate(pred_relevance, split_idx):
	rel_query_model = pred_relevance
	print type(rel_query_model)
	print rel_query_model.shape.eval()
	with open("query_model.pkl", "rb") as file: query_model = Pickle.load(file)[:split_idx]
	with open("query_list.pkl", "rb") as file:	query_list = Pickle.load(file)[:split_idx]

	with open("doc_model.pkl", "rb") as file: doc_model = Pickle.load(file)
	with open("doc_list.pkl", "rb") as file: doc_list = Pickle.load(file)
	#with open("relevance_model_RM.pkl", "rb") as file : rel_query_model = Pickle.load(file)
	#with open("query_relevance_model_RLE.pkl", "rb") as file : rel_query_model = Pickle.load(file)

	background_model = ProcDoc.read_background_dict()
	qry_eval = evaluate.evaluate_model(True)

	''' document smoothing '''
	for doc_idx in range(doc_model.shape[0]):
		doc_vec = doc_model[doc_idx]
		doc_model[doc_idx] = (1 - doc_lambda) * doc_vec + doc_lambda * background_model
	
	mAP_list = []
	query_rel_list = []
	query_bg_list = []	
	doc_model = np.log(doc_model)

	doc_model = doc_model
	for rel_qry_lambda in np.linspace(0, 1., num=11):
		''' query smoothing '''	
		with open("query_model.pkl", "rb") as file: query_model = Pickle.load(file)[:split_idx]
		X = T.matrix()
		Y = (1- rel_qry_lambda)*X + rel_qry_lambda * rel_query_model 
		f = theano.function([X], Y)
		query_model = f(query_model)
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
			for doc_`idx in query_result:
				docs_ranking.append(doc_list[doc_idx])
				query_docs_ranking[query_key] = docs_ranking
			
		mAP = eval.mean_average_precision(query_docs_ranking)	
		print mAP, qry_lambda, rel_qry_lambda
		'''
		mAP = qry_eval.mean_average_precision(query_docs_ranking)	
		mAP_list.append(mAP)
	return max(mAP_list)

if __name__ == "__main__":
	with open("relevance_model_RM.pkl", "rb") as file : rel_query_model = Pickle.load(file)[:720]
	theano_rel_query_model = _shared(rel_query_model)
	calculate(theano_rel_query_model, 720)
