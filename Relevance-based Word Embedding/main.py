import operator
import numpy as np
import ProcDoc
from collections import defaultdict
from math import log
import cPickle as Pickle
import os

data = {}				# content of document (doc, content)
background_model = {}	# word count of 2265 document (word, number of words)
general_model = {}
query = {}				# query
vocabulary = np.zeros(51253)

document_path = "../Corpus/SPLIT_DOC_WDID_NEW"
query_path = "../Corpus/Train/XinTrainQryTDT2/QUERY_WDID_NEW"

# document model
data = ProcDoc.read_file(document_path)
doc_wordcount = ProcDoc.doc_preprocess(data)

# HMMTraingSet
HMMTraingSetDict = ProcDoc.read_relevance_dict()
query_relevance = {}


query = ProcDoc.read_file(query_path)
query = ProcDoc.query_preprocess(query)
query_wordcount = {}

for q, q_content in query.items():
	query_wordcount[q] = ProcDoc.word_count(q_content, {})

query_unigram = ProcDoc.unigram(query_wordcount)


# query model
query_model = []
q_list = query_unigram.keys()
for q, w_uni in query_unigram.items():
	if q in HMMTraingSetDict:
		vocabulary = np.zeros(51253)
		for w, uni in w_uni.items():
			vocabulary[int(w)] = uni
		query_model.append(np.copy(vocabulary))
	else:
		q_list.remove(q)
query_model = np.array(query_model)


# relevance model
query_relevance = []
for q in q_list:
	vocabulary = np.zeros(51253)
	for doc_name in HMMTraingSetDict[q]:
		for word, count in doc_wordcount[doc_name].items():
			vocabulary[int(word)] += count
	vocabulary /= vocabulary.sum(axis = 0)
	query_relevance.append(np.copy(vocabulary))
query_relevance = np.array(query_relevance)

