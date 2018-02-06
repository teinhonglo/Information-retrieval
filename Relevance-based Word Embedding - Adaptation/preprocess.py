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

document_path = "../Corpus/TDT2/Spoken_Doc"
#document_path = "../Corpus/SPLIT_DOC_WDID_NEW"
query_path = "../Corpus/TDT2/Train/XinTrainQryTDT2/QUERY_WDID_NEW"


# read document
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


# create outside query model
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
query_model = np.array(query_model).astype(np.float32)

with open("query_model.pkl", "wb") as file:
	Pickle.dump(query_model, file, True)
with open("query_list.pkl", "wb") as file:
	Pickle.dump(q_list, file, True)

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

with open("relevance.pkl", "wb") as file:
	Pickle.dump(query_relevance, file, True)
with open("HMMTraingSetDict.pkl", "wb")	as file:
	Pickle.dump(HMMTraingSetDict, file, True)


# document model
doc_list = doc_wordcount.keys()
doc_model = []
for doc_name in doc_list:
	vocabulary = np.zeros(51253)
	for word, count in doc_wordcount[doc_name].items():
		vocabulary[int(word)] = count
	vocabulary /= vocabulary.sum(axis = 0)
	doc_model.append(np.copy(vocabulary))
doc_model = np.array(doc_model).astype(np.float32)	

with open("doc_list.pkl", "wb") as file: Pickle.dump(doc_list, file, True)
with open("doc_model.pkl", "wb") as file: Pickle.dump(doc_model, file, True)

# test query model
query_path = "../Corpus/TDT2/QUERY_WDID_NEW_middle"
test_query = ProcDoc.read_file(query_path)
test_query = ProcDoc.query_preprocess(test_query)
test_query_wordcount = {}

for q, q_content in test_query.items():
	test_query_wordcount[q] = ProcDoc.word_count(q_content, {})

test_query_unigram = ProcDoc.unigram(test_query_wordcount)
test_query_list = test_query_unigram.keys()

test_query_model = []
for q in test_query_list:
	vocabulary = np.zeros(51253)
	for word, unigram in test_query_unigram[q].items():
		vocabulary[int(word)] = unigram
	test_query_model.append(np.copy(vocabulary))
test_query_model = np.array(test_query_model).astype(np.float32)

with open("test_query_list.pkl", "wb") as file:
	Pickle.dump(test_query_list, file, True)
with open("test_query_model.pkl", "wb") as file:
	Pickle.dump(test_query_model, file, True)

