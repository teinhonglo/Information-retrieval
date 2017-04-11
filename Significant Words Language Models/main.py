import os
import fileinput
import collections
from math import log
import operator
import numpy as np
import readAssessment
import ProcDoc

data = {}				# content of document (doc, content)
background_model = {}	# word count of 2265 document (word, number of words)
query = {}				# query
query_lambda = 0.4
doc_lambda = 0.8

document_path = "../Corpus/SPLIT_DOC_WDID_NEW"
query_path = "../Corpus/QUERY_WDID_NEW_middle"

# preprocess
data = ProcDoc.read_file(document_path)
doc_wordcount = ProcDoc.doc_preprocess(data)
doc_unigram = ProcDoc.unigram(doc_wordcount)

# count background_model
'''
for key, value in data.items():
	background_model = ProcDoc.word_count(value, background_model)
print len(background_model)
background_word_sum = ProcDoc.word_sum(background_model)
'''
background_model = ProcDoc.read_background_dict()

# preprocess
query = ProcDoc.read_file(query_path)
query = ProcDoc.query_preprocess(query)
query_wordcount = {}

for q, q_content in query.items():
	query_wordcount[q] = ProcDoc.word_count(q_content, {})

query_unigram = ProcDoc.unigram(query_wordcount)
query_model = ProcDoc.modeling(query_unigram, background_model, query_lambda)

# query process
assessment = readAssessment.get_assessment()

docs_point = {}
AP = 0
mAP = 0
for q_key, q_word_prob in query_model.items():
	for doc_key, doc_words_prob in doc_unigram.items():
		point = 0
		# calculate each query value for the document
		for query_word, query_prob in q_word_prob.items():
			count = 0						# C(w , D)
			word_probability = 0			# P(w | D)
			# check if word at query exists in the document
			if query_word in doc_words_prob:
				word_probability = doc_words_prob[query_word]
			
			# (query model) * log(doc_model) 			
			point += query_model[q_key][query_word] * log((1-doc_lambda) * word_probability + doc_lambda * background_model[query_word])
		docs_point[doc_key] = point
		# sorted each doc of query by point
		docs_point_list = sorted(docs_point.items(), key=operator.itemgetter(1), reverse = True)
	AP += readAssessment.precision(docs_point_list, assessment[q_key])
# mean average precision	
mAP = AP / len(query)
print mAP