# -*- coding: utf-8 -*-
import codecs
import io
import os
import fileinput
import collections
from math import exp
import codecs
import copy

def extQueryModel(query_model, rank_list, doc_model, feedback_model, top_N):
	'''
	query_model: {query_name: query word count}
	rank_list: {doc_name, doc_rank_value}
	doc_model: {doc_name, doc_word_count}
	'''
	
	for query, query_wc in query_model.items():
		ex_query_wc = copy.deepcopy(query_wc)
		for doc_name in rank_list[query][0:top_N]:
			if not doc_name in feedback_model:
				feedback_model.append(doc_name)
				for word, count in doc_model[doc_name].items():
					if word in query_model[query]:
						ex_query_wc[word] = query_model[query][word] + count
					else:
						ex_query_wc[word] = count
		query_model[query] = ex_query_wc
	return [query_model, feedback_model]

