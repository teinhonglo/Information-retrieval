# -*- coding: utf-8 -*-
import codecs
import io
import os
import fileinput
import collections
from math import exp
import codecs

def extQueryModel(query_model, rank_list, doc_model, feedback_model, top_N):
	'''
	query_model: {query_name: query word count}
	rank_list: {doc_name, doc_rank_value}
	doc_model: {doc_name, doc_word_count}
	'''
	for query, query_wc in query_model.items():
		for doc_name, doc_rank_value in rank_list[query][0:top_N]:
			if not doc_name in feedback_model:
				feedback_model.append(doc_name)
				for word, count in doc_model[doc_name].items():
					if word in query_model[query]:
						query_model[query][word] = query_model[query][word] + count
					else:
						query_model[query][word] = count
	return [query_model, feedback_model]

def outputRank(QueryName, DocRanking, isWrite):
	operation = "w"
	if isWrite == True:
		operation = "w"
	else:
		operation = "a"
	with codecs.open("FeedBack_Ranking.txt", operation, "utf-8") as outfile:
		outfile.write(QueryName + "\n")	
		out_str = ""
		for docname, score in DocRanking:
			out_str += docname + " " + str(score) + "\n"
		outfile.write(out_str)	
	
	