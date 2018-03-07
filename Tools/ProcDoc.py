# -*- coding: utf-8 -*-
import codecs
import io
import os
import fileinput
import collections
import numpy as np
import operator
import types
from math import exp
from webbrowser import BackgroundBrowser
from collections import defaultdict
from math import log

bg_modle_path = "../Information-retrieval/Corpus/background"
Cluster_path = "Topic"


# read file(query or document)
def read_file(filepath):
	data = {}				# content of document (doc, content)
	# list all files of a directory(Document)
	for dir_item in os.listdir(filepath):
		# join dir path and file name
		dir_item_path = os.path.join(filepath, dir_item)
		# check whether a file exists before read
		if os.path.isfile(dir_item_path):
			with open(dir_item_path, 'r') as f:
				# read content of document (doc, content)
				data[dir_item] = f.read()
	# data(dict)
	return data	

# read background model
def read_relevance_dict():
	HMMTraingSetDict = defaultdict(list)
	HMMTraingSet_Path = "../Corpus/TDT2/Train/QDRelevanceTDT2_forHMMOutSideTrain"
	with io.open(HMMTraingSet_Path, 'r', encoding = 'utf8') as file:
		# read content of query document (doc, content)
		query_name = ""
		for line in file.readlines():
			result = line.split()
			if len(result) > 1:
				query_name = result[1]
				continue
			HMMTraingSetDict[query_name].append(result[0])
	# HMMTraingSetDict{word, probability}
	return HMMTraingSetDict	

# read background model
def read_background_dict():
	BGTraingSet = np.zeros(51253)
	# XIN1998
	for doc_item in os.listdir(bg_modle_path):
		# join dir path and file name
		doc_item_path = os.path.join(bg_modle_path, doc_item)
		# check whether a file exists before read
		if os.path.isfile(doc_item_path):
			with io.open(doc_item_path, 'r', encoding = 'utf8') as f:
				# read content of query document (doc, content)
				lines = f.readlines()
				for line in lines:
					[id, prob] = line.split()
					prob = exp(float(prob))
					BGTraingSet[int(id)] = prob
	# Background{word, probability}
	return np.array([BGTraingSet])

	
# document preprocess
def doc_preprocess(dictionary, res_pos = False, str2int = False):
	dictionary = collections.OrderedDict(sorted(dictionary.items()))
	for key, value in dictionary.items():
		content = ""
		temp_content = ""
		count = 0
		# split content by special character
		for line in dictionary[key].split('\n'):
			if count < 3:
				count += 1
				continue
			else:	
				for word in line.split('-1'):
					temp_content += word + " "
		# delete double white space
		for word in temp_content.split():
			content += word + " "
		# replace old content
		dictionary[key]	= content
		# content to int list
		if str2int: 
			int_rep = map(int, content.split())
			# top200
			if len(int_rep) > 200:
				int_rep = int_rep[:200]
			dictionary[key] = int_rep
		
	if not res_pos:
		doc_freq = {}	
		# term probability(word_count / word sum)	
		for doc_key, doc_content in dictionary.items():
			doc_words = word_count(doc_content, {})
			dictionary[doc_key] = doc_words
		#dictionary = TFIDF(dictionary)	
		
	return dictionary

# query preprocess
def query_preprocess(dictionary, res_pos = False, str2int = False):
	dictionary = collections.OrderedDict(sorted(dictionary.items()))
	for key, value in dictionary.items():
		content = ""
		temp_content = ""
		# split content by special character
		for line in dictionary[key].split('\n'):
			for word in line.split('-1'):
				temp_content += word + " "
		# delete double white space
		for word in temp_content.split():
			content += word + " "
		# replace old content
		dictionary[key]	= content
		# content to int list
		if str2int: 
			int_rep = map(int, content.split())
			# top200
			if len(int_rep) > 200:
				int_rep = int_rep[:200]
			dictionary[key] = int_rep
	if not res_pos:	
		qry_freq = {}	
		# term probability(word_count / word sum)	
		for qry_key, qry_content in dictionary.items():
			qry_words = word_count(qry_content, {})
			dictionary[qry_key] = qry_words
		#dictionary = TFIDF(dictionary)	
	return dictionary
	
# create unigram
def unigram(topic_wordcount_dict):
	topic_wordprob_dict = {}
	for topic, wordcount in topic_wordcount_dict.items():
		length = 1.0 * word_sum(wordcount)
		word_prob = {}
		for word, count in wordcount.items():
			word_prob[word] = count / length
		topic_wordprob_dict[topic] = word_prob
	topic_wordprob_dict = collections.OrderedDict(sorted(topic_wordprob_dict.items()))	
	return topic_wordprob_dict 

# modeling	
def modeling(topic_wordprob_dict, background_model, alpha):
	modeling_dict = {}
	for topic, wordprob in topic_wordprob_dict.items():
		word_model = {}
		for word in wordprob.keys():
			word_model[word] = (1-alpha) * wordprob[word] + (alpha) * background_model[word]
		modeling_dict[topic] = dict(word_model)
	return modeling_dict

# word count
def word_count(content, bg_word):
	for part in content.split():
		if part in bg_word:
			bg_word[part] += 1
		else:
			bg_word[part] = 1
	# return word count dictionary		
	return bg_word

# input dict
# output sum of word
def word_sum(data):
	return np.array(data.values()).sum(axis = 0)
	
# output ranking list	
def outputRank(query_docs_point_dict):
	cquery_docs_point_dict = sorted(query_docs_point_dict.items(), key=operator.itemgetter(0))
	operation = "w"
	with codecs.open("Query_Results.txt", operation, "utf-8") as outfile:
		for query, docs_point_list in query_docs_point_dict.items():
			outfile.write(query + "\n")	
			out_str = ""
			for docname, score in docs_point_list:
				out_str += docname + " " + str(score) + "\n"
			outfile.write(out_str)
			outfile.write("\n")		

# softmax			
def softmax(model):
	model_word_sum  = 1.0 * word_sum(model)
	model = {w: c / model_word_sum for w, c in dict(model).items()}
	return model
	
def docFreq(doc, vocab_size = 51253):
	#0:docfreq 1:count
	corpus_dFreq_total = np.zeros((vocab_size, 2))
	for name, word_list in doc.items():
		temp_word_list = {}
		cont_type = type(word_list)
		# str to dict
		if isinstance(word_list, types.StringType):
			temp_word_list = word_count(word_list, {})
		# list to dict
		elif isinstance(word_list, types.ListType):
			temp_word_list = {}
			for part in word_list:
				if part in temp_word_list:
					temp_word_list[part] += 1
				else:
					temp_word_list[part] = 1
		elif isinstance(word_list, types.DictType):
			temp_word_list = dict(word_list)
		# assume type of word_list is dictionary
		for word, word_count in temp_word_list.items():
			corpus_dFreq_total[int(word), 0] += 1
			corpus_dFreq_total[int(word), 1] += word_count
	return corpus_dFreq_total
		
def rmStopWord(ori_content, corpus_d_freq_total, threshold = 0.1):
	weight_list = []
	corpus_length = corpus_dFreq_total[:, 1].sum(axis=0)
	cont_type = type(dict)
	for name, word_list in ori_content:
		cont_type = type(word_list)
		# str to dict
		if cont_type == type(str):
			temp_word_list = word_count(word_list, {})
		# list to dict
		elif cont_type == type(list):
			temp_word_list = {}
			for part in word_list.split():
				if part in temp_word_list:
					temp_word_list[part] += 1
				else:
					temp_word_list[part] = 1
		elif cont_type == type(dict):
			temp_word_list = dict(word_list)
		# assume type of word_list is dictionary
		cur_length = sum(temp_word_list.values())
		for word, word_count in temp_word_list.items():
			word_prob = word_count * 1.0 / cur_length
			corpus_word_prob = corpus_dFreq_total[int(word)][0] * 1.0 / corpus_length
			weight = word_prob * log(word_prob / corpus_word_prob)
			weight_list.append([name, word, weight])
	
	sorted(weight_list, key = lambda x : x[2])
	len_weight_list = len(weight_list)
	for i in xrange(len_weight_list * threshold):
		[name, word, weight] = weight_list[i]
		word_list = ori_content[name]
		# remove low weighted word(string)
		if cont_type == type(str):
			temp_list = word_list.replace(word + " ", "")
			word_list = temp_list.replace(" " + word, "")
		# remove low weighted word(list)
		elif cont_type == type(list):
			word_list = filter(lambda a: a != word, word_list)
		# remove low weighted word(dict)
		elif cont_type == type(dict):
			word_list.pop(word, None)
		# assign new value to name	
		ori_content[name] = word_list	
			
	return ori_content