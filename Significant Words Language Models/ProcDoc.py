# -*- coding: utf-8 -*-
import codecs
import io
import os
import fileinput
import collections
import numpy as np
from math import exp
from webbrowser import BackgroundBrowser

bg_modle_path = "../Corpus/background"
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
def read_background_dict():
	BGTraingSetDict = {}
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
					BGTraingSetDict[id] = prob
	# Background{word, probability}
	return BGTraingSetDict

# document preprocess
def doc_preprocess(dictionary):
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
	# term probability(word_count / word sum)	
	for doc_key, doc_content in dictionary.items():
		doc_words = word_count(doc_content, {})
		dictionary[doc_key] = doc_words
		
	return dictionary

# query preprocess
def query_preprocess(dictionary):
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
	modeling_dict = collections.OrderedDict(sorted(modeling_dict.items()))		
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
	cquery_docs_point_dict = collections.OrderedDict(sorted(query_docs_point_dict.items()))
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
			
			
def outputModel(model):
	cquery_docs_point_dict = collections.OrderedDict(sorted(query_docs_point_dict.items()))
	operation = "w"
	with codecs.open("Query_Results.txt", operation, "utf-8") as outfile:
		for query, docs_point_list in query_docs_point_dict.items():
			outfile.write(query + "\n")	
			out_str = ""
			for docname, score in docs_point_list:
				out_str += docname + " " + str(score) + "\n"
			outfile.write(out_str)
			outfile.write("\n")				