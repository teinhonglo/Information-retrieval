# -*- coding: utf-8 -*-
import codecs
import io
import os
import fileinput
import collections
import numpy as np
import operator
from math import exp
from webbrowser import BackgroundBrowser
from collections import defaultdict
from math import log

bg_modle_path = "../../Corpus/background"
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

def inverse_document_frequency(doc_word_unigram_dcit):
	invert_word_document = inverted_word_doc(doc_word_unigram_dcit)
	word_idf = {}
	document_list = []
	for word, doc_wordcount in invert_word_document.items():
		word_idf[word] = len(doc_wordcount.keys())
		for doc, count in doc_wordcount.items():
			if doc in document_list:
				continue
			document_list.append(doc)	
	total_doc_length = 1.0 * len(document_list)
	word_idf = {word: 1 / log(1 + total_doc_length / df) for word, df in dict(word_idf).items()}
	return word_idf
	
def inverted_word_doc(doc_word_unigram_dcit):	
    inverted_w_doc = defaultdict(dict)
    for doc_name, word_unigram in doc_word_unigram_dcit.items(): 
        for word, prob in word_unigram.items():
            inverted_w_doc[word][doc_name] = prob	
    return inverted_w_doc		
	
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

def compute_TFIDF(doc_wordcount):
	tfidf = {}
	total_docs = len(doc_wordcount.keys()) * 1.0
	# compute idf
	doc_freq = {}
	for doc, word_count_dict in doc_wordcount.items():
		for word, count in word_count_dict.items():
			if word in doc_freq:
				doc_freq[word] += 1
			else:
				doc_freq[word] = 1
	# compute tfidf
	for doc, word_count_dict in doc_wordcount.items():
		doc_tfidf = {}
		
		for word, count in word_count_dict.items():
			idf = log(1 + total_docs / doc_freq[word])
			tf = 1 + log(count)
			doc_tfidf[word] = tf * idf
		tfidf[doc] = doc_tfidf
	
	return [tfidf, doc_freq]
		
		
