# -*- coding: utf-8 -*-
import codecs
import io
import os
import fileinput
import collections
from math import exp

bg_modle_path = "background"
Cluster_path = "Topic"

# read cluster
def read_clusters():
	clusters = {}
	# cluster_only_utf8
	for cluster_item in os.listdir(Cluster_path):
		# join dir path and file name
		cluster_item_path = os.path.join(Cluster_path, cluster_item)
		# check whether a file exists before read
		if os.path.isfile(cluster_item_path):
			with io.open(cluster_item_path, 'r', encoding = 'utf8') as f:
				# read content of query document (doc, content)
				content =  f.readlines()
				words = (content[0]).split(",")
				for line in content[1:]:
					cluster_info = line.split(",")
					cluster_name = cluster_info[0]
					word_prob_dict = {}
					for w_index in range(1, len(cluster_info)):
						[word, prob ]= [words[w_index], cluster_info[w_index]]
						word_prob_dict[word] = float(prob)
					clusters[cluster_name] = word_prob_dict
	# clusters(list)
	return clusters	
	
# read document
def read_background_dict():
	BGTraingSetDict = {}
	# CNA_only_utf8
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
	# CNATraingSetDict(No., content)
	return BGTraingSetDict

# preprocess
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
	dict_tf = {}	
	dictionary_prob = {}
	# term probablity(word_count / word sum)	
	for doc_key, doc_content in dictionary.items():
		doc_words = word_count(doc_content, {})
		# term frequency
		dict_tf[doc_key] = dict(doc_words)
		
		# term probablity
		doc_words_sum = word_sum(doc_words) * 1.0
		for word, word_val  in doc_words.items():
			doc_words[word] = word_val / doc_words_sum
		dictionary_prob[doc_key] = dict(doc_words)
		
	return dictionary_prob

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
	
def background_word_reprobability(background_word):
	background_word_sum = word_sum(background_word) * 1.0
	for word, count in background_word.items():
		background_word[word] = (count  + 0.01) / (background_word_sum  + 0.01)
	return background_word	
	
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
	num = 0
	for key, value in data.items():
		num += int(value)
	return num