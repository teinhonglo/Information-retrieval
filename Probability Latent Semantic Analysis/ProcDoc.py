# -*- coding: utf-8 -*-
import codecs
import io
import os
import fileinput
import collections


CNA_path = "../Corpus/SPLIT_DOC_WDID_NEW"
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
def read_doc_dict():
	CNATraingSetDict = {}
	title = "Doc "
	numOfDoc = 0
	# CNA_only_utf8
	for doc_item in os.listdir(CNA_path):
		# join dir path and file name
		doc_item_path = os.path.join(CNA_path, doc_item)
		# check whether a file exists before read
		if os.path.isfile(doc_item_path):
			with io.open(doc_item_path, 'r', encoding = 'utf8') as f:
				# read content of query document (doc, content)
				CNATraingSetDict[str(numOfDoc)] = f.read()
				numOfDoc += 1
	# CNATraingSetDict(No., content)
	return CNATraingSetDict

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
			content += str(int(word)) + " "
		# replace old content
		dictionary[key]	= content

	#dictionary = TFIDF(dictionary)	
	return dictionary
	
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