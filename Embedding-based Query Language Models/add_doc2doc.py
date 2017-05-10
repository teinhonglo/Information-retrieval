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

	return dictionary

document_path = "../Corpus/SPLIT_DOC_WDID_NEW"
doc_content = read_file(document_path)
doc_content = doc_preprocess(doc_content)

with open("XIN1998.18461.wid.Train", 'a') as outfile:
	for doc, content in doc_content.items():
		outfile.write(content)
		outfile.write("\n")