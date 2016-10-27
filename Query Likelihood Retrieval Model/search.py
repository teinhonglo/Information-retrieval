import os
import fileinput
import collections
from math import log
import operator

documant_path = os.getcwd() + "/SPLIT_DOC_WDID_NEW"
query_path = os.getcwd() + "/QUERY_WDID_NEW"
data = {}				# content of document (doc, content)
background_word = {}	# word count of 2265 documant (word, number of words)
query = {}				# query
my_lambda = 0.5

# preprocess
def preprocess(dictionary):
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

# word count
def word_count(content, bg_word):
	for part in content.split():
		if part in bg_word:
			bg_word[part] += 1
		else:
			bg_word[part] = 1
	return bg_word
	
def word_sum(data):
	num = 0
	for key, value in data.items():
		num += int(value)
	return num

# list all files of a directory(Document)
for dir_item in os.listdir(documant_path):
	# join dir path and file name
    dir_item_path = os.path.join(documant_path, dir_item)
	# check whether a file exists before read
    if os.path.isfile(dir_item_path):
        with open(dir_item_path, 'r') as f:
			# read content of documant (doc, content)
            data[dir_item] = f.read()

# preprocess
data = preprocess(data)

# count background_word
for key, value in data.items():
	background_word = word_count(value, background_word)

background_word_sum = word_sum(background_word)

# 16 query documants
for query_item in os.listdir(query_path):
	# join dir path and file name
    query_item_path = os.path.join(query_path, query_item)
	# check whether a file exists before read
    if os.path.isfile(query_item_path):
        with open(query_item_path, 'r') as f:
			# read content of query documant (doc, content)
            query[query_item] = f.read()

# preprocess
query = preprocess(query)

# query
docs_point = {}
for q_key, q_val in query.items():
	for doc_key, doc_val in data.items():
		doc_words = {}
		doc_words = word_count(doc_val, doc_words)
		doc_words_sum = word_sum(doc_words)
		point = 0
		for query_word in q_val.split():
			count = 0
			word_probability = 0
			background_probability = 0
			if (query_word in doc_words):
				count = doc_words[query_word]
				word_probability = doc_words[query_word] / doc_words_sum
				background_probability = (background_word[query_word] + 0.01) / (background_word_sum + 0.01)
			else:
				count = 0
				word_probability = 0
				if(query_word in background_word):
					background_probability = (background_word[query_word] + 0.01) / (background_word_sum + 0.01)
				else:
					background_probability = 0.01 / (background_word_sum + 0.01)
			point += count * log(my_lambda * word_probability + my_lambda * background_probability)
			
		docs_point[doc_key] = point
	docs_point_list = sorted(docs_point.items(), key=operator.itemgetter(1), reverse = True)
	
	print q_key
	for key, val in docs_point_list:
		print key, val
	c = raw_input()