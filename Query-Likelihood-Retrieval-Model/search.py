import os
import fileinput
import collections
from math import log
import operator
import numpy as np
import readAssessment


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
			# read content of document (doc, content)
            data[dir_item] = f.read()

# preprocess
data = preprocess(data)

# count background_word
for key, value in data.items():
	background_word = word_count(value, background_word)
print len(background_word)
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
# query process
assessment = readAssessment.get_assessment()
lambda_test = {0: 0}
interval	= 0.1
isBreak = "run"
while isBreak != "exit":
	for my_lambda in np.arange(0, 1, interval):
		if my_lambda in lambda_test: 	continue
		else:							
			docs_point = {}
			AP = 0
			mAP = 0
			for q_key, q_val in query.items():
				for doc_key, doc_val in data.items():
					doc_words = {}
					doc_words = word_count(doc_val, doc_words)
					doc_words_sum = word_sum(doc_words) * 1.0
					point = 0
					# calculate each query value for the document
					for query_word in q_val.split():
						count = 0						# C(w , D)
						word_probability = 0			# P(w | D)
						background_probability = 0		# BG(w | D)
						# check if word at query exists in the document
						if query_word in doc_words:
							count = doc_words[query_word]
							word_probability = doc_words[query_word] / doc_words_sum
						if query_word in background_word:	
							background_probability = (background_word[query_word] + 0.01) / (background_word_sum + 0.01)
						else:
							background_probability = (0.01) / (background_word_sum + 0.01)
						point += log(my_lambda * word_probability + (1 - my_lambda ) * background_probability)
					docs_point[doc_key] = point
				# sorted each doc of query by point
				docs_point_list = sorted(docs_point.items(), key=operator.itemgetter(1), reverse = True)
				AP += readAssessment.precision(docs_point_list, assessment[q_key])
			# mean average precision	
			mAP = AP / len(query)
			print my_lambda
			print mAP
			lambda_test[my_lambda] = mAP
	# get key with maximum value in lambda_test dictionary	
	max_lambda = max(lambda_test.iteritems(), key=operator.itemgetter(1))[0]
	max_mAP = lambda_test[my_lambda]
	# print Lambda and Max value
	print "Max:"
	print max_lambda, max_mAP
	isBreak = raw_input("exit ?\n")
	interval /= 10