import codecs
import io
import os
import fileinput
import collections


CNA_path = "Corpus"
Cluster_path = "clusters"
# read document
def read_doc():
	CNATraingSet = []
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
				CNATraingSet.append(f.read())
	# CNATraingSet(list)
	return CNATraingSet
	
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