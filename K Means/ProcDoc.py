import codecs
import io
import os
import fileinput
import collections


CNA_path = "Corpus"

# read documant
def read_doc():
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
				# read content of query documant (doc, content)
				for line in f.readlines():
					numOfDoc += 1
					CNATraingSetDict[str(numOfDoc)] = line
	# CNATraingSetDict(No., content)
	return CNATraingSetDict
	
# word count
def word_count(content, bg_word):
	for part in content.split():
		if part in bg_word:
			bg_word[part] += 1
		else:
			bg_word[part] = 1
	return bg_word	
	
def printDict(dict):
	for key, val in dict.items():
		print key
		print val
	c = raw_input("Press any key to continue...")