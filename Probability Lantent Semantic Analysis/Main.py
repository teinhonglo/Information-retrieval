# -*- coding: utf-8 -*-
import ProcDoc
import PLSA
import codecs
from collections import defaultdict
import numpy as np

def run():
	INIT_PROBABILITY = 1.0 / 60
	topic_word_prob_dict = ProcDoc.read_clusters()									# read cluster P(W|T), {T: {W:Prob}}
	doc_topic_prob_dict = defaultdict(dict)														# P(T|D),{D:{T:Prob}} 
	doc_word_topic_prob_dict = defaultdict(dict)									# P(T| w, D), {D: {word:{T:prob}}}
	doc_wc_dict = ProcDoc.read_doc_dict()  											# read document (Doc No.,Doc content)  
	doc_wc_dict = ProcDoc.doc_preprocess(doc_wc_dict)
	# calculate word of the background
	# convert (Doc No.,Doc content) to (Doc_No, {word, count})
	for docName, content in doc_wc_dict.items():
		temp_dict = ProcDoc.word_count(content, {})
		doc_wc_dict[docName] = temp_dict

	# initialize P(T|D)
	print "Initialize P(T|D)"
	for docName, wordCount in doc_wc_dict.items():
		topic_prob = {}
		for topic, wordProb in topic_word_prob_dict.items():
			doc_topic_prob_dict[docName][topic] = INIT_PROBABILITY
			
	'''
	print "Initialize P(T| w, D)"
	for docName, wordCount in doc_wc_dict.items():	
		word_list = {}
		for word, frequency in wordCount.items():	
			topic_prob = {}
			for topic, wordProb in topic_word_prob_dict.items():
				topic_prob[topic] = 0.0
			word_list[word] = topic_prob
		doc_word_topic_prob_dict[docName] = word_list
	'''
	print "start PLSA"
	[topic_word_prob_dict, doc_topic_prob_dict] = PLSA.Probability_LSA(doc_wc_dict, doc_topic_prob_dict, topic_word_prob_dict, doc_word_topic_prob_dict)
	print "end PLSA"
	
	p_plsa = {}			# PLSA P(W|D) {D: {W : Prob}}
	for doc, topic_prob_list in doc_topic_prob_dict.items():
		p_plsa_word = {}
		for topic, doc_prob in topic_prob_list.items():
			for word, word_prob in topic_word_prob_dict[topic].items():
				print word, word_prob
				if word in p_plsa_word:
					p_plsa_word[word] += word_prob * doc_prob
				else:
					p_plsa_word[word] = word_prob * doc_prob
			
		p_plsa[doc] = p_plsa_word

	return p_plsa
	
p_w_d = run()
with codecs.open("cluster_word_prob.txt", 'w', "utf-8") as outfile:	
	isFirst = True
	for d, w_p in p_w_d.items():
		if isFirst:
			title = "doc"
			for w in w_p.keys():
				title += ", " + w
			outfile.write(title + "\n")	
			isFirst = False
		else:	
			outfile.write(d)
			word_prob = ""
			for p in w_p.values():
				prob += "," + p 
			outfile.write(prob)
			outfile.write("\n")