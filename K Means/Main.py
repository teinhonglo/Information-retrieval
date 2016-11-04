# -*- coding: utf-8 -*- 
import numpy as np
import ProcDoc
import Kmeans


doc_wordCount = ProcDoc.read_doc()  # read document (Doc No.,Doc content)  
bg_word = {}  # background (word, count)

# calculate word of the background
# convert (Doc No.,Doc content) to (Doc_No, {word, count})
for docName, content in doc_wordCount.items():
	temp_dict = ProcDoc.word_count(content, {})
	doc_wordCount[docName] = temp_dict
	for word, frequency in temp_dict.items():
		if word in bg_word:
			bg_word[word] += int(frequency)
		else:
			bg_word[word] = int(frequency)

		
dict_vectorSpace = dict(doc_wordCount)  # vector space of the document (Doc_No, [0, 0, 1, 0, 1 ...])
dataset = []  # K-means data sets. List[np.array([])]

# calculate vector space of each document	
for docName, word_dict in dict_vectorSpace.items():
	word_vector = []
	for word, frequency in bg_word.items():
		if word in word_dict:
			word_vector.append(1)
		else:
			word_vector.append(0)	
	dict_vectorSpace[docName] = np.array(word_vector)
	# assign ID and coordinate
	ID = docName
	Coor = np.array(word_vector)
	# create data object
	data = Kmeans.dataInfo(ID, np.array(Coor))
	# append np_array to data sets
	dataset.append(data)

# K means
clusters = Kmeans.kmeans(dataset, 160)

# cluster
cluster_wordProb = {}  # word probability {word, probability}
cluster_Num = 0
for cluster in clusters:
	# calculate number of word
	cluster_wordCount = {}
	for d in cluster:
		cluster_wordCount = ProcDoc.word_count_dict(doc_wordCount[d.ID], cluster_wordCount)
    
	# calculate probability of word	
	word_prob = {}	
	for word, frequency in cluster_wordCount.items():
		word_prob[word] = frequency * 0.1 / bg_word[word]
    
	# assign each cluster to word probability	    
	cluster_wordProb["cluster " + str(cluster_Num)]	 = word_prob
	cluster_Num += 1
	    
# print word probability of the cluster
for cluster, word_prob in cluster_wordProb.items():
	print(cluster)
	for word, prob in word_prob.items():
		print(word, prob)
	print("Cluster ends here.")
