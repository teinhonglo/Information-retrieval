import codecs
import io
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from math import log, sqrt
import numpy as np
import timeit
import ProcDoc

start = timeit.default_timer()

[doc_name, documents] = ProcDoc.read_doc()
texts = [[word for word in document.lower().split()] for document in documents]
sentences = [[sentence for sentence in document.lower().split("\n")] for document in documents]
outputDoc = ["PTSND20020124_1",  "PTSND20020207_2", "PTSND20020225_2", "PTSND20020305_2", "PTSNE20020814_1", "PTSND20020130_1",  "PTSND20020218_2",  "PTSND20020225_3", "PTSND20020613_2",  "PTSNE20020814_4",
			"PTSND20020131_1",  "PTSND20020220_1",  "PTSND20020226_1",  "PTSND20020624_1",  "PTSNE20020819_1", "PTSND20020201_1",  "PTSND20020221_1", "PTSND20020305_1", "PTSNE20020731_2", "PTSNE20020820_3"]

total_docs = len(texts) * 1.0

# TFIDF
# Document Frequency and Term Frequency
term_freq = []
doc_freq = {}
for text in texts:
	cur_term_freq = {}
	for token in text:
		if token in cur_term_freq:
			cur_term_freq[token] += 1
		else:	
			cur_term_freq[token] = 1
			if token in doc_freq:
				doc_freq[token] += 1
			else:
				doc_freq[token] = 1
	term_freq.append(cur_term_freq)

# Compute TFIDF
tfidf = []	
for doc_tf in term_freq:
	doc_tfidf = {}
	for term, tf in doc_tf.items():
		idf = log(1 + total_docs / doc_freq[term])
		doc_tfidf[term] = tf / idf	
	tfidf.append(doc_tfidf)	

	
# Compute sentence in document
for doc_index in range(len(tfidf)):	
	if doc_name[doc_index] in outputDoc:
		with codecs.open("result/" + doc_name[doc_index] + ".txt", 'w', "utf-8") as outfile:
			print "Document:", doc_index + 1
			# Create BOW(using TFIDF)
			_tfidf = []
			total_sentences = len(sentences[doc_index])
			for sentence in sentences[doc_index]:
				vector = []
				for token in tfidf[doc_index].keys():
					if token in sentence:
						vector.append(tfidf[doc_index][token])
					else:	
						vector.append(0)
				_tfidf.append(vector)

			# Cosine similarity
			_tfidf = sparse.csr_matrix(_tfidf)
			similarities = cosine_similarity(_tfidf)
			# print('pairwise sparse output:\n {}\n'.format(similarities))
			# print similarities.shape

			# Density
			print "density"
			miss_add = np.array(int(total_sentences) * [1])
			density = (np.array(similarities).sum(axis = 0)  - miss_add) / (total_sentences - 1)
			# print density.shape

			# Divergences
			print "divergences"
			divergences = []
			for sentence_index in range(len(density)):
				larger_density = similarities[sentence_index] * (density > density[sentence_index])
				divergences.append(1 - larger_density[np.argmax(larger_density, axis = 0)])
				
			divergences = np.array(divergences)	
			# print divergences.shape
			# print divergences

			# Top 3
			percentage = 0.1
			print "top ", percentage
			score = (density * divergences)
			higher_score = np.argsort(-score)
			doc_text = 1.0 * len(texts[doc_index])
			cur_percentage = 0
			
			
			for index in higher_score:
				print sentences[doc_index][index], score[index]
				cur_percentage += len(sentences[doc_index][index].split())
				outfile.write(sentences[doc_index][index])
				outfile.write("\n")
				if cur_percentage / doc_text > percentage:
					break
			
			print
stop = timeit.default_timer()
print "Result : ", stop - start