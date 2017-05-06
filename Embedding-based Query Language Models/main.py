import operator
import numpy as np
import readAssessment
import ProcDoc
import Expansion
import plot_diagram
import word2vec_model
from collections import defaultdict
from math import log
import cPickle as Pickle

data = {}				# content of document (doc, content)
background_model = {}	# word count of 2265 document (word, number of words)
general_model = {}
query = {}				# query
query_lambda = 0.4
doc_lambda = 0.8

document_path = "../Corpus/SPLIT_DOC_WDID_NEW"
query_path = "../Corpus/QUERY_WDID_NEW_middle"

# document model
data = ProcDoc.read_file(document_path)
doc_wordcount = ProcDoc.doc_preprocess(data)
doc_unigram = ProcDoc.unigram(dict(doc_wordcount))

# background_model
background_model = ProcDoc.read_background_dict()

# general model
collection = {}
collection_total_similarity = {}
for key, value in doc_wordcount.items():
	for word, count in value.items():
		if word in collection:
			collection[word] += count
		else:
			collection[word] = count
			
collection_word_sum = 1.0 * ProcDoc.word_sum(collection)
general_model = {k : v / collection_word_sum for k, v in collection.items()}

# query model
query = ProcDoc.read_file(query_path)
query = ProcDoc.query_preprocess(query)
query_wordcount = {}
query_embedded = {}

for q, q_content in query.items():
	query_wordcount[q] = ProcDoc.word_count(q_content, {})
	for word in query_wordcount[q].keys():
		query_embedded[word] = 0

query_unigram = ProcDoc.unigram(dict(query_wordcount))
query_model = query_unigram
Pickle.dump(query_model, open("model/query_model_prev.pkl", "wb"), True)
Pickle.dump(doc_unigram, open("model/doc_unigram_prev.pkl", "wb"), True)


# Conditional Independence of Query Terms
m = 50
interpolated_aplpha_list = np.linspace(0.1, 1.0, num=10)
interpolated_aplpha = interpolated_aplpha_list[5]
word2vec = word2vec_model.word2vec_model()
word2vec_wv = word2vec.getWord2Vec()
vocab = word2vec_wv.vocab
vocab_length = 100

# assign word vector to collection
for word, count in collection.items():
	if word in vocab:
		collection[word] = word2vec_wv[word]
	else:
		collection[word] = np.random.rand(vocab_length) * 5 - 2.5

# assign word vector to query embedded
for word, count in query_embedded.items():
	if word in vocab:
		query_embedded[word] = word2vec_wv[word]
	else:
		query_embedded[word] = np.random.rand(vocab_length) * 5 - 2.5
'''
count_of_summation = 1		
# sum of total similarity, adding collection
for word, w_vec in collection.items():
	print count_of_summation
	collection_total_similarity[word] = word2vec.sumOfTotalSimiliary(w_vec, collection)
	count_of_summation += 1

# sum of total similarity, adding query
for word, w_vec in query_embedded.items():
	if word in collection:
		continue
	print count_of_summation	
	collection_total_similarity[word] = word2vec.sumOfTotalSimiliary(w_vec, collection)
	count_of_summation += 1
'''	
#Pickle.dump(collection_total_similarity, open("model/collection_total_similarity.pkl", "wb"), True)
collection_total_similarity = Pickle.load(open("model/collection_total_similarity.pkl", "rb"))

# query process
print "query ..."
assessment = readAssessment.get_assessment()
query_docs_point_fb = {}
query_model_fb = {}
mAP_list = []

for interpolated_aplpha in interpolated_aplpha_list:
	for query_exp in range(2):
		origin_query = {}
		if query_exp < 1:
			print "Conditional Independence of Query Terms"	
			# Conditional Independence of Query Terms
			query_model_eqe1 = Expansion.embedded_query_expansion_ci(query_model, query_embedded, query_wordcount, collection, collection_total_similarity, word2vec, interpolated_aplpha, m)
			origin_query = query_model_eqe1
			#Pickle.dump(query_model_eqe1, open("model/eqe1.pkl", "wb"), True)
		else:
			print "Query-Independent Term Similarities"	
			# Query-Independent Term Similarities
			query_model_eqe2 = Expansion.embedded_query_expansion_qi(query_model, query_embedded, query_wordcount, collection, collection_total_similarity, word2vec, interpolated_aplpha, m)
			origin_query = query_model_eqe2
			#Pickle.dump(query_model_eqe2, open("model/eqe2.pkl", "wb"), True)

		query_model = ProcDoc.modeling(origin_query, background_model, query_lambda)

		for step in range(1):
			query_docs_point_dict = {}
			AP = 0
			mAP = 0
			for q_key, q_word_prob in query_model.items():
				docs_point = {}
				for doc_key, doc_words_prob in doc_unigram.items():
					point = 0
					# calculate each query value for the document
					for query_word, query_prob in q_word_prob.items():
						word_probability = 0			# P(w | D)
						# check if word at query exists in the document
						if query_word in doc_words_prob:
							word_probability = doc_words_prob[query_word]
						# KL divergence 
						# (query model) * log(doc_model) 			
						point += query_model[q_key][query_word] * log((1-doc_lambda) * word_probability + doc_lambda * background_model[query_word])
					docs_point[doc_key] = point
					# sorted each doc of query by point
				docs_point_list = sorted(docs_point.items(), key=operator.itemgetter(1), reverse = True)
				query_docs_point_dict[q_key] = docs_point_list
			# mean average precision	
			mAP = readAssessment.mean_average_precision(query_docs_point_dict, assessment)
			mAP_list.append(mAP)
			print "mAP"
			print mAP
			if step < 1:
				query_docs_point_fb = dict(query_docs_point_dict)
				query_model_fb = dict(query_model)
	
	# query_model = Expansion.feedback(query_docs_point_fb, query_model_fb, dict(doc_unigram), dict(doc_wordcount), dict(general_model), dict(background_model), step + 1)
X_list = interpolated_aplpha_list
'''
plot_diagram.plotList(X_list, mAP_list[::2], "Conditional Independence of Query Terms")
plot_diagram.plotList(X_list, mAP_list[1::2], "Conditional Independence of Query Terms")
'''
	