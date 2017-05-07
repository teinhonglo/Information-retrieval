import numpy as np
import Expansion
import word2vec_model
import cPickle as Pickle

def EmbeddedQuery(query_model, query_wordcount, collection, word2vec, interpolated_aplpha, m):
	
	word2vec_wv = word2vec.getWord2Vec()
	vocab = word2vec_wv.vocab
	vocab_length = 100
	query_embedded = {}
	# assign word vector to collection
	for word, count in collection.items():
		if word in vocab:
			collection[word] = word2vec_wv[word]
		else:
			collection[word] = np.random.rand(vocab_length) * 5 - 2.5

	# assign word vector to query embedded
	for query_key, wordcount in query_wordcount.items():
		for word, count in wordcount.items():
			if not word in query_embedded:
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

	print "Conditional Independence of Query Terms"	
	# Conditional Independence of Query Terms
	query_model_eqe1 = Expansion.embedded_query_expansion_ci(query_model, query_embedded, query_wordcount, collection, collection_total_similarity, word2vec, interpolated_aplpha, m)
	Pickle.dump(query_model_eqe1, open("model/eqe1.pkl", "wb"), True)

	print "Query-Independent Term Similarities"	
	# Query-Independent Term Similarities
	query_model_eqe2 = Expansion.embedded_query_expansion_qi(query_model, query_embedded, query_wordcount, collection, collection_total_similarity, word2vec, interpolated_aplpha, m)
	Pickle.dump(query_model_eqe2, open("model/eqe2.pkl", "wb"), True)
	
	return [query_model_eqe1, query_model_eqe2]