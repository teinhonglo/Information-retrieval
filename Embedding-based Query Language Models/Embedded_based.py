import numpy as np
import Expansion
import word2vec_model
import cPickle as Pickle
import os.path

def EmbeddedQuery(query_wordcount, collection, word2vec, interpolated_aplpha_list, m):
	
	word2vec_wv = word2vec.getWord2Vec()
	vocab = word2vec_wv.vocab
	vocab_length = len(word2vec_wv[vocab.keys()[0]])
	query_embedded = {}
	collection_total_similarity = {}
	
	np.random.seed(1337)
	# assign word vector to collection
	if os.path.isfile("model/collection_embedded.pkl") == True:
		# check if a file exist
		collection = Pickle.load(open("model/collection_embedded.pkl", "rb"))
	else:
		for word, count in collection.items():
			if word in vocab:
				collection[word] = word2vec_wv[word]
			else:
				#collection[word] = np.random.rand(vocab_length) * 5 - 2.5
				collection[word] = np.random.uniform(-2.5, +2.5, vocab_length)
				#collection[word] = word2vec.getMeanVec()
				#collection.pop(word, None)
				
			collection[word] /= np.sqrt((collection[word] ** 2).sum(axis = 0))
				
		Pickle.dump(collection, open("model/collection_embedded.pkl", "wb"), True)
	
	# assign word vector to query embedded	
	if os.path.isfile("model/query_embedded.pkl") == True:
		# check if a file exist
		query_embedded = Pickle.load(open("model/query_embedded.pkl", "rb"))
	else:	
		for query_key, wordcount in query_wordcount.items():
			for word, count in wordcount.items():
				if not word in query_embedded:
					if word in vocab:
						query_embedded[word] = word2vec_wv[word]
						query_embedded[word] /= np.sqrt((query_embedded[word] ** 2).sum(axis=0))		
					else:
						if word in collection:
							query_embedded[word] = collection[word]
						else:
							#squery_embedded[word] = np.random.rand(vocab_length) * 5 - 2.5
							query_embedded[word] = np.random.uniform(-2.5, +2.5, vocab_length)
							query_embedded[word] /= np.sqrt((query_embedded[word]**2).sum(axis = 0))
							#query_embedded[word] = word2vec.getMeanVec()
							#pass
		Pickle.dump(query_embedded, open("model/query_embedded.pkl", "wb"), True)				
	
	if os.path.isfile("model/collection_total_similarity.pkl") == True: 
		collection_total_similarity = Pickle.load(open("model/collection_total_similarity.pkl", "rb"))
	else:
		# sum of total similarity, adding collection
		collection_total_similarity = word2vec.sumOfTotalSimilarity(collection, collection)

		# sum of total similarity, adding query
		query_total_similarity = {}
		query_total_similarity = word2vec.sumOfTotalSimilarity(query_embedded, collection)
		for word, word_vec in query_total_similarity.items():
			collection_total_similarity[word] = word_vec

		Pickle.dump(collection_total_similarity, open("model/collection_total_similarity.pkl", "wb"), True)
		

	print "Conditional Independence of Query Terms"	
	# Conditional Independence of Query Terms
	query_model_eqe1 = Expansion.embedded_query_expansion_ci(query_embedded, query_wordcount, collection, collection_total_similarity, word2vec, interpolated_aplpha_list, m)
	#Pickle.dump(query_model_eqe1, open("model/eqe1.pkl", "wb"), True)

	print "Query-Independent Term Similarities"	
	# Query-Independent Term Similarities
	query_model_eqe2 = Expansion.embedded_query_expansion_qi(query_embedded, query_wordcount, collection, collection_total_similarity, word2vec, interpolated_aplpha_list, m)
	#Pickle.dump(query_model_eqe2, open("model/eqe2.pkl", "wb"), True)
			
	return [query_model_eqe1, query_model_eqe2]
