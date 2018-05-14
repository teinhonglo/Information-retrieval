import numpy as np
import word2vec_model
import cPickle as Pickle
import os.path
import ProcDoc
import operator
class EmbeddedBased():
    def __init__(self, query_wordcount, collection, word2vec):
        word2vec = word2vec
        query_embedded = {}
        collection_total_similarity = {}
        collection = collection
        query_wordcount = query_wordcount
        
        word2vec_wv = word2vec.getWord2Vec()
        vocab = word2vec.getVocab()
        vocab_length = len(word2vec_wv[vocab[0]])
        
        np.random.seed(1337)
        print "Word vector"
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
                                #query_embedded[word] = word2vec.getMeanVec()
                        query_embedded[word] /= np.sqrt((query_embedded[word]**2).sum(axis = 0))
            Pickle.dump(query_embedded, open("model/query_embedded.pkl", "wb"), True)                
        print "Calculate Similarity"
        if os.path.isfile("model/collection_total_similarity.pkl") == True: 
            collection_total_similarity = Pickle.load(open("model/collection_total_similarity.pkl", "rb"))
        else:
            # sum of total similarity, adding collection
            collection_total_similarity = word2vec.sumOfTotalSimilarity(collection, collection)
            print "collection_total_similarity end"
            # sum of total similarity, adding query
            query_total_similarity = {}
            query_total_similarity = word2vec.sumOfTotalSimilarity(query_embedded, collection)
            for word, word_vec in query_total_similarity.items():
                collection_total_similarity[word] = word_vec

            Pickle.dump(collection_total_similarity, open("model/collection_total_similarity.pkl", "wb"), True)
        self.word2vec = word2vec
        self.query_embedded = query_embedded
        self.collection_total_similarity = collection_total_similarity
        self.collection = collection
        self.query_wordcount = query_wordcount
        self.word2vec_wv = word2vec_wv
        self.vocab = vocab

    '''
    Query Expansion using global analysis
        embedded_query_expansion_ci: # Conditional Independence of Query Terms
        embedded_query_expansion_qi: # Query-Independent Term Similarities
        
    '''	
    # Conditional Independence of Query Terms	
    def embedded_query_expansion_ci(self, interpolated_aplpha, m):
        query_embedded = self.query_embedded
        query_wordcount = self.query_wordcount
        collection = self.collection
        collection_total_similarity = self.collection_total_similarity
        word2vec = self.word2vec
    
        # load query model
        query_model = Pickle.load(open("model/query_model.pkl", "rb"))
        embedded_query_expansion = query_model
        
        
        update_embedded_query_expansion = {}
        if os.path.isfile("model/update_embedded_query_expansion_ci.pkl") == True:
            # check if a file exist
            update_embedded_query_expansion = Pickle.load(open("model/update_embedded_query_expansion_ci.pkl", "rb"))
        else:	
            # calculate every query
            for query, query_word_count_dict in query_wordcount.items():
                top_prob_dict = {}
                # calculate every word in collection
                for word in collection.keys():
                    total_probability = collection_total_similarity[word]
                    p_w_q = 0
                    if not word in query_word_count_dict:
                        p_w_q = total_probability				# p(w|q)
                        # total probability theory(for every query term)
                        for query_term in query_word_count_dict.keys():
                            if query_term in query_embedded:
                                cur_word_similarity = word2vec.getWordSimilarity(query_embedded[query_term], collection[word])
                                p_w_q *= (cur_word_similarity / total_probability)
                    # storage probability
                    top_prob_dict[word] = p_w_q
                # softmax
                top_prob_dict = ProcDoc.softmax(top_prob_dict)
                # sorted top_prob_dict by value(probability)
                top_prob_list = sorted(top_prob_dict.items(), key=operator.itemgetter(1), reverse = True)
                update_embedded_query_expansion[query] = top_prob_list
            # storage update expansion	
            Pickle.dump(update_embedded_query_expansion, open("model/update_embedded_query_expansion_ci.pkl", "wb"), True)
        
        # update query model	
        for update_query, update_query_word_list in update_embedded_query_expansion.items():
            filepath = "visual/" + update_query + "_ci.png"
            if os.path.isfile(filepath) == False:
                visualization.visualization(collection, update_query_word_list, filepath)
                
            for update_word, update_count in update_query_word_list[:m]:
                update = update_count
                origin = 0
                if update_word in query_model[update_query]:
                    origin = query_model[update_query][update_word]
                    query_model[update_query].pop(update_word, None)
                    
                embedded_query_expansion[update_query][update_word] = interpolated_aplpha * origin + (1 - interpolated_aplpha) * update	
            
            for un_changed_word in query_model[update_query].keys():
                embedded_query_expansion[update_query][un_changed_word] *= interpolated_aplpha
            
            # softmax	
            embedded_query_expansion[update_query] = ProcDoc.softmax(embedded_query_expansion[update_query])	
        return 	embedded_query_expansion		
        
    # Query-Independent Term Similarities
    def embedded_query_expansion_qi(self, interpolated_aplpha, m):
        query_embedded = self.query_embedded
        query_wordcount = self.query_wordcount
        collection = self.collection
        collection_total_similarity = self.collection_total_similarity
        word2vec = self.word2vec
        # copy query model
        query_model = Pickle.load(open("model/query_model.pkl", "rb"))
        embedded_query_expansion = query_model
        
        update_embedded_query_expansion = {}
        if os.path.isfile("model/update_embedded_query_expansion_qi.pkl") == True:
            # check if a file exist
            update_embedded_query_expansion = Pickle.load(open("model/update_embedded_query_expansion_qi.pkl", "rb"))
        else:	
            # calculate every query
            for query, query_word_count_dict in query_wordcount.items():
                top_prob_dict = {}
                # calculate every word in collection
                for word in collection.keys():
                    # for every word in current query
                    query_length = ProcDoc.word_sum(query_word_count_dict) * 1.0
                    # p(w|q)
                    p_w_q = 0
                    if not word in query_word_count_dict:
                        for word_sq, word_sq_count in query_word_count_dict.items():
                            total_probability = collection_total_similarity[word_sq]
                            if word_sq in query_embedded:
                                cur_word_similarity = word2vec.getWordSimilarity(collection[word], query_embedded[word_sq])
                                p_w_q += (cur_word_similarity / total_probability )  * (word_sq_count / query_length)
                    
                    # storage probability
                    top_prob_dict[word] = p_w_q
                # softmax	
                top_prob_dict = ProcDoc.softmax(top_prob_dict)
                # sorted top_prob_dict by value(probability)
                top_prob_list = sorted(top_prob_dict.items(), key=operator.itemgetter(1), reverse = True)
                # storage update query model value
                update_embedded_query_expansion[query] = top_prob_list
            Pickle.dump(update_embedded_query_expansion, open("model/update_embedded_query_expansion_qi.pkl", "wb"), True)	
        
        # update query model	
        for update_query, update_query_word_list in update_embedded_query_expansion.items():
            filepath = "visual/" + update_query + "_qi.png"
            if os.path.isfile(filepath) == False:
                visualization.visualization(collection, update_query_word_list, filepath)
            for update_word, update_count in update_query_word_list[:m]:
                update = update_count
                origin = 0
                if update_word in query_model[update_query]:
                    origin = query_model[update_query][update_word]
                    query_model[update_query].pop(update_word, None)
                    
                embedded_query_expansion[update_query][update_word] = interpolated_aplpha * origin + (1 - interpolated_aplpha) * update
                
            for un_changed_word in query_model[update_query].keys():
                embedded_query_expansion[update_query][un_changed_word] *= interpolated_aplpha	
            # softmax		
            embedded_query_expansion[update_query] = ProcDoc.softmax(embedded_query_expansion[update_query])	
        return 	embedded_query_expansion			
        
