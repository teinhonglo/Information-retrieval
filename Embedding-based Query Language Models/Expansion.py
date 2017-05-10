import ProcDoc
from math import log
import plot_diagram
from collections import defaultdict
import cPickle as Pickle
import operator
import copy
import os.path

def specific_modeling(feedback_doc):
    # normalize, sum of the (word_prob = 1) in the document
    feedback_w_doc = ProcDoc.inverted_word_doc(dict(feedback_doc))
    for word, doc_unigram in feedback_w_doc.items():
        feedback_w_doc[word] = ProcDoc.softmax(dict(doc_unigram))

    # specific modeling
    # if the term frequency is supported by almost all documents 
    # the term will be penalized because of its low prevalence.
    specific_model = {}
    for word, doc_unigram in feedback_w_doc.items():
        # calculate each word in current document
        word_specific_level = 0
        for doc_name, prob in doc_unigram.items():
            cur_doc_word_prob = prob
            for other_doc_name, other_prob in doc_unigram.items():
                if doc_name == other_doc_name:
                    continue
                cur_doc_word_prob *= (1 - other_prob)
            word_specific_level += cur_doc_word_prob	
        specific_model[word] = word_specific_level
	# softmax
    specific_model = ProcDoc.softmax(dict(specific_model))
	
    return specific_model

def significant_modeling(general_model, specific_model, feedback_doc, feedback_doc_wc):
    lambda_sw = 0.1
    lambda_s = 0.2
    lambda_g = 0.7
    significant_model = {}
    # initialize
    feedback_word = []
    for doc_name, word_count in feedback_doc_wc.items():
        for word, count in word_count.items():
            if word in feedback_word:
                continue
            else:
                feedback_word.append(word)
    for s_word in feedback_word:            
        significant_model[s_word] =  1.0 / len(feedback_word)
        
    hidden_significant_doc_word = {}
    objective_value_list = []
    # EM training
    for step in range(100):
        # E Step:
        for doc_name, word_count in feedback_doc_wc.items():
            hidden_word_variable = {}
            for word, count in word_count.items():
                denominator = lambda_sw * significant_model[word] + lambda_s * specific_model[word] + lambda_g * general_model[word]
                hidden_word_variable[word] = lambda_sw * significant_model[word] / denominator
            hidden_significant_doc_word[doc_name] = hidden_word_variable 
        # M Step:
        denominator = 0.0
        for word in list(significant_model.keys()):
            word_sum = 0
            for doc_name, word_count in feedback_doc_wc.items():
                if word in word_count:
                    word_sum += word_count[word] * hidden_significant_doc_word[doc_name][word]
                    denominator += word_sum
            significant_model[word] = word_sum
        
        significant_model = {word: word_sum / denominator for word, word_sum in dict(significant_model).items()}
		
		# softmax
        significant_model = ProcDoc.softmax(dict(significant_model))
		# Objective function
        objective_value = 0.0
        for doc_name, word_count in feedback_doc_wc.items():
            for word, count in word_count.items():
                objective_value += count * log(lambda_sw * significant_model[word] + lambda_g * general_model[word] + lambda_s * specific_model[word])
        objective_value_list.append(objective_value)
    #plot_diagram.plotList(objective_value_list)
    return significant_model            

def feedback(query_docs_point_dict, query_model, doc_unigram, doc_wordcount, general_model, background_model, topN):
    lambda_bg = 0.1
    lambda_fb = 0.8
    lambda_ir_fb = 0.2
    lambda_q = 0.1
    specific_model = {}
    for q_key, docs_point_list in query_docs_point_dict.items():
        feedback_doc = {}
        feedback_doc_wc = {}
        # Extract feedback document 
        for doc_name, point in docs_point_list[0:topN]:
            feedback_doc[doc_name] = doc_unigram[doc_name]
            feedback_doc_wc[doc_name] = doc_wordcount[doc_name]
        # generate specific model    
        specific_model = specific_modeling(dict(feedback_doc))
        # generate significant model
        significant_model = significant_modeling(general_model, specific_model, feedback_doc, feedback_doc_wc)
        '''
        ir_feedback_doc = {}
        ir_feedback_doc_wc = {}
		# Extract irrelevant feedback document 
        for doc_name, point in docs_point_list[len(docs_point_list)-topN:]:
            ir_feedback_doc[doc_name] = doc_unigram[doc_name]
            ir_feedback_doc_wc[doc_name] = doc_wordcount[doc_name]
        # generate specific model    
        ir_specific_model = specific_modeling(dict(ir_feedback_doc))
        # generate significant model
        ir_significant_model = significant_modeling(general_model, ir_specific_model, ir_feedback_doc, ir_feedback_doc_wc)
        '''
        for word, fb_w_prob in significant_model.items():
            original_prob = 0.0
            if word in query_model[q_key]:
                original_prob = query_model[q_key][word]
            else:
                original_prob = 0.0
            # update query unigram  
            query_model[q_key][word] = (lambda_q * original_prob) + (lambda_fb * fb_w_prob) + (lambda_bg * background_model[word])	
        '''
        for word, ir_fb_w_prob in ir_significant_model.items():
            if word in query_model[q_key]:
                query_model[q_key][word] = (1 - lambda_ir_fb) * query_model[q_key][word] + lambda_ir_fb * ir_fb_w_prob
        '''	
        query_model[q_key] = ProcDoc.softmax(dict(query_model[q_key]))	
        
        # plot_diagram.plotModel(general_model, specific_model, significant_model, feedback_doc_wc, feedback_doc)
        
    return query_model 
'''
Query Expansion using global analysis
	embedded_query_expansion_ci: # Conditional Independence of Query Terms
	embedded_query_expansion_qi: # Query-Independent Term Similarities
	
'''	
# Conditional Independence of Query Terms	
def embedded_query_expansion_ci(query_embedded, query_wordcount, collection, collection_total_similarity, word2vec, interpolated_aplpha, m):
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
		time_to_break = m
		for update_word, update_count in update_query_word_list:
			if time_to_break == 0: 	break
			update = update_count
			if update_word in query_model[update_query]:
				origin = query_model[update_query][update_word]
			else:
				origin = 0
			embedded_query_expansion[update_query][update_word] = interpolated_aplpha * origin + (1 - interpolated_aplpha) * update	
			time_to_break -=1
		# softmax	
		embedded_query_expansion[update_query] = ProcDoc.softmax(embedded_query_expansion[update_query])	
	return 	embedded_query_expansion		
	
# Query-Independent Term Similarities
def embedded_query_expansion_qi(query_embedded, query_wordcount, collection, collection_total_similarity, word2vec, interpolated_aplpha, m):
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
		time_to_break = m
		for update_word, update_count in update_query_word_list:
			if time_to_break == 0:	break
			update = update_count
			if update_word in query_model[update_query]:
				origin = query_model[update_query][update_word]
			else:
				origin = 0
			embedded_query_expansion[update_query][update_word] = interpolated_aplpha * origin + (1 - interpolated_aplpha) * update
			time_to_break -=1
		# softmax		
		embedded_query_expansion[update_query] = ProcDoc.softmax(embedded_query_expansion[update_query])	
	return 	embedded_query_expansion			