import ProcDoc

def specific_modeling(feedback_doc):
    # normalize, sum of the (word_prob = 1) in the document
    for doc_name, word_unigram in feedback_doc.items():
        prob_sum = 1.0 * ProcDoc.word_sum(word_unigram)
        uni_normalize = {w: w_uni / prob_sum for w, w_uni in word_unigram.items()}
        feedback_doc[doc_name] = uni_normalize
        
    # specific modeling
    # if the term frequency is supported by almost all documents 
    # the term will be penalized because of its low prevalence.
    specific_model = {}
    for cur_doc_name, cur_word_unigram in feedback_doc.items():
        # calculate each word in current document
        for word, prob in cur_word_unigram.items():
            # calculate each word in other document
            for other_doc_name, other_word_unigram in feedback_doc.items():
                if cur_doc_name == other_doc_name:
                    continue
                # penalize term frequency is supported by other documents
                if word in other_word_unigram:
                    prob *= (1 - other_word_unigram[word])   
            if word in specific_model:
                specific_model[word] += prob
            else:            
                specific_model[word] = prob
                
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
        for word in significant_model.keys():
            word_sum = 0
            for doc_name, word_count in feedback_doc.items():
                if word in word_count:
                    word_sum += word_count[word] * hidden_significant_doc_word[doc_name][word]
                    denominator += word_sum
            significant_model[word] = word_sum
        
        significant_model = {word: word_sum / denominator for word, word_sum in dict(significant_model).items()}
    return significant_model            

def feedback(query_docs_point_dict, query_unigram, doc_unigram, doc_wordcount, general_model, background_model, topN):
    lambda_bg = 0.1
    lambda_fb = 0.8
    lambda_q = 1 - lambda_bg - lambda_fb 
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
        
        for word, fb_w_prob in significant_model.items():
            original_prob = 0.0
            if word in query_unigram[q_key]:
                original_prob = query_unigram[q_key][word]
            else:
                original_prob = 0.0
            # update query unigram    
            query_unigram[q_key][word] = (lambda_q * original_prob) + (lambda_fb * fb_w_prob) + (lambda_bg * background_model[word])    
 
    return query_unigram 