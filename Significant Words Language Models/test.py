import ProcDoc
import plot_diagram

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
            print word
            for other_doc_name, other_word_unigram in feedback_doc.items():
                if cur_doc_name == other_doc_name:
                    continue
                # penalize term frequency is supported by other documents
                if word in other_word_unigram:
                    print prob, " * ",(1 - other_word_unigram[word])
                    prob *= (1 - other_word_unigram[word])   
            if word in specific_model:
                specific_model[word] += prob
            else:            
                specific_model[word] = prob
                
    return [feedback_doc, specific_model]


feedback_doc = {"1100":{"a": 0.5, "b": 0.1}, "1101":{"c": 0.1, "b": 0.2}, "1103":{"a": 0.4, "e": 0.2}}	
feedback_doc_wc = {"1100":{"a": 5, "b": 1}, "1101":{"c": 1, "b": 2}, "1103":{"a": 4, "e": 2}}
[feedback_doc_n, specific_model] = specific_modeling(dict(feedback_doc))
from pprint import pprint  # pretty-printer
pprint(feedback_doc)
pprint(feedback_doc_n)
pprint(specific_model)
specific_model = ProcDoc.softmax(specific_model)
pprint(specific_model)

plot_diagram.plotModel(specific_model, specific_model, specific_model, feedback_doc_wc, feedback_doc)