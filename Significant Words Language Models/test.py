import ProcDoc
import plot_diagram
from collections import defaultdict
from pprint import pprint  # pretty-printer

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
                    print "the same"
                    continue
                cur_doc_word_prob *= (1 - other_prob)
            word_specific_level += cur_doc_word_prob	
        specific_model[word] = word_specific_level
	# softmax
    # specific_model = ProcDoc.softmax(dict(specific_model))
                
    return [feedback_w_doc, specific_model]


feedback_doc = {"1100":{"a": 0.5, "b": 0.1}, "1101":{"c": 0.1, "b": 0.2}, "1103":{"a": 0.4, "e": 0.2}}	
feedback_doc_wc = {"1100":{"a": 5, "b": 1}, "1101":{"c": 1, "b": 2}, "1103":{"a": 4, "e": 2}}
[feedback_doc_n, specific_model] = specific_modeling(dict(feedback_doc))
print "origin"
pprint(feedback_doc)
pprint(feedback_doc_n)
pprint(specific_model)
specific_model = ProcDoc.softmax(specific_model)
pprint(specific_model)

plot_diagram.plotModel(specific_model, specific_model, specific_model, feedback_doc_wc, feedback_doc)