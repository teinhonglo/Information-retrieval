import os
import fileinput
from collections import defaultdict
import collections
from math import log, sqrt
import operator
import numpy as np
import Evaluate
import ProcDoc
import readAssessment
import timeit

data = {}                # content of document (doc, content)
query = {}                # query
doc_freq ={}
k1 = 1.4
b = 0.75

document_path = "../Corpus/TDT2/SPLIT_DOC_WDID_NEW"
#document_path = "../Corpus/TDT2/Spoken_Doc"
#query_path = "../Corpus/TDT2/QUERY_WDID_NEW"
#query_path = "../Corpus/TDT2/Train/XinTrainQryTDT2/QUERY_WDID_NEW"
query_path = "../Corpus/TDT2/QUERY_WDID_NEW_middle"

# document model
data = ProcDoc.read_file(document_path)
doc_wordcount = ProcDoc.doc_preprocess(data)
ave_doc_length = ProcDoc.compute_average_doc_length(doc_wordcount)
total_docs = len(doc_wordcount.keys()) * 1.0
[doc_model, doc_freq] = ProcDoc.compute_TFIDF(doc_wordcount)

# query model
query = ProcDoc.read_file(query_path)
query = ProcDoc.query_preprocess(query)
query_wordcount = {}

for q_key, q_content in query.items():
    query_wordcount[q_key] = ProcDoc.word_count(q_content, {})

query_model = defaultdict(dict)    
for q_key, word_count_dict in query_wordcount.items():
    max_freq = np.max(np.array(word_count_dict.values()), axis = 0)
    for word, count in word_count_dict.items():
        if word in doc_freq:
            idf = log((total_docs - doc_freq[word] + 0.5) / (doc_freq[word] + 0.5))            
        else:
            idf = log((total_docs + 0.5) / 0.5)
        query_model[q_key][word] = (1 + log(count)) * idf    

#HMMTraingSetDict = ProcDoc.read_relevance_dict()
#for q, w_uni in query_model.items():
#    if q in HMMTraingSetDict:
#        continue
#    else:
#        query_model.pop(q, None)
#        query_wordcount.pop(q, None)
#print(len(query_model.keys()))

assessment = readAssessment.getAssessment()
# query process
print "query ..."
start = timeit.default_timer()
feedback_model = []
feedback_ranking_list = []
results_file = "BM25_spk.txt"
topN = 10

for step in range(1):
    query_docs_point_dict = {}
    AP = 0
    mAP = 0
    q_idx = 0
    with open("pseudo_" + str(topN) + "_" + results_file, "wb") as writer:
        for q_key, q_words_count_list in query_wordcount.items():
            docs_point = {}
            writer.write("Query " + str(q_idx) + " " + q_key + " " + str(topN) + "\n")
            count = 0
            q_idx += 1
            for doc_key, doc_words_count_dict in doc_wordcount.items():
                relevant_point = 0
                irrelevant_point = 0
                query_length = 1
                doc_length = 0
            
                doc_length = np.array(doc_words_count_dict.values()).sum(axis = 0)
                # calculate each query value for the document
                for doc_word, doc_word_count in doc_words_count_dict.items():
                    query_word_freq = 0
                    idf = log((total_docs + 0.5) / 0.5)
                    if doc_word in q_words_count_list:
                        query_word_freq = q_words_count_list[doc_word]
                        idf = query_model[q_key][doc_word]
                    else:
                        continue
                                
                    relevant_point += idf * (query_word_freq * (k1 + 1) ) / (query_word_freq + k1*( 1 - b + b * doc_length / ave_doc_length))
            
                # cosine measure
                docs_point[doc_key] = relevant_point
                # sorted each doc of query by point
            docs_point_list = sorted(docs_point.items(), key=operator.itemgetter(1), reverse = True)
            query_docs_point_dict[q_key] = docs_point_list
            
            for doc_key, doc_point in docs_point_list:
                writer.write(doc_key + "\n")
                count += 1
                if(count == topN): break
            writer.write("\n")
    # mean average precision    
    mAP = readAssessment.mean_average_precision(query_docs_point_dict, assessment)
    print "mAP:", mAP
    print "feedback:", step
    if step < 1:
        feedback_ranking_list = dict(query_docs_point_dict)
    #[query_model, feedback_model] = Expansion.extQueryModel(query_model, feedback_ranking_list, doc_model, feedback_model, step + 1)
stop = timeit.default_timer()
print "Result : ", stop - start    
