import sys
sys.path.append("../tools")
import operator
import numpy as np
import ProcDoc
from collections import defaultdict
from math import log
import os

data = {}                # content of document (doc, content)
background_model = {}    # word count of 2265 document (word, number of words)
general_model = {}
query = {}                # query
vocabulary = np.zeros(51253)

corpus = "TDT2"
document_path = "../Corpus/" + corpus + "/SPLIT_DOC_WDID_NEW"    
query_path = "../Corpus/" + corpus + "/XinTrainQryTDT3/QUERY_WDID_NEW"
test_query_path = "../Corpus/"+ corpus + "/XinTestQryTDT3/QUERY_WDID_NEW"
resPos = True

# read document, reverse position
doc = ProcDoc.read_file(document_path)
doc = ProcDoc.doc_preprocess(doc, resPos)

# read query, reserve position
query = ProcDoc.read_file(query_path)
query = ProcDoc.query_preprocess(query, resPos)

# read test lone query model, reserve postion
test_query = ProcDoc.read_file(query_path)
test_query = ProcDoc.query_preprocess(test_query, resPos)

# HMMTrainingSet
HMMTraingSetDict = ProcDoc.read_relevance_dict()
query_relevance = {}

# create outside query model
query_model = []
q_list = query_unigram.keys()
for q, w_uni in query_unigram.items():
    if q in HMMTraingSetDict:
        vocabulary = np.zeros(51253)
        for w, uni in w_uni.items():
            vocabulary[int(w)] = uni
        query_model.append(np.copy(vocabulary))
    else:
        q_list.remove(q)
query_model = np.array(query_model).astype(np.float32)

# document model
doc_list = doc_wordcount.keys()
doc_model = []
for doc_name in doc_list:
    vocabulary = np.zeros(51253)
    for word, count in doc_wordcount[doc_name].items():
        vocabulary[int(word)] = count
    #vocabulary /= vocabulary.sum(axis = 0)
    doc_model.append(np.copy(vocabulary))
doc_model = np.array(doc_model).astype(np.float32)    

with open(storage_path + "doc_list.pkl", "wb") as file: Pickle.dump(doc_list, file, True)
with open(storage_path + "doc_model.pkl", "wb") as file: Pickle.dump(doc_model, file, True)



#test_query_unigram = ProcDoc.unigram(test_query_wordcount)
test_query_unigram = test_query_wordcount
test_query_list = test_query_unigram.keys()

test_query_model = []
for q in test_query_list:
    vocabulary = np.zeros(51253)
    for word, unigram in test_query_unigram[q].items():
        vocabulary[int(word)] = unigram
    test_query_model.append(np.copy(vocabulary))
test_query_model = np.array(test_query_model).astype(np.float32)

