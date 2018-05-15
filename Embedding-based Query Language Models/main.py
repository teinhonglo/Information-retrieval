import os, sys
sys.path.append("../Tools")

import operator
import numpy as np
import ProcDoc
from Evaluate import EvaluateModel
import Expansion
import plot_diagram
import word2vec_model
from Embedded_based import EmbeddedBased
from collections import defaultdict
import cPickle as Pickle

data = {}                # content of document (doc, content)
background_model = {}    # word count of 2265 document (word, number of words)
general_model = {}
query = {}                # query

query_lambda = 0
doc_lambda = 0.9
#remove_list = ["update_embedded_query_expansion_ci.pkl", "update_embedded_query_expansion_qi.pkl", "collection_embedded.pkl", "query_embedded.pkl", "collection_total_similarity.pkl"]
remove_list=[]

document_path = "../Corpus/TDT2/SPLIT_DOC_WDID_NEW"
query_path = "../Corpus/TDT2/QUERY_WDID_NEW_middle"
word_emb_path = "data/word2vec_dict.pkl"
relevance_path = "../Corpus/TDT2/AssessmentTrainSet/AssessmentTrainSet.txt"

# document model
data = ProcDoc.read_file(document_path)
doc_wordcount = ProcDoc.doc_preprocess(data)
doc_unigram = ProcDoc.unigram(dict(doc_wordcount))
doc_mdl, doc_IDs = ProcDoc.dict2np(doc_unigram)
# background_model
background_model = ProcDoc.read_background_dict()
background_model_np = ProcDoc.read_background_np()

# document smoothing 
for doc_idx in xrange(doc_mdl.shape[0]):
	doc_vec = doc_mdl[doc_idx]
	doc_mdl[doc_idx] = (1 - doc_lambda) * doc_vec + doc_lambda * background_model_np

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

for q, q_content in query.items():
    query_wordcount[q] = ProcDoc.word_count(q_content, {})

query_unigram = ProcDoc.unigram(dict(query_wordcount))
query_model = query_unigram
Pickle.dump(query_model, open("model/query_model.pkl", "wb"), True)

# remove template file
for rm_file in remove_list:
    if os.path.isfile("model/" + rm_file):
        os.remove("model/" + rm_file)
        
# Embedded Query Expansion
m_list = np.linspace(4, 4, num=1)
m = 1
interpolated_aplpha_list = np.linspace(0, 1.0, num=11)
word2vec = word2vec_model.word2vec_model(word_emb_path)

embd = EmbeddedBased(query_wordcount, collection, word2vec)
evaluate_model = EvaluateModel(relevance_path)
EQE1 = []
EQE2 = []
print "Embedded..."

tmp_eqe1 = embd.embedded_query_expansion_ci(0.4, 4)
tmp_eqe2 = embd.embedded_query_expansion_qi(0.4, 4)
tmp_eqe1 = ProcDoc.modeling(tmp_eqe1, background_model, query_lambda)
tmp_eqe2 = ProcDoc.modeling(tmp_eqe2, background_model, query_lambda)
EQE1.append([ProcDoc.dict2np(tmp_eqe1), tmp_eqe1])
EQE2.append([ProcDoc.dict2np(tmp_eqe2), tmp_eqe2])

Pickle.dump(EQE1, open("model/eqe1_10.pkl", "wb"), True)
Pickle.dump(EQE2, open("model/eqe2_10.pkl", "wb"), True)
'''
EQE1 = Pickle.load(open("model/eqe1_10.pkl", "rb"))
EQE2 = Pickle.load(open("model/eqe2_10.pkl", "rb"))
'''
# query process
print "query ..."
query_docs_point_fb = {}
query_model_fb = {}
mAP_list = []
for eqe_list in EQE2:
    query_model, query_model_dict = eqe_list
    qry_mdl, qry_IDs = query_model
    for step in range(2):
        # kl divergence
        query_result = np.dot(qry_mdl, np.log(doc_mdl.T))
        result = np.argsort(-query_result, axis = 1)
        query_docs_point_dict = defaultdict(dict)

        for q_idx in xrange(len(qry_IDs)):
            docs_ranking = []
            for doc_idx in result[q_idx]:
                docs_ranking.append(doc_IDs[doc_idx])
            query_docs_point_dict[qry_IDs[q_idx]] = docs_ranking
        mAP = evaluate_model.mAP(query_docs_point_dict)
        mAP_list.append(mAP)
        print "mAP"
        print mAP
        
        if step < 1:
            # save one shot result
            Pickle.dump(query_model_dict, open("model/query_model.pkl", "wb"), True)
            Pickle.dump(query_docs_point_dict, open("model/query_docs_point_dict.pkl", "wb"), True)
        # load one shot result
        query_docs_point_fb = Pickle.load(open("model/query_docs_point_dict.pkl", "rb"))
        query_model_fb = Pickle.load(open("model/query_model.pkl", "rb"))
            
        [qry_mdl, qry_IDs] = Expansion.feedback(query_docs_point_fb, query_model_fb, dict(doc_unigram), dict(doc_wordcount), dict(general_model), dict(background_model), 10)       
        
   
print np.argmax(np.array(mAP_list), axis = 0), mAP_list[np.argmax(np.array(mAP_list), axis = 0)]
# plot_diagram.plotList(m_list, mAP_list, "Conditional Independence of Query Terms", "mAP")


