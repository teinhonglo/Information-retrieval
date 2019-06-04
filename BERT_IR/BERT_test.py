# -*- coding: utf-8 -*-
#!/usr/bin/env python
import numpy as np
import sys
sys.path.append("../Tools")

import ProcDoc
import Evaluate

is_training = False
is_short = False
is_spoken = False

if is_training:
    qry_path = "../Corpus/TDT2/Train/XinTrainQryTDT2/QUERY_WDID_NEW"
    rel_path = "../Corpus/TDT2/Train/QDRelevanceTDT2_forHMMOutSideTrain"
else:
    if is_short:
        qry_path = "../Corpus/TDT2/QUERY_WDID_NEW_middle"
        bert_results = "exp/TDT2_exp/pointwise_ranking_results.short.txt"
    else:
        qry_path = "../Corpus/TDT2/QUERY_WDID_NEW"
        bert_results = "exp/TDT2_exp/pointwise_ranking_results.txt"
    rel_path = "../Corpus/TDT2/AssessmentTrainSet/AssessmentTrainSet.txt"

if is_spoken:
   doc_path = "../Corpus/TDT2/Spoken_Doc"
else:
   doc_path = "../Corpus/TDT2/SPLIT_DOC_WDID_NEW"

dict_path = "../Corpus/TDT2/LDC_Lexicon.txt"
bg_path = "../Corpus/background"

# Mean Max Normalization
def MMNorm(rawdata):
    norm = np.zeros((len(qry_IDs), len(doc_IDs)))
    row=0
    col=0
    for query in rawdata:
        for doc in query:
           norm[row][col] = (doc - np.min(query) ) / (np.max(query) - np.min(query))
           col+=1
        row+=1
        col=0
    return norm

# Bert results
def BERTResults(results_path, qry_IDs, doc_IDs):
    rel_mat = np.zeros((len(qry_IDs), len(doc_IDs)))
    import codecs
    with codecs.open(results_path, 'r', encoding='utf-8') as rf:
        for idx, line in enumerate(rf.readlines()):
            info = line.split("\n")[0].split(",")
            qry_idx = qry_IDs.index(info[0])
            doc_idx = doc_IDs.index(info[1])
            irrel_score = float(info[2])
            rel_score = float(info[3])
            rel_mat[qry_idx, doc_idx] = rel_score - irrel_score
    return rel_mat

# read relevant set for queries and documents
eval_mdl = Evaluate.EvaluateModel(rel_path, is_training)
rel_set = eval_mdl.getAset()

alpha = 0.8
beta = 0.4

qry_file = ProcDoc.readFile(qry_path)
doc_file = ProcDoc.readFile(doc_path)

qry_mdl_dict = ProcDoc.qryPreproc(qry_file, rel_set)
doc_mdl_dict = ProcDoc.docPreproc(doc_file)

qry_unimdl_dict = ProcDoc.unigram(qry_mdl_dict)
doc_unimdl_dict = ProcDoc.unigram(doc_mdl_dict)

qry_mdl_np, qry_IDs = ProcDoc.dict2npSparse(qry_unimdl_dict)
doc_mdl_np, doc_IDs = ProcDoc.dict2npSparse(doc_unimdl_dict)

bg_mdl_np = ProcDoc.readBGnp(bg_path)


# smoothing
for doc_idx in range(doc_mdl_np.shape[0]):
    doc_mdl_np[doc_idx] = (1-alpha) * doc_mdl_np[doc_idx] + alpha * bg_mdl_np

# smoothing
for qry_idx in range(qry_mdl_np.shape[0]):
    qry_mdl_np[qry_idx] = (1-beta) * qry_mdl_np[qry_idx] + beta * bg_mdl_np

kl_div = np.dot(qry_mdl_np, np.log(doc_mdl_np.T))
norm = MMNorm(kl_div)
results = np.argsort(-norm, axis = 1)
#print(norm)

#BERT_rel_mat = BERTResults(bert_results, qry_IDs, doc_IDs)
#results = np.argsort(-BERT_rel_mat, axis = 1)

qry_docs_ranking = {}
for q_idx, q_ID in enumerate(qry_IDs):
    docs_ranking = []
    for doc_idx in results[q_idx]:
        docs_ranking.append(doc_IDs[doc_idx])
    qry_docs_ranking[q_ID] = docs_ranking

#eval_mdl = EvaluateModel(rel_path, isTraining)
mAP = eval_mdl.mAP(qry_docs_ranking)

#print(qry_docs_ranking)
print mAP

