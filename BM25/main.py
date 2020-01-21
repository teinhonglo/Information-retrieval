#!/usr/bin/env python
import numpy as np
import sys
sys.path.append("../Tools")

import ProcDoc
import Evaluate
from CommonPath import CommonPath
import Statistical 

# Test Condition
is_training = False
is_short = False
is_spoken = False
# Parameters of BM25
k1 = 2.0
b = 0.75

path = CommonPath(is_training, is_short, is_spoken)
log_filename = path.getLogFilename()
qry_path = path.getQryPath()
doc_path = path.getDocPath()
rel_path = path.getRelPath()

dict_path = path.getDictPath()
bg_path = path.getBGPath()

# Read relevant set for queries and documents
eval_mdl = Evaluate.EvaluateModel(rel_path, is_training)
rel_set = eval_mdl.getAset()

# Preprocess for queries and documents
qry_file = ProcDoc.readFile(qry_path)
doc_file = ProcDoc.readFile(doc_path)

# Term Frequency
qry_mdl_dict = ProcDoc.qryPreproc(qry_file, rel_set)
doc_mdl_dict = ProcDoc.docPreproc(doc_file)

# Convert dictionary to numpy array (feasible to compute)
qry_mdl_np, qry_IDs = ProcDoc.dict2npSparse(qry_mdl_dict)
doc_mdl_np, doc_IDs = ProcDoc.dict2npSparse(doc_mdl_dict)

# Document frequency
print("Document frequency")
idf = Statistical.IDF(doc_mdl_np)
doc_len = Statistical.compLenAve(doc_mdl_np)

# BM25
print("BM25")
ranking = np.zeros((qry_mdl_np.shape[0], doc_mdl_np.shape[0]))
# score(Q, D)
for q_idx, q_vec in enumerate(qry_mdl_np):
    qw_idx = np.where(q_vec != 0)
    for d_idx, d_vec in enumerate(doc_mdl_np):
        nominator = idf[qw_idx] * d_vec[qw_idx] * (k1 + 1)
        denominator = d_vec[qw_idx] + (k1 * (1 - b + b * doc_len[d_idx]))
        ranking[q_idx][d_idx] = (nominator / denominator).sum(axis = 0)
    
# Ranking
results = np.argsort(-ranking, axis = 1)

qry_docs_ranking = {}
for q_idx, q_ID in enumerate(qry_IDs):
    docs_ranking = []
    for doc_idx in results[q_idx]:
        docs_ranking.append(doc_IDs[doc_idx])
    qry_docs_ranking[q_ID] = docs_ranking

mAP = eval_mdl.mAP(qry_docs_ranking)
print mAP
