#!/usr/bin/env python
import numpy as np
import sys
sys.path.append("../Tools")

import ProcDoc
import Evaluate
from CommonPath import CommonPath  

is_training = False
is_short = True
is_spoken = False

is_training = False
is_short = False
is_spoken = False
alpha = 0.8
beta = 0.4

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

# Preprocess
qry_file = ProcDoc.readFile(qry_path)
doc_file = ProcDoc.readFile(doc_path)

# Term Frequency
qry_mdl_dict = ProcDoc.qryPreproc(qry_file, rel_set)
doc_mdl_dict = ProcDoc.docPreproc(doc_file)

# Unigram
qry_unimdl_dict = ProcDoc.unigram(qry_mdl_dict)
doc_unimdl_dict = ProcDoc.unigram(doc_mdl_dict)

# Convert dictionary to numpy array (feasible to compute)
qry_mdl_np, qry_IDs = ProcDoc.dict2npSparse(qry_unimdl_dict)
doc_mdl_np, doc_IDs = ProcDoc.dict2npSparse(doc_unimdl_dict)

bg_mdl_np = ProcDoc.readBGnp(bg_path)

# Smoothing
for doc_idx in range(doc_mdl_np.shape[0]):
    doc_mdl_np[doc_idx] = (1-alpha) * doc_mdl_np[doc_idx] + alpha * bg_mdl_np

# Smoothing
for qry_idx in range(qry_mdl_np.shape[0]):
    qry_mdl_np[qry_idx] = (1-beta) * qry_mdl_np[qry_idx] + beta * bg_mdl_np

results = np.argsort(-np.dot(qry_mdl_np, np.log(doc_mdl_np.T)), axis = 1)

qry_docs_ranking = {}
for q_idx, q_ID in enumerate(qry_IDs):
    docs_ranking = []
    for doc_idx in results[q_idx]:
        docs_ranking.append(doc_IDs[doc_idx])
    qry_docs_ranking[q_ID] = docs_ranking

mAP = eval_mdl.mAP(qry_docs_ranking)
print mAP
