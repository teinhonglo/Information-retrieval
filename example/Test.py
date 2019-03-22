#!/usr/bin/env python
import numpy as np
import sys
sys.path.append("../Tools")

import ProcDoc
import Evaluate

qry_path = "../Corpus/TDT2/QUERY_WDID_NEW"
doc_path = "../Corpus/TDT2/SPLIT_DOC_WDID_NEW"
bg_path = "../Corpus/background"
rel_path = ""
isTraining = False
alpha = 0.8
beta = 0.4

qry_file = ProcDoc.readFile(qry_path)
doc_file = ProcDoc.readFile(doc_path)

qry_mdl_dict = ProcDoc.qryPreproc(qry_file)
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

results = np.argsort(-np.dot(qry_mdl_np, np.log(doc_mdl_np.T)), axis = 1)

qry_docs_ranking = {}
for q_idx, q_ID in enumerate(qry_IDs):
    docs_ranking = []
    for doc_idx in results[q_idx]:
        docs_ranking.append(doc_IDs[doc_idx])
    qry_docs_ranking[q_ID] = docs_ranking

#eval_mdl = EvaluateModel(rel_path, isTraining)
eval_mdl = Evaluate.EvaluateModel()
mAP = eval_mdl.mAP(qry_docs_ranking)
print mAP
