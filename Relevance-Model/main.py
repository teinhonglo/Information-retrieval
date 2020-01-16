#!/usr/bin/env python
import numpy as np
import sys
sys.path.append("../Tools")

import ProcDoc
import Evaluate
import RelevanceModel as RM3

is_training = False
is_short = False
is_spoken = False

if is_training:
    qry_path = "../Corpus/TDT2/Train/XinTrainQryTDT2/QUERY_WDID_NEW"
    rel_path = "../Corpus/TDT2/Train/QDRelevanceTDT2_forHMMOutSideTrain"
else:
    if is_short:
        qry_path = "../Corpus/TDT2/QUERY_WDID_NEW_middle"
    else:
        qry_path = "../Corpus/TDT2/QUERY_WDID_NEW"
    rel_path = "../Corpus/TDT2/AssessmentTrainSet/AssessmentTrainSet.txt"

if is_spoken:
    doc_path = "../Corpus/TDT2/Spoken_Doc"
else:
    doc_path = "../Corpus/TDT2/SPLIT_DOC_WDID_NEW"

dict_path = "../Corpus/TDT2/LDC_Lexicon.txt"
bg_path = "../Corpus/background"

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

# KL divergence
results = np.argsort(-np.dot(qry_mdl_np, np.log(doc_mdl_np.T)), axis = 1)

# one-stage retrieval
qry_docs_ranking = {}
for q_idx, q_ID in enumerate(qry_IDs):
    docs_ranking = []
    for doc_idx in results[q_idx]:
        docs_ranking.append(doc_IDs[doc_idx])
    qry_docs_ranking[q_ID] = docs_ranking

#eval_mdl = EvaluateModel(rel_path, isTraining)
mAP = eval_mdl.mAP(qry_docs_ranking)
print mAP

# Relevance Feedback
RM_mdl_np = RM3.feedback(qry_IDs, qry_mdl_np, doc_IDs, doc_mdl_np, bg_mdl_np, qry_docs_ranking, 9)
qry_mdl_np = 1.0 * RM_mdl_np + 0.0 * qry_mdl_np

# KL divergence
results = np.argsort(-np.dot(qry_mdl_np, np.log(doc_mdl_np.T)), axis = 1)

# second-stage retrieval
qry_docs_ranking = {}
for q_idx, q_ID in enumerate(qry_IDs):
    docs_ranking = []
    for doc_idx in results[q_idx]:
        docs_ranking.append(doc_IDs[doc_idx])
    qry_docs_ranking[q_ID] = docs_ranking

mAP = eval_mdl.mAP(qry_docs_ranking)
print mAP
