#!/usr/bin/env python
import numpy as np
import sys
import nn_Predict as nn_model
sys.path.append("../Tools")

import ProcDoc
import Evaluate

is_training = True
is_short = False
is_spoken = True
model_name = "SSWLM_E"
results_file = "NRM_rank_" + model_name
nn_method = "NN_Model/TDT2/RLE_" + model_name
topN = 10

if is_training:
    qry_path = "../Corpus/TDT2/Train/XinTrainQryTDT2/QUERY_WDID_NEW"
    rel_path = "../Corpus/TDT2/Train/QDRelevanceTDT2_forHMMOutSideTrain"
    results_file += "_train"
else:
    if is_short:
        qry_path = "../Corpus/TDT2/QUERY_WDID_NEW_middle"
        results_file += "_short"
    else:
        qry_path = "../Corpus/TDT2/QUERY_WDID_NEW"
        results_file += ""
    rel_path = "../Corpus/TDT2/AssessmentTrainSet/AssessmentTrainSet.txt"

if is_spoken:
    doc_path = "../Corpus/TDT2/Spoken_Doc"
    nn_method += "_S.h5"
    results_file += "_spk.txt"
    rel_lambda = 0.6
else:
    doc_path = "../Corpus/TDT2/SPLIT_DOC_WDID_NEW"
    nn_method += ".h5"
    results_file += ".txt"
    rel_lambda = 0.5

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

# origin query model
qry_mdl_np, qry_IDs = ProcDoc.dict2npSparse(qry_unimdl_dict)
# refine query model
ref_qry_mdl_np, qry_IDs = ProcDoc.dict2npSparse(qry_unimdl_dict)
doc_mdl_np, doc_IDs = ProcDoc.dict2npSparse(doc_unimdl_dict)

NRM_mdl_np = nn_model.predict(nn_method, qry_mdl_np)

bg_mdl_np = ProcDoc.readBGnp(bg_path)

# smoothing
for doc_idx in range(doc_mdl_np.shape[0]):
    doc_mdl_np[doc_idx] = (1-alpha) * doc_mdl_np[doc_idx] + alpha * bg_mdl_np

# smoothing
for qry_idx in range(qry_mdl_np.shape[0]):
    qry_mdl_np[qry_idx] = (1-beta) * qry_mdl_np[qry_idx] + beta * bg_mdl_np

# query results
with open(results_file, "wb") as writer:
    ''' query smoothing '''    
    for qry_idx in range(ref_qry_mdl_np.shape[0]):
        ref_qry_mdl_np[qry_idx] = (1 - rel_lambda) * qry_mdl_np[qry_idx]  + rel_lambda * NRM_mdl_np[qry_idx]
    ranking = np.dot(ref_qry_mdl_np, np.log(doc_mdl_np.T))
    results = np.argsort(-ranking, axis = 1)

    qry_docs_ranking = {}
    for q_idx, q_ID in enumerate(qry_IDs):
        docs_ranking = []
        for doc_idx in results[q_idx]:
            docs_ranking.append(doc_IDs[doc_idx])
            writer.write(q_ID + "," + doc_IDs[doc_idx] + ",0," + str(ranking[q_idx][doc_idx]) + "\n")
        qry_docs_ranking[q_ID] = docs_ranking

    #eval_mdl = EvaluateModel(rel_path, isTraining)
    mAP = eval_mdl.mAP(qry_docs_ranking)
    print rel_lambda, mAP

# assessment results
with open("pseudo_" + str(topN) + "_" + results_file, "wb") as writer:
    ''' query smoothing '''    
    for qry_idx in range(ref_qry_mdl_np.shape[0]):
        ref_qry_mdl_np[qry_idx] = (1 - rel_lambda) * qry_mdl_np[qry_idx]  + rel_lambda * NRM_mdl_np[qry_idx]
    ranking = np.dot(ref_qry_mdl_np, np.log(doc_mdl_np.T))
    results = np.argsort(-ranking, axis = 1)

    qry_docs_ranking = {}
    for q_idx, q_ID in enumerate(qry_IDs):
        writer.write("Query " + str(q_idx) + " " + q_ID + " " + str(topN) + "\n")
        docs_ranking = []
        for count, doc_idx in enumerate(results[q_idx]):
            docs_ranking.append(doc_IDs[doc_idx])
            writer.write(doc_IDs[doc_idx] + "\n")
            if count == topN:
                break
        writer.write("\n")
        qry_docs_ranking[q_ID] = docs_ranking

    #eval_mdl = EvaluateModel(rel_path, isTraining)
    mAP = eval_mdl.mAP(qry_docs_ranking)
    print rel_lambda, mAP
