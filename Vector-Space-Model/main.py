#!/usr/bin/env python
import numpy as np
import sys
sys.path.append("../Tools")

import ProcDoc
import Evaluate
from CommonPath import CommonPath
import Statistical 
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--is_training",
                     default="False",
                     type=str2bool,
                     required=False)

parser.add_argument("--is_short",
                     default="False",
                     type=str2bool,
                     required=False)

parser.add_argument("--is_spoken",
                     default="False",
                     type=str2bool,
                     required=False)

args = parser.parse_args()
is_training = args.is_training
is_short = args.is_short
is_spoken = args.is_spoken

path = CommonPath(is_training, is_short, is_spoken)
log_filename = path.getLogFilename()
qry_path = path.getQryPath()
doc_path = path.getDocPath()
rel_path = path.getRelPath()

dict_path = path.getDictPath()
bg_path = path.getBGPath()

print("Vector-Space-Model")
# read relevant set for queries and documents
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

# TF-IDF
print("TF-IDF")
[qry_mdl_np, doc_mdl_np] = Statistical.TFIDF(qry_mdl_np, doc_mdl_np)

# Cosine Similarity
# L2-normalize
qry_mdl_np = Statistical.l2Norm(qry_mdl_np)
doc_mdl_np = Statistical.l2Norm(doc_mdl_np)

print("Retrieval")
# Dot Product
results = np.argsort(-np.dot(qry_mdl_np, doc_mdl_np.T), axis = 1)

qry_docs_ranking = {}
for q_idx, q_ID in enumerate(qry_IDs):
    docs_ranking = []
    for doc_idx in results[q_idx]:
        docs_ranking.append(doc_IDs[doc_idx])
    qry_docs_ranking[q_ID] = docs_ranking

mAP = eval_mdl.mAP(qry_docs_ranking)
print(mAP)
