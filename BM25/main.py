#!/usr/bin/env python3
import numpy as np
import sys
import argparse
sys.path.append("../Tools")

import ProcDoc
import Evaluate
from CommonPath import CommonPath
import Statistical

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

# Test Condition
args = parser.parse_args()
is_training = args.is_training
is_short = args.is_short
is_spoken = args.is_spoken
# Parameters of BM25
idf_t = 6
b = 0.75
k1 = 2.0
k3 = 2.0

path = CommonPath(is_training, is_short, is_spoken)
log_filename = path.getLogFilename()
qry_path = path.getQryPath()
doc_path = path.getDocPath()
rel_path = path.getRelPath()

dict_path = path.getDictPath()
bg_path = path.getBGPath()

print("Best Match 25")
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
qry_mdl_np_, qry_IDs = ProcDoc.dict2npSparse(qry_mdl_dict)
doc_mdl_np_, doc_IDs = ProcDoc.dict2npSparse(doc_mdl_dict)

# Document frequency
print("Document frequency")
avg_len = Statistical.avgLen(doc_mdl_np_)

idf = Statistical.IDF(doc_mdl_np_)
# BM25
[qry_bm25, doc_bm25] = Statistical.BM25(qry=qry_mdl_np_, doc=doc_mdl_np_, idf=idf, avg_len=avg_len, b=b, k1=k1, k3=k3, delta=0)
ranking = -np.dot(qry_bm25, doc_bm25.T)
results = np.argsort(ranking, axis=1)

qry_docs_ranking = {}
for q_idx, q_ID in enumerate(qry_IDs):
    docs_ranking = []
    for doc_idx in results[q_idx]:
        docs_ranking.append(doc_IDs[doc_idx])
    qry_docs_ranking[q_ID] = docs_ranking

mAP = eval_mdl.mAP(qry_docs_ranking)
print(mAP)
