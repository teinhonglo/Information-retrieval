#!/usr/bin/env python3
import sys
sys.path.append("../Tools")
import Statistical
from Evaluate import EvaluateModel
import numpy as np

class VSM(Object):
    def __init__(self, qry_path = None, doc_path = None):
        if qry_path == None: 
            qry_path = "../Corpus/" + corpus + "/Train/XinTrainQryTDT2/QUERY_WDID_NEW"
        if doc_path == None:
            doc_path = "../Corpus/" + corpus + "/SPLIT_DOC_WDID_NEW"
        
        # relevance set
        self.hmm_training_set = ProcDoc.readRELdict()
        
        # read document
        doc = ProcDoc.readFile(doc_path)
        self.doc = ProcDoc.docPreproc(doc)
        self.doc_len = Statistical.docLen(self.doc)
		
        # read query
        qry = ProcDoc.readFile(qry_path)
        self.qry = ProcDoc.qryPreproc(qry, self.hmm_training_set)
        [self.qry, self.doc] = Statistical.TFIDF(qry, doc)
    
    def evaluate(self, qry_docs_ranking, rel_path = None):
        if rel_path == None:
            rel_path = "../Corpus/TDT2/Train/QDRelevanceTDT2_forHMMOutSideTrain"
        evaluate_model = EvaluateModel(rel_path, True)
        mAP = evaluate_model.mAP(qry_docs_ranking)
        return mAP
        
    def __cosineFast(self):
        qry, qry_IDs = __dict2np(self.qry_np)
        doc, doc_IDs = __dict2np(self.doc_np)
        result = np.argsort(-np.dot(qry, doc.T), axis = 1)
        qry_docs_ranking {}
        for q_idx in range(len(qry_IDs)):
            docs_ranking = []
            for doc_idx in result[q_idx]:
                docs_ranking.append(doc_IDs[doc_idx])
            qry_docs_ranking[qry_IDs[q_idx]] = docs_ranking
        return qry_docs_ranking
    
    def __dict2np(self, ori_dict, vocab_size = 51253):
        num_tar = len(list(ori_dict.keys()))
        tar_vec = np.zeros((num_tar, vocab_size))
        IDs_list = list(ori_dict.keys())
        for idx, o_id in enumerate(IDs_list):
            for o_wid, o_wc in ori_dict[o_id].items():
                tar_vec[idx][o_wid] = o_wc
        return num_tar, IDs_list