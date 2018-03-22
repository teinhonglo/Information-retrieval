#!/usr/bin/env python3
import sys
sys.path.append("../../Tools")
import Statistical
from Evaluate import EvaluateModel
import ProcDoc
import numpy as np

class VSM(object):
    def __init__(self, qry_path = None, doc_path = None, rel_path = None, isTraining = True):
        # default training step
        if qry_path == None: 
            qry_path = "../../Corpus/TDT2/Train/XinTrainQryTDT2/QUERY_WDID_NEW"
        if doc_path == None:
            doc_path = "../../Corpus/TDT2/SPLIT_DOC_WDID_NEW"
        if rel_path == None:
            rel_path = "../../Corpus/TDT2/Train/QDRelevanceTDT2_forHMMOutSideTrain"
        
        # relevance set
        self.rel_set = ProcDoc.readRELdict(rel_path, isTraining)
        print len(list(self.rel_set.keys()))
        evaluate_model = EvaluateModel(rel_path, isTraining)
        
        # read document
        doc = ProcDoc.readFile(doc_path)
        self.doc = ProcDoc.docPreproc(doc)
        self.doc_len = Statistical.docLen(self.doc)
		
        # read query
        qry = ProcDoc.readFile(qry_path)
        self.qry = ProcDoc.qryPreproc(qry, self.rel_set)
        [self.qry, self.doc] = Statistical.TFIDF(self.qry, self.doc)
    
    def evaluate(self, qry_docs_ranking):
        evaluate_model = self.evaluate_model
        mAP = evaluate_model.mAP(qry_docs_ranking)
        return mAP
        
    def cosineFast(self):
        qry, qry_IDs = self.__dict2np(self.qry)
        doc, doc_IDs = self.__dict2np(self.doc)
        result = np.argsort(np.dot(qry, doc.T), axis = 1)
        qry_docs_ranking = {}
        for q_idx in xrange(len(qry_IDs)):
            docs_ranking = []
            for doc_idx in result[q_idx]:
                docs_ranking.append(doc_IDs[doc_idx])
            qry_docs_ranking[qry_IDs[q_idx]] = docs_ranking
        return qry_docs_ranking
    
    def cosineSlow(self):
        qry, qry_IDs = self.__dict2np(self.qry)
        doc, doc_IDs = self.__dict2np(self.doc)
        qry_docs_ranking = {}
        for q_idx, q_ID in xrange(len(qry_IDs)):
            result = np.argsort(np.dot(qry[q_ID], doc.T), axis = 1)
            docs_ranking = []
            for doc_idx in result:
                docs_ranking.append(doc_IDs[doc_idx])
            qry_docs_ranking[q_ID] = docs_ranking
        return qry_docs_ranking
    
    def __dict2np(self, ori_dict, vocab_size = 51253):
        num_tar = len(list(ori_dict.keys()))
        tar_vec = np.zeros((num_tar, vocab_size))
        IDs_list = list(ori_dict.keys())
        for idx, o_id in enumerate(IDs_list):
            for o_wid, o_wc in ori_dict[o_id].items():
                tar_vec[idx][o_wid] = o_wc
        return tar_vec, IDs_list

def main():
    qry_path = None
    doc_path = None
    rel_path = None
    isTraining = True
    model = VSM(qry_path, doc_path, rel_path, isTraining)
    qry_docs_ranking = model.cosineFast()
    mAP = model.evaluate(qry_docs_ranking)
    print mAP
        
if __name__ == "__main__":
    main()