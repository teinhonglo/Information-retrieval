#!/usr/bin/env python3
import os, sys
currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, '..'))
sys.path.append(rootDir)
sys.path.append(rootDir + "/../Tools")

import argparse
from argparse import RawTextHelpFormatter
import Statistical
from Evaluate import EvaluateModel
import ProcDoc
import timeit
import numpy as np
import cPickle as pickle

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class VSM(object):
    def __init__(self, qry_path = None, rel_path = None, isTraining = True, doc_path = None):
        # default training step
        if qry_path == None: 
            qry_path = "../Corpus/TDT2/Train/XinTrainQryTDT2/QUERY_WDID_NEW"
        if doc_path == None:
            doc_path = "../Corpus/TDT2/SPLIT_DOC_WDID_NEW"
        if rel_path == None:
            rel_path = "../Corpus/TDT2/Train/QDRelevanceTDT2_forHMMOutSideTrain"
        self.vocab_size = 51253
        # relevance set
        self.rel_set = ProcDoc.readRELdict(rel_path, isTraining)
        self.evaluate_model = EvaluateModel(rel_path, isTraining)
        
        # read documents
        doc = ProcDoc.readFile(doc_path)
        self.doc = ProcDoc.docPreproc(doc)
        self.doc_len = Statistical.compLenAcc(self.doc)
		
        # read queries
        qry = ProcDoc.readFile(qry_path)
        self.qry_tf = ProcDoc.qryPreproc(qry, self.rel_set)
        self.qry_len = Statistical.compLenAcc(self.qry_tf)
        [self.qry, self.doc] = Statistical.TFIDF(self.qry_tf, self.doc, self.qry_len, self.doc_len)
        
        # dict to numpy
        self.qry_tf, self.qry_tf_IDs = self.__dict2np(self.qry_tf)
        self.qry, self.qry_IDs = self.__dict2np(self.qry, self.qry_tf_IDs)
        self.doc, self.doc_IDs = self.__dict2np(self.doc)
        
        # precompute len(document)
        for idx, d_len in enumerate(np.sqrt((self.doc ** 2).sum(axis = 1))):
            self.doc[idx] = self.doc[idx] / d_len
  
    def evaluate(self, qry_docs_ranking):
        evaluate_model = self.evaluate_model
        mAP = evaluate_model.mAP(qry_docs_ranking)
        return mAP
       
    def cosineFast(self, qry = None, qry_IDs = None):
        # First Step
        if qry is None or qry_IDs is None:
            qry, qry_IDs = self.qry, self.qry_IDs 
        doc, doc_IDs = self.doc, self.doc_IDs
        # cosine similarity
        result = np.argsort(-np.dot(qry, doc.T), axis = 1)
        # prepare ranking list
        qry_docs_ranking = {}
        for q_idx, q_ID in enumerate(qry_IDs):
            docs_ranking = []
            for doc_idx in result[q_idx]:
                docs_ranking.append(doc_IDs[doc_idx])
            qry_docs_ranking[q_ID] = docs_ranking
        return qry_docs_ranking 
     
    def cosineFast_(self, qry, qry_IDs, doc, doc_IDs):
        # cosine similarity
        result = np.argsort(-np.dot(qry, doc.T), axis = 1)
        # prepare ranking list
        qry_docs_ranking = {}
        for q_idx, q_ID in enumerate(qry_IDs):
            docs_ranking = []
            for doc_idx in result[q_idx]:
                docs_ranking.append(doc_IDs[doc_idx])
            qry_docs_ranking[q_ID] = docs_ranking
        return qry_docs_ranking 
    
    def cosineSlow(self, qry = None, qry_IDs = None):
        # First Step
        if qry is None or qry_IDs is None:
            qry, qry_IDs = self.qry, self.qry_IDs 
        doc, doc_IDs = self.doc, self.doc_IDs
        qry_docs_ranking = {}
        for q_idx, q_ID in enumerate(qry_IDs):
            # cosine similarity
            result = np.argsort(-(qry[q_idx] * doc).sum(axis = 1))
            docs_ranking = []
            # prepare ranking list
            for doc_idx in result:
                docs_ranking.append(doc_IDs[doc_idx])
            qry_docs_ranking[q_ID] = docs_ranking
        return qry_docs_ranking
    
    def PRF(self, qry_docs_ranking, topN, alpha = 0.3):
        # Pseudo Relevance Feedback
        qry, qry_IDs = np.copy(self.qry), self.qry_IDs 
        doc, doc_IDs = self.doc, self.doc_IDs
        for q_idx, q_ID in enumerate(qry_IDs):
            ext_vec = np.zeros(self.vocab_size)
            # topN relevance documents
            for d_ID in qry_docs_ranking[q_ID][:topN]:
                d_idx = doc_IDs.index(d_ID)
                ext_vec += doc[d_idx]
            # intepolated original queries
            qry[q_idx] = alpha * qry[q_idx] + (1 - alpha) * ext_vec
        return qry, qry_IDs, doc, doc_IDs
        
    def __dict2np(self, ori_dict, IDs_list = None):
        num_tar = len(list(ori_dict.keys()))
        obj_vec = np.zeros((num_tar, self.vocab_size))
        if IDs_list is None:
            IDs_list = list(ori_dict.keys())
        for idx, o_id in enumerate(IDs_list):
            for o_wid, o_wc in ori_dict[o_id].items():
                obj_vec[idx][o_wid] = o_wc
        return obj_vec, IDs_list
    
    def saveMdl(self, obj_qry, data_path):
        # save list
        with open(data_path + "/qry_IDs.pkl", "wb") as f: pickle.dump(self.qry_IDs, f, True)
        with open(data_path + "/doc_IDs.pkl", "wb") as f: pickle.dump(self.doc_IDs, f, True)
        # save numpy
        np.save(data_path + "/x_qry_tf_mdl.npy", self.qry_tf)
        np.save(data_path + "/x_qry_mdl.npy", self.qry)
        np.save(data_path + "/doc_mdl.npy", self.doc)
        # objective model
        np.save(data_path + "/y_qry_mdl.npy", obj_qry)
        
        
def main(args):
    '''
    qry_path = "../Corpus/TDT2/QUERY_WDID_NEW"
    rel_path = "../Corpus/TDT2/AssessmentTrainSet/AssessmentTrainSet.txt"
    data_path = "data/Test"
    is_train = False
    doc_path = None
    '''
    qry_path = args['qry_dataset']
    rel_path = args['rel_dataset']
    data_path = args['data_storage']
    is_train = args['is_train'] 
    doc_path = None
    model = VSM(qry_path, rel_path, is_train, doc_path)
    qry_docs_ranking = model.cosineFast()
    best_qry_mdl = 0
    best_mAP = 0
    for i in range(1, 15, 1):
        qry, qry_IDs, doc, doc_IDs = model.PRF(qry_docs_ranking, i)
        qry_docs_ranking_ = model.cosineFast(qry, qry_IDs)
        mAP = model.evaluate(qry_docs_ranking_)
        print(i, mAP)
        if mAP > best_mAP:
            best_qry_mdl = np.copy(qry)
            best_mAP = mAP
    print("best", best_mAP)
    model.saveMdl(best_qry_mdl, data_path)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""This program runs NRM-VSM on a prepared corpus.\n
                                                    sample argument setting is as follows:\n
                                                    python local/VSM.py --qry_dataset ../Corpus/TDT2/QUERY_WDID_NEW --rel_dataset ../Corpus/TDT2/AssessmentTrainSet/AssessmentTrainSet.txt --data_storage data/Test --is_train True
    """, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-qry', '--qry_dataset', help='query dataset', required=True)
    parser.add_argument('-rel', '--rel_dataset', help='relevant dataset', required=True)
    parser.add_argument('-data', '--data_storage', help='document dataset', required=True)
    parser.add_argument('-it', '--is_train', type=str2bool, nargs='?', const=True, help='Steps', required=True)
    args = vars(parser.parse_args())
    main(args)
