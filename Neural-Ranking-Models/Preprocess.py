import sys
sys.path.append("../Tools")
import numpy as np
np.random.seed(5566)
from sklearn import preprocessing
import ProcDoc

class InputDataProcess(object):
    def __init__(self, len_feats = 10, 
                 type_rank = "pointwise", type_feat = "sparse"
                 query_path = None, document_path = None, corpus = "TDT2"):
        #ranks = ["pointwise", "pairwise"]
        #feats = ["spare", "emb"]
        res_pos = True
        self.num_vocab = 51253
        self.num_feats = len_feats
        self.type_rank = type_rank
        self.type_feat = type_feat
        # qry and doc
        if query_path == None: 
            query_path = "../Corpus/" + corpus + "/Train/XinTrainQryTDT2/QUERY_WDID_NEW"
        if document_path == None:
            document_path = "../Corpus/" + corpus + "/SPLIT_DOC_WDID_NEW"
        
        # relevancy set
        self.hmm_training_set = ProcDoc.readRELdict()
        
	# read document, reserve position
        doc = ProcDoc.readFile(document_path)
        self.doc = ProcDoc.docPreproc(doc, res_pos)
		
        # read query, reserve position
        qry = ProcDoc.readFile(query_path)
        self.qry = ProcDoc.qryPreproc(qry, self.hmm_training_set, res_pos)        
        
        # generate h featrues
        self.input_feats = self.__genFeature(self.num_feats)
        
    def genTrainValidSet(self, percent = None, isTest = False):
        print "generate training set and validation set"
        if percent == None: percent = 80
        qry = self.qry
        doc = self.doc
        total_qry = len(qry.keys())
        total_doc = len(doc.keys())
        hmm_training_set = self.hmm_training_set
        labels = {}
        partition = {'train': [], 'validation': []}
        part_answer = {'train': [], 'validation': []}	
        # relevance between queries and documents
        # labels = answer ----------------------------------------------------
        # if pointwise and (sparse or embedding) key = "q_d"
        # if pairwise (d+, d-)and (sparse or embedding) k = "q_d+_d-"
        # ---------------------------------------------------------------------
        # partition
        ID_list = labels.keys()
        total = len(ID_list)
        num_of_train = total * (percent / 100)
        num_of_valid = total - num_of_train
        # shuffle
        np.random.shuffle(ID_list)
        # training set
        partition['train'] = [id for id in ID_list[:num_of_train]]
        # validation set
        partition['validation'] = [id for id in ID_list[num_of_train:]]      
        return [partition, labels]
    
    def __genFeature(self, len_feats):
        type_feats = self.type_feats
        type_rank = self.type_rank
        print("generate " + type_feats + " features, type of the rank " + type_rank)
        qry = self.qry
        doc = self.doc
        # --------------------------------------------------------------------------
        # if pointwise and (sparse or embedding) input = 1 (i.e., concat qry and doc)
        # if pairwise (d+, d-)and (sparse and embedding) input = 2 (e.g., q_d+, q_d-)
        # --------------------------------------------------------------------------
        return feats
		
if __name__ == "__main__":
    a = InputDataProcess()


