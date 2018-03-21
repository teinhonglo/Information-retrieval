#!/usr/bin/env python3
import sys
import Statistical

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
        
    def __cosineSimilarity(self):
        qry = self.qry
        doc = self.doc
        doc_len = self.doc_len
        for q_id, q_cont in qry.items():
            for d_id, d_cont in doc.items():
                