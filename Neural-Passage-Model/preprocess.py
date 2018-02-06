import sys
sys.path.append("../tools")

import operator
import numpy as np
import ProcDoc
from collections import defaultdict
from math import log
import os

data = {}                # content of document (doc, content)
background_model = {}    # word count of 2265 document (word, number of words)
query = {}                # query

corpus = "TDT2"
document_path = "../Corpus/" + corpus + "/SPLIT_DOC_WDID_NEW"    
query_path = "../Corpus/" + corpus + "/XinTrainQryTDT3/QUERY_WDID_NEW"
test_query_path = "../Corpus/"+ corpus + "/XinTestQryTDT3/QUERY_WDID_NEW"
resPos = True

# read document, reverse position
doc = ProcDoc.read_file(document_path)
doc = ProcDoc.doc_preprocess(doc, resPos)

# read query, reserve position
query = ProcDoc.read_file(query_path)
query = ProcDoc.query_preprocess(query, resPos)

# read test lone query model, reserve postion
test_query = ProcDoc.read_file(query_path)
test_query = ProcDoc.query_preprocess(test_query, resPos)

# HMMTrainingSet
HMMTraingSetDict = ProcDoc.read_relevance_dict()
query_relevance = {}

# create passage matrix
query_model = []
qry_list = query.keys()
doc_list = doc.keys()
rel_qd_list = []
patMatAll = []
# passage model (q_length X d_length)
for q, q_cont in query.items():
    if q in HMMTraingSetDict:
		q_terms = q_cont.split()
		height = len(q_terms)
		for d, d_cont in doc.items():
			d_terms = d_cont.split()
			width = len(d_terms)
			psgMat = np.zeros(height, width)
			for q_idx in xrange(len(q_terms)):
				q_term = q_terms[q_idx]
				for d_idx in xrange(len(d_terms)):
					d_term = d_terms[d_idx]
					if q_term == d_term:
						psgMat[q_idx][d_idx] = 1
					else:	
						psgMat[q_idx][d_idx] = 0
			if d in HMMTraingSetDict[q]:
				rel_qd_list.append(1)
			else:
				rel_qd_list.append(-1)
        patMatAll.append(psgMat)
    else:
        qry_list.remove(q)

# list to numpy
qry_list = np.array(qry_list)
doc_list = np.array(doc_list)
rel_qd_list	= np.array(rel_qd_list)
# zero padding 
from keras.layers import ZeroPadding2D
patMatAll = ZeroPadding2D(np.array(patMatAll).astype(np.float32))
# save
np.save("passageModel.np", patMatAll)
np.save("rel_list.np", rel_qd_list)
np.save("qry_list.np", qry_list)
np.save("doc_list.np", doc_list)
