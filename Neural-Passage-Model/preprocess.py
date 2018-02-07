import sys
sys.path.append("../Tools")

import numpy as np
import gc
import ProcDoc

data = {}                # content of document (doc, content)
background_model = {}    # word count of 2265 document (word, number of words)
query = {}                # query

corpus = "TDT2"
document_path = "../Corpus/" + corpus + "/SPLIT_DOC_WDID_NEW"    
query_path = "../Corpus/" + corpus + "/Train/XinTrainQryTDT2/QUERY_WDID_NEW"
test_query_path = "../Corpus/"+ corpus + "/Train/XinTestQryTDT2/QUERY_WDID_NEW"
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
max_q = 0
max_d = 0
# create passage matrix
query_model = []
qry_doc_list = []
rel_qd_list = []
patMatAll = []
qry_length = 1794
doc_length = 2907
batch_size = 512
count = 0
# passage model (q_length X d_length)
for q, q_cont in query.items():
	if q in HMMTraingSetDict:
		q_terms = q_cont.split()
		for d, d_cont in doc.items():
			qry_doc_list.append([q, d])
			d_terms = d_cont.split()
			psgMat = np.zeros((qry_length, doc_length, 1))
			# create passage matrix
			for q_idx, q_term in enumerate(q_terms):
				for d_idx, d_term in enumerate(d_terms):
					# hit = 1, otherwise = 0
					if q_term == d_term:
						psgMat[q_idx][d_idx] = 1
					else:	
						psgMat[q_idx][d_idx] = 0
			if d in HMMTraingSetDict[q]:
				rel_qd_list.append(1)
			else:
				rel_qd_list.append(-1)
			count += 1
			patMatAll.append(np.copy(psgMat))
			
			if (count % batch_size) == 0:
				np.save("exp/trainPsg_" + str((count - batch_size) / batch_size) + ".npy", np.asarray(patMatAll))
				patMatAll = []
				np.save("exp/labels_" + str((count - batch_size) / batch_size) + ".npy", np.asarray(rel_qd_list))
				rel_qd_list = []
				np.save("exp/pair_" + str((count - batch_size) / batch_size) + ".npy", np.asarray(qry_doc_list))
				qry_doc_list = []
			print (str(count) + "/ 1812000")	
print (max_q, max_d)
# list to numpy
qry_list = np.array(qry_list)
doc_list = np.array(doc_list)
rel_qd_list	= np.array(rel_qd_list)
# zero padding 
#from keras.layers import ZeroPadding2D
#patMatAll = ZeroPadding2D(padding=(1, 1), np.array(patMatAll).astype(np.float32))
# save

'''
np.save("exp/passageModel.np", patMatAll)
np.save("exp/rel_list.np", rel_qd_list)
np.save("exp/qry_list.np", qry_list)
np.save("exp/doc_list.np", doc_list)
'''
