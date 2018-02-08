import sys
sys.path.append("../Tools")

import numpy as np
import gc
import ProcDoc

data = {}                # content of document (doc, content)
background_model = {}    # word count of 2265 document (word, number of words)
query = {}                # query

class InputDataProcess(object):
	def __init__(self, query_path = None, document_path = None, corpus = "TDT2"):
		resPos = True
		self.max_qry_length = 1794
		self.max_doc_length = 2907
		if query_path == None: 
			query_path = "../Corpus/" + corpus + "/Train/XinTrainQryTDT2/QUERY_WDID_NEW"
		if document_path == None:
			document_path = "../Corpus/" + corpus + "/SPLIT_DOC_WDID_NEW"
		# read document, reverse position
		doc = ProcDoc.read_file(document_path)
		self.doc = ProcDoc.doc_preprocess(doc, resPos)

		# read query, reserve position
		qry = ProcDoc.read_file(query_path)
		self.qry = ProcDoc.query_preprocess(qry, resPos)	
		
		# HMMTrainingSet
		self.hmm_training_set = ProcDoc.read_relevance_dict()
		self.heter_feature = __genFeature()
		
	def genPassage(self, IDs_list, batch_size):
		qry = self.qry
		doc = self.doc
		max_qry_length = self.max_qry_length
		max_doc_length = self.max_doc_length
		heter_feature = self.heter_feature
		pas_mat_batch = []
		heter_feature_batch = []
		rel_batch = []
		# generate passage
		for data_ID in IDs_list:
			[q_id, d_id] = data_ID.split("_")
			q_terms = map(int, qry[q_id].split())
			d_terms = np.asarray(map(int, doc[d_id].split()))
			psg_mat = np.asarray([[(q_t == d_terms)] for q_t in q_terms]).reshape(len(q_terms), len(d_terms), 1)
			psg_mat = self.__mergeMat(np.zeros((qry_length, doc_length, 1)), psg_mat)
			psa_mat_batch.append(np.copy(psg_mat))
			heter_feature_batch.append(heter_feature[q_id])
			
		return [np.asarray(psa_mat_batch), np.asarray(heter_feature_batch)]
	
	def __genFeature(self):
		######################## TODO ######################
		qry = self.qry
		doc = self.doc
		return np.random.rand(800, 10)
		
	def __mergeMat(self, b1, b2, pos = [0, 0]):
		[pos_v, pos_h] = pos
		v_range1 = slice(max(0, pos_v), max(min(pos_v + b2.shape[0], b1.shape[0]), 0))
		h_range1 = slice(max(0, pos_h), max(min(pos_h + b2.shape[1], b1.shape[1]), 0))
		v_range2 = slice(max(0, -pos_v), min(-pos_v + b1.shape[0], b2.shape[0]))
		h_range2 = slice(max(0, -pos_h), min(-pos_h + b1.shape[1], b2.shape[1]))
		b1[v_range1, h_range1] += b2[v_range2, h_range2]
		return b1
	
	def genTrainValidSet(self):
		qry = self.qry
		doc = seld.doc
		total_qry = len(qry.keys())
		total_doc = len(doc.keys())
		hmm_training_set = self.hmm_training_set
		labels = {}
		total = total_qry * total_doc
		num_of_train = total / 8
		num_of_valid = total - num_of_train
		partition = {'train': [], 'validation': []}
		# relevance between queries and documents
		for q_id in qry:
			for d_id in doc:
				if d_id in hmm_training_set[q_id]:
					labels[q_id + "_" + d_id] = 1
				else:
					labels[q_id + "_" + d_id] = 0
		# partition
		ID_list = np.random.shuffle(labels.keys())
		partition['train'] = [id] for id in ID_list[:num_of_train]
		partition['validation'] = [id] for id in ID_list[num_of_train:]
		return [partition, labels]
