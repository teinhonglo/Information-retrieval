import sys
sys.path.append("../Tools")

import numpy as np
import ProcDoc

class InputDataProcess(object):
	def __init__(self, num_of_heter_feats = 10, max_qry_length = 1794, max_doc_length = 2907, query_path = None, document_path = None, corpus = "TDT2"):
		resPos = True
		self.max_qry_length = max_qry_length
		self.max_doc_length = max_doc_length
		self.num_of_heter_feats = num_of_heter_feats
		if query_path == None: 
			query_path = "../Corpus/" + corpus + "/Train/XinTrainQryTDT2/QUERY_WDID_NEW"
		if document_path == None:
			document_path = "../Corpus/" + corpus + "/SPLIT_DOC_WDID_NEW"
		# read document, reserve position
		doc = ProcDoc.read_file(document_path)
		self.doc = ProcDoc.doc_preprocess(doc, resPos)

		# read query, reserve position
		qry = ProcDoc.read_file(query_path)
		self.qry = ProcDoc.query_preprocess(qry, resPos)	
		
		# HMMTrainingSet
		self.hmm_training_set = ProcDoc.read_relevance_dict()
		self.heter_feature = self.__genFeature(num_of_heter_feats)
		
	def genPassageAndLabels(self, list_IDs, labels, batch_size):
		qry = self.qry
		doc = self.doc
		max_qry_length = self.max_qry_length
		max_doc_length = self.max_doc_length
		heter_feature = self.heter_feature
		num_of_heter_feats = self.num_of_heter_feats
		psg_mat_batch = np.zeros((batch_size, max_qry_length, max_doc_length, 1))
		heter_feature_batch = np.zeros((batch_size, num_of_heter_feats))
		rel_batch = np.zeros((batch_size))
		# generate passage
		for idx, data_ID in enumerate(list_IDs):
			[q_id, d_id] = data_ID.split("_")
			q_terms = map(int, qry[q_id].split())
			d_terms = np.asarray(map(int, doc[d_id].split()))
			psg_mat = np.asarray([[(q_t == d_terms)] for q_t in q_terms]).reshape(len(q_terms), len(d_terms), 1)
			psg_mat = self.__mergeMat(np.zeros((max_qry_length, max_doc_length, 1)), psg_mat)
			psg_mat_batch[idx] = np.copy(psg_mat)
			heter_feature_batch[idx] = heter_feature[q_id]
			rel_batch[idx] = labels[data_ID]
		return [np.asarray(psg_mat_batch), np.asarray(heter_feature_batch), np.asarray(rel_batch)]
	
	def genTrainValidSet(self):
		print "generate training set and validation set"
		qry = self.qry
		doc = self.doc
		total_qry = len(qry.keys())
		total_doc = len(doc.keys())
		hmm_training_set = self.hmm_training_set
		labels = {}
		total = total_qry * total_doc
		num_of_train = total * 8 / 10 
		num_of_valid = total - num_of_train
		partition = {'train': [], 'validation': []}
		# relevance between queries and documents
		for q_id in qry:
			for d_id in doc:
				if d_id in hmm_training_set[q_id]:
					labels[q_id + "_" + d_id] = 1
				else:
					labels[q_id + "_" + d_id] = -1
		# partition
		ID_list = labels.keys()
		# shuffle
		np.random.shuffle(ID_list)
		partition['train'] = [id for id in ID_list[:num_of_train]]
		partition['validation'] = [id for id in ID_list[num_of_train:]]
		return [partition, labels]
	
	def __genFeature(self, num_of_heter_feats):
		###################### TODO ######################
		print "generate h features"
		qry = self.qry
		doc = self.doc
		heter_feats = {}
		for q_id in qry.keys():
			heter_feats[q_id] = np.random.rand(num_of_heter_feats)
		return heter_feats
		
	def __mergeMat(self, b1, b2, pos = [0, 0]):
		[pos_v, pos_h] = pos
		v_range1 = slice(max(0, pos_v), max(min(pos_v + b2.shape[0], b1.shape[0]), 0))
		h_range1 = slice(max(0, pos_h), max(min(pos_h + b2.shape[1], b1.shape[1]), 0))
		v_range2 = slice(max(0, -pos_v), min(-pos_v + b1.shape[0], b2.shape[0]))
		h_range2 = slice(max(0, -pos_h), min(-pos_h + b1.shape[1], b2.shape[1]))
		b1[v_range1, h_range1] += b2[v_range2, h_range2]
		return b1


