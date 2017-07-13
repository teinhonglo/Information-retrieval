import numpy as np
import copy
import cPickle as Pickle

np.random.seed(1337)

with open("test_query_list.pkl", "rb") as q_file : query_list = Pickle.load(q_file)
with open("doc_list.pkl", "rb") as d_file : doc_list = Pickle.load(d_file)
with open("Pseudo_16_qry_VSM_Spk.pkl", "rb") as h_file : HMMTraingSetDict = Pickle.load(h_file)

print len(query_list)
print len(doc_list)
print len(HMMTraingSetDict.keys())

train_set = []
# create train data
# iterate each query
for q_key in query_list:
	# iterate each relevant document
	# !!!! check data structure !!!!
	for d_key, point in HMMTraingSetDict[q_key]:
		# cteate 3 train set for each P(D+|Q)
		total_ir_set = []
		# select irrelevant set, [D-, D-, D-] * 3
		for i in xrange(10):
			ir_set = []
			# select irrelevant doc, D- * 3
			for n in xrange(3):
				while True:
					# random select irrelevant
					rand_idx = int(np.random.rand(1) * 2265)
					doc_id = doc_list[rand_idx]

					# check if irrelevant or not
					if doc_id in HMMTraingSetDict[q_key]:
						continue
					# check if exist in train set
					if doc_id in total_ir_set:
						continue
					total_ir_set.append(doc_id)
					ir_set.append(doc_id)
					break
			train_set.append([q_key, d_key, ir_set[0], ir_set[1], ir_set[2]])	

print len(train_set)
print train_set[50]

with open("pseudo_qry_val_set_Spk.pkl", "wb") as file: Pickle.dump(train_set, file, True)
with open("pseudo_qry_val_set_Spk.pkl", "rb") as file: trn_set = Pickle.load(file)
