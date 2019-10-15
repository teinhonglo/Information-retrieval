import numpy as np
import copy
import cPickle as Pickle

np.random.seed(1337)

model_path = "../Corpus/model/TDT3/UM/"
with open(model_path + "query_list.pkl", "rb") as q_file : query_list = Pickle.load(q_file)
with open(model_path + "doc_list.pkl", "rb") as d_file : doc_list = Pickle.load(d_file)
with open(model_path + "HMMTraingSetDict.pkl", "rb") as h_file : HMMTraingSetDict = Pickle.load(h_file)

print len(query_list)
print len(doc_list)
print len(HMMTraingSetDict.keys())

rel_count = 0
train_set = []
# create train data
# iterate each query
for q_key in query_list:
    # iterate each relevant document
    # !!! check data structure !!!
    for d_key in HMMTraingSetDict[q_key]:
        rel_count += 1
        # cteate 3 train set for each P(D+|Q)
        total_ir_set = []
        # select irrelevant set, [D-, D-, D-] * 3
        for i in xrange(3):
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
print rel_count
print train_set[50]
with open("TDT3/qry_train_set.pkl", "wb") as file: Pickle.dump(train_set, file, True)
with open("TDT3/qry_train_set.pkl", "rb") as file: trn_set = Pickle.load(file)
