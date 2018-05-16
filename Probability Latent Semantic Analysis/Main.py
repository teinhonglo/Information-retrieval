import os
import sys
sys.path.append("../Tools")

import numpy as np
import cPickle as pickle
import ProcDoc
from PLSA_class import pLSA
from Clustering import ClusterModel

np.random.seed(1337)
corpus = "TDT2"
doc_path = "../Corpus/" + corpus + "/SPLIT_DOC_WDID_NEW"
cluster_dir = "Topic"
num_of_topic = 4
doc = ProcDoc.readFile(doc_path)
doc_dict = ProcDoc.docPreproc(doc)

# general model
collection = {}
for doc_ID, word_count in doc_dict.items():
    for word, count in word_count.items():
        if word in collection:
            collection[word] += count
        else:
            collection[word] = count

if not os.path.isfile(cluster_dir + "/pwz_list.pkl"):
    cluster_mdl = ClusterModel(doc_dict, collection.keys(), num_of_topic)
    cluster_mdl.save(cluster_dir)

with open(cluster_dir + "/pwz_list.pkl", "rb") as pwz_file: pwz = pickle.load(pwz_file)
doc_np, doc_IDs = ProcDoc.dict2npDense(doc_dict, collection.keys())
pwd = np.ones((doc_np.shape[0], num_of_topic))
doc_np = np.transpose(doc_np)
# PLSA
model = pLSA(doc_np, num_of_topic, pwz, pwd)
[pzd, pwz, pzdw] = model.EM_Trainging(20)
with open("exp/pzd.pkl", "wb") as pzd_file : pickle.dump(pzd, pzd_file, True)
with open("exp/pwz.pkl", "wb") as pwz_file : pickle.dump(pwz, pwz_file, True)
with open("exp/pzdw.pkl", "wb") as pzdw_file : pickle.dump(pzwd, pzwd_file, True)
            