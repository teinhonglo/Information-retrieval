import os
import sys
sys.path.append("../Tools")

import numpy as np
import ProcDoc
from PLSA_class import pLSA
from Clustering import ClusterModel

np.random.seed(1337)
corpus = "TDT2"
doc_path = "../Corpus/" + corpus + "/SPLIT_DOC_WDID_NEW"
cluster_dir = "Topic"
num_topics = 4
iterations = 20
doc_file = ProcDoc.readFile(doc_path)
doc_mdl_dict = ProcDoc.docPreproc(doc_file)

# general model
vocab = {}
for doc_ID, word_count in doc_mdl_dict.items():
    for word, count in word_count.items():
        if word in vocab:
            continue
        else:
            vocab[word] = len(list(vocab.keys()))

if not os.path.isfile(cluster_dir + "/pwz_list.npy"):
    np.save("exp/w_IDs", vocab)
    cluster_mdl = ClusterModel(doc_mdl_dict, vocab, num_topics)
    cluster_mdl.save(cluster_dir)

pwz = np.load(cluster_dir + "/pwz_list.npy")
doc_mdl_np, _, doc_IDs = ProcDoc.dict2npDense(doc_mdl_dict, list(vocab.keys()))
pzd = np.ones((doc_mdl_np.shape[0], num_topics))
doc_mdl_np = np.transpose(doc_mdl_np)
# PLSA
model = pLSA(doc_mdl_np, num_topics, pwz, pzd)
[pzd, pwz, pzdw] = model.EM_Trainging(iterations)
np.save("exp/pzd", pzd)
np.save("exp/pwz", pwz)
np.save("exp/pzdw", pzdw) 
            
