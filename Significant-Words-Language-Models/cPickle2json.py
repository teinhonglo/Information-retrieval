import cPickle as pickle
import numpy as np

with open("rel_supervised_swlm_entropy.pkl", "rb") as f:
    rel = pickle.load(f)

qry_keys = rel.keys()
for qry_idx, qry_key in enumerate(qry_keys):
    rel_np = np.zeros(51253)
    for word, count in rel[qry_key].items():
        rel_np[int(word)] = count
    rel[qry_key]=np.array(rel_np)

np.save("rel_supervised_swlm_entropy.np", rel)

