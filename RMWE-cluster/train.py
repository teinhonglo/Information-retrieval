import numpy as np
import cPickle as Pickle
import nn_Train

# Kmeans filename
cluster_fpath = "clusters/"
num_of_clusters = [2, 4, 8]
# model_path
model_path = "../Corpus/model/TDT2/UM/"
# objective function
# method and filename
obj_fpath = "obj_func/"
obj_funcs = ["RM", "SRM", "SWLM", "SSWLM", "RM_S", "SRM_S", "SWLM_S", "SSWLM_S"]
obj_fnames = ["relevance_model_RM.pkl",
             "relevance_model_SRM.pkl",
			 "rel_supervised_swlm_entropy.pkl",
			 "rel_swlm_entropy_9.pkl",
			 "relevance_model_RM_S.pkl",
			 "relevance_model_SRM_S.pkl",
			 "rel_supervised_swlm_entropy_S.pkl",
			 "rel_swlm_entropy_9_S.pkl"]

def clust2train(model_path, obj_fpath, obj_fname, fpath, k):
	# Preprocess training data
	X_train = []
	Y_train = []
	with open(model_path + "query_model.pkl", "rb") as f: query_model = Pickle.load(f)
	with open(obj_fpath + obj_fname, "rb") as f: obj_model = Pickle.load(f)
	with open(fpath + "kmeans_" + str(k) + ".txt", "r") as f:
		for line in f.readlines():
			data = line.split(",")
			cur_qry = []
			cur_obj = []
			for idx in data[1:]:
				cur_qry.append(query_model[idx])
				cur_obj.append(obj_model[idx])
			X_train.append(np.copy(np.vstack(cur_qry))	
			Y_train.append(np.copy(np.vstack(cur_obj))	
	return X_train, Y_train
	
# Training Script
for k_clusters in num_of_clusters:
	for obj_idx, obj_fname in enumerate(obj_fnames):
		# Read the K-Cluster file and objective function
		X_train, Y_train = clust2train(model_path, obj_fpath, obj_fname, cluster_fpath, k_clusters)
		# Iterate each cluster
		for idx in len(X_train):
			X, Y = X_train[idx], Y_train[idx]
			# Call for nn_Train.py
			model = nn_Train.create_model()
			model.train(X, Y)
			# Storage training model
			model.save("TDT2", obj_funcs[obj_idx], idx)