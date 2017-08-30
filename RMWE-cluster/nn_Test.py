#-*- coding: utf-8 -*-
''' Import theano and numpy '''
import theano
import numpy as np

''' Import keras to build a DL model '''
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
import cPickle as Pickle

num_of_clusters = 2
corpus = "TDT2"
method = "RM"
spoken = ""
short = ""
model_name = "RLE_" + method + spoken
model_path = "../Corpus/model/" + corpus + "/UM/"
result_path = "NN_Result/" + corpus + "/" + method + "/" + num_of_clusters + "/"

def create_model:
	def __init__(self, num_of_clusters, corpus, model_name):
		# Load NN_model
		models = []
		for cur_cluster in xrange(num_of_clusters):
			model = load_model("NN_Model/" + corpus + "/" + model_name + "_" + cur_cluster + ".h5")
			models.append(model)
		self.models = models	
		# Get centroids
		self.centroids = get_centroids(corpus, num_of_clusters)
		
	def predict(self, qry_model):
		centroids = self.centroids
		models = self.models
		re_qry_model = []
		# Iterate each query
		for qry_vec in qry_model:
			# Get labels
			label = np.sqrt(((qry_vec - centroids) ** 2).sum(axis=1)).argmin(axis=0)
			# Predict
			re_qry_vec = models[label].predict(qry_vec)
			re_qry_model.append(re_qry_vec)
		re_qry_model = np.vstack(re_qry_model)	
		return re_qry_model
		
	def get_centroids(self, corpus, num_of_clusters):
		# read cluster file
		centroids = []
		with open("clusters/kmeans_centroids_" + str(num_of_clusters) + ".pkl", "rb") as f:
			centroids = Pickle.load(f)
		return centroids

if __name__ == "__main__":
	with open(model_path + "test_query_model.pkl", "rb") as f : qry_model = Pickle.load(f)
	test_model = create_model(num_of_clusters, corpus, model_name)
	qry_model = test_model.predict(qry_model)
	with open(result_path + "test_query_model" + short + spoken + ".pkl", "wb") as f: Pickle.dump(qry_model, f, True)