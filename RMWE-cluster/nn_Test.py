#-*- coding: utf-8 -*-
''' Import theano and numpy '''
import theano
import numpy as np

''' Import keras to build a DL model '''
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
import cPickle as Pickle

num_of_clusters = 8
corpus = "TDT2"
method = "SSWM"
spoken = "_S"
short = "_short"
model_name = "RLE_" + method + spoken
model_path = "../Corpus/model/" + corpus + "/UM/"
result_path = "NN_Result/" + corpus + "/" + method + "/" + str(num_of_clusters) + "/"

class create_model:
    def __init__(self, num_of_clusters, method, corpus, model_name):
        # Load NN_model
        models = []
        for k in xrange(num_of_clusters):
            NN_Model_path = "NN_Model/" + corpus + "/" + method + "/" + str(num_of_clusters) + "/" + model_name + "_" + str(k) + ".h5"
            print NN_Model_path
            model = load_model(NN_Model_path)
            models.append(model)
        self.models = models    
        # Get centroids
        self.centroids = self.get_centroids(corpus, num_of_clusters)
        
    def predict(self, qry_model):
        centroids = self.centroids
        models = self.models
        re_qry_model = []
        # Iterate each query
        for qry_vec in qry_model:
            # Get labels
            label = np.sqrt(((qry_vec - centroids) ** 2).sum(axis=1)).argmin(axis=0)
            print label
            # Predict
            pred_vec = np.array([qry_vec.reshape(qry_vec.shape[0])])
            re_qry_vec = models[label].predict(pred_vec)
            re_qry_model.append(re_qry_vec)
        re_qry_model = np.vstack(re_qry_model)    
        return re_qry_model
        
    def get_centroids(self, corpus, num_of_clusters):
        # read cluster file
        centroids = []
        with open("clusters/" + corpus + "/kmeans_centroids_" + str(num_of_clusters) + ".pkl", "rb") as f:
            centroids = Pickle.load(f)
        return centroids

if __name__ == "__main__":
    with open(model_path + "test_query_model" + short + ".pkl", "rb") as f : qry_model = Pickle.load(f)
    test_model = create_model(num_of_clusters, method, corpus, model_name)
    qry_model = test_model.predict(qry_model)
    with open(result_path + "test_query_model" + short + spoken + ".pkl", "wb") as f: Pickle.dump(qry_model, f, True)
