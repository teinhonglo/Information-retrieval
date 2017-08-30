# -*- coding: utf-8 -*-
import csv
import random
import numpy as np
import cPickle as Pickle 
import kmeans

# Read Query Model
model_path = "../Corpus/model/TDT2/UM/"
query_model = Pickle.load(model_path + "query_model.pkl")
# Preprocess(fit data structure in kmeans.py)
for idx, vec in enumerate(query_model):
	data = kmeans.dataInfo(idx, vec)
	data_list.append(data)
	
# Calculate
for num_of_cluster in [2, 4, 8]:			
	# kmeans
	# return clusters and centeroids
	[clusters, centroids] = kmeans.kmeans(data_list, num_of_cluster)	

	with open('clusters/kmeans_' + str(num_of_cluster) + '.txt', 'w') as output:
		# Compute SSE
		sse = 0
		for cur_cluster in xrange(0, len(clusters)):
			# Cluster 
			cluster_name = str(cur_cluster) + "," + centroids[cur_cluster]
			output.write(cluster_name)
			
			# Centeroid
			#print centroids[cur_cluster]
			cluster_data = []
			data_str = ""
			
			# Item
			for data in clusters[cur_cluster]:
				data_str += "," + data.getID()
				cluster_data.append(data.getCoor())
			output.write(data_str)
			output.write("\n")
			
#			# Sum of Squared Error
#			sse += ((cluster_data - centroids[cur_cluster]) ** 2).sum(axis = 1).sum(axis = 0)
#		output.write(unicode("SSE : ", 'utf-8', errors="replace"))
#		output.write(unicode(str(sse) + "\n", 'utf-8', errors="replace"))