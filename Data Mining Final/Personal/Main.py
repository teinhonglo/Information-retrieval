# -*- coding: utf-8 -*-
import csv
import io
import random
import numpy as np
import kmeans

vocabulary = []
data_list = []
isFirst = True
# Read Data and Preprocess
with io.open('animate_re.txt', 'r', encoding = 'utf-8') as f:
	for row in f.readlines():
		if not isFirst:
			name = row.split(",")[0]
			info = np.array(row.split(",")[1:])
			info = np.array([float(i) for i in info])
			data = kmeans.dataInfo(name, info)
			data_list.append(data)
		else:
			vocabulary = row.split(",")[1:]
			isFirst = False
# Calculate
for num in range(4, 65):			
	num_of_cluster = num
	# kmeans
	# return clusters and centeroids
	[clusters, centeriods] = kmeans.kmeans(data_list, num_of_cluster)	

	with io.open('cluster_re/kmeans_' + str(num_of_cluster) + '.txt', 'w', encoding = 'utf-8') as output:
		# Compute SSE
		sse = 0
		for i in range(0, len(clusters)):
			# Cluster 
			cluster_name = "cluster" + str(i) + ":\n"
			output.write(unicode(cluster_name, 'utf-8', errors="replace"))
			
			# Centeroid
			#print centeriods[i]
			cluster_data = []
			data_str = ""
			
			# Anime name
			for data in clusters[i]:
				data_str += "," + data.getID()
				cluster_data.append(data.getCoor())
			output.write(data_str[1:])
			output.write(unicode("\n", 'utf-8', errors="replace"))
			
			# Five hightest title	
			cluster_data = np.array(cluster_data)
			cluster_sum = cluster_data.sum(axis = 0)
			cluster_sum = cluster_sum.tolist()
			output.write(unicode("Related Topic:\n", 'utf-8', errors="replace"))
			topic_str = ""
			print sorted(cluster_sum, reverse=True)[:5]
			print cluster_sum
			for fh in sorted(cluster_sum, reverse=True)[:5]:
				topic_str += "," + vocabulary[cluster_sum.index(fh)]
			output.write(topic_str[1:])
			output.write(unicode("\n", 'utf-8', errors="replace"))
			output.write(unicode("\n", 'utf-8', errors="replace"))
			
			# Sum of Squared Error
			sse += ((cluster_data - centeriods[i]) ** 2).sum(axis = 1).sum(axis = 0)
			print 
		output.write(unicode("SSE : ", 'utf-8', errors="replace"))
		output.write(unicode(str(sse) + "\n", 'utf-8', errors="replace"))