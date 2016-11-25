# kmeans clustering algorithm
# data = set of data points
# k = number of clusters
# c = initial list of centroids (if provided)
import os
import numpy as np
import sys

class dataInfo:
    def __init__(self, ID, coor):
        self.ID = ID        # title
        self.coor = coor    # numpy array
        
    def getID(self):
        return self.ID
    
    def getCoor(self):
        return self.coor 

def kmeans(data, k):
    centroids = []

    centroids = randomize_centroids(data, centroids, k)  

    old_centroids = [[] for i in range(k)] 

    iterations = 0
    while not (has_converged(centroids, old_centroids, iterations)):
        iterations += 1

        clusters = [[] for i in range(k)]

        # assign data points to clusters
        clusters = euclidean_dist(data, centroids, clusters)

        # recalculate centroids
        index = 0
        for cluster in clusters:
            old_centroids[index] = centroids[index]
            cluster_union = []
            cluster_np = []
            # create union in cluster
            for d in cluster:
                cluster_union = list(set(cluster_union) | set(d.getCoor().flatten().tolist()))
            cluster_union = sorted(cluster_union)
            threshold = [0.3 for i in range(len(cluster_union))]
			
            # get intersection between element and union	
            for d in cluster:
                d_list = [0] * len(cluster_union)
                d_intesection = sorted(list(set(d.getCoor().flatten().tolist()) & set(cluster_union)))
                for d_i in d_intesection:
                    d_list[cluster_union.index(d_i)] = 1
                cluster_np.append(np.array(d_list))
            new_centroids = filter(lambda a: a != 0, np.array(cluster_union) * (np.mean(cluster_np, axis=0) >= np.array(threshold)))
            centroids[index] = (new_centroids).tolist()
            index += 1

    return clusters

# Calculates euclidean distance between
# a data point and all the available cluster
# centroids.      
def euclidean_dist(data, centroids, clusters):
    for instance in data:  
        # Find which centroid is the closest
        # to the given data point.
        # convert to two similarity np array
		
        temp_instanse = []
        temp_centroid = []
        mu_value = sys.maxint
        mu_index = 0
		
        for c in centroids:
            [temp_instanse, temp_centroid] = list_compare(instance.getCoor().flatten().tolist(), c)
            cur_idx = centroids.index(c)
            cur_value = np.sqrt((temp_instanse - temp_centroid)**2).sum(axis = 0)
            if cur_value < mu_value:
                mu_value = cur_value
                mu_index = cur_idx   

        try:
            clusters[mu_index].append(instance)
        except KeyError:
            clusters[mu_index] = [instance]
		
    # If any cluster is empty then assign one point
    # from data set randomly so as to not have empty
    # clusters and 0 means.        
    for cluster in clusters:
        if not cluster:
            cluster.append(data[np.random.randint(0, len(data), size=1)])

    return clusters


# randomize initial centroids
def randomize_centroids(data, centroids, k):
    for cluster in range(0, k):
        centroids.append(data[np.random.randint(0, len(data), size=1)].coor.flatten().tolist())
    return centroids


# check if clusters have converged    
def has_converged(centroids, old_centroids, iterations):
    MAX_ITERATIONS = 1000
    if iterations > MAX_ITERATIONS:
        return True
    return old_centroids == centroids

# convert to two similarity np array
def list_compare(l1, l2):	
	union = sorted(list(set(l1) | set(l2)))
	intersection_1 = sorted(list(set(l1) & set(union)))
	intersection_2 = sorted(list(set(l2) & set(union)))
	
	l3 = [0] * len(union)
	l4 = [0] * len(union)
	
	for item1 in intersection_1:
		l3[union.index(item1)] = 1 
	for item1 in intersection_2:	
		l4[union.index(item1)] = 1 
		
	return [np.array(l3), np.array(l4)]	