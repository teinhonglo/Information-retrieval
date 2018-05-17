import os
import numpy as np

class dataInfo:
    def __init__(self, ID, coor):
        self.ID = ID        # title
        self.coor = coor    # numpy array
        
    def getID(self):
        return self.ID
    
    def getCoor(self):
        return self.coor 

# data = set of data points
# k = number of clusters
def kmeans(dataSet, k):
    centroids = []

    centroids = getCentroids(dataSet, centroids, k)  

    old_centroids = [[] for i in range(k)] 

    iterations = 0
    while (not shouldStop(centroids, old_centroids, iterations)):
        iterations += 1

        clusters = [[] for i in range(k)]

        # assign data points to clusters
        clusters = getLabels(dataSet, centroids, clusters)

        # recalculate centroids
        index = 0
        for cluster in clusters:
            old_centroids[index] = centroids[index]
            cluster_np = []
            for d in cluster:
                cluster_np.append(d.getCoor().flatten().tolist())
			# mean	
            centroids[index] = np.mean(cluster_np, axis=0).tolist()
            index += 1

    return [clusters, centroids]

# calculates euclidean distance
def getLabels(dataSets, centroids, clusters):
    for instance in dataSets:  
        # find which centroid is the closest
        closest_index = np.sqrt(((instance.getCoor()-centroids)**2).sum(axis=1)).argmin(axis=0)
        try:
            clusters[closest_index].append(instance)
        except KeyError:
            clusters[closest_index] = [instance]
	
    # if any cluster is empty, then assign one point from data set randomly
    for cluster in clusters:
        if not cluster:
            cluster.append(dataSets[int(np.random.randint(0, len(dataSets), size=1))])

    return clusters


# get randomize initial centroids
def getCentroids(dataSets, centroids, k):
    for cluster in range(0, k):
        centroids.append(dataSets[int(np.random.randint(0, len(dataSets), size=1))].getCoor().flatten().tolist())
    return centroids


# check if clusters have converged    
def shouldStop(centroids, old_centroids, iterations):
    MAX_ITERATIONS = 1000
    if iterations > MAX_ITERATIONS: return True
    return old_centroids == centroids