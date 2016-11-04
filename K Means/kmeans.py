# kmeans clustering algorithm
# data = set of data points
# k = number of clusters
# c = initial list of centroids (if provided)
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
            cluster_np = []
            for d in cluster:
                cluster_np.append(d.coor.flatten().tolist())
            centroids[index] = np.mean(cluster_np, axis=0).tolist()
            index += 1

	'''
    print("The total number of data instances is: " + str(len(data)))
    print("The total number of iterations necessary is: " + str(iterations))
    print("The means of each cluster are: " + str(centroids))
    print("The clusters are as follows:")
    for cluster in clusters:
        print("Cluster with a size of " + str(len(cluster)) + " starts here:")
        print(np.array(cluster).tolist())
        print("Cluster ends here.")
	'''
    return clusters

# Calculates euclidean distance between
# a data point and all the available cluster
# centroids.      
def euclidean_dist(data, centroids, clusters):
    for instance in data:  
        # Find which centroid is the closest
        # to the given data point.
        mu_index = np.sqrt(((instance.coor-centroids)**2).sum(axis=1)).argmin(axis=0)
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