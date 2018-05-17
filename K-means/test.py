import Kmeans
import numpy as np
names = ["1", "2", "3", "4", "5"]
coors = np.array([[0, 1, 2, 3, 4, 5],
				[5, 6, 7, 8, 9, 10],
				[0, 2, 4, 7, 11],
				[2, 4, 5],
				[5, 6, 7]])
datasets = {"1025": [0, 1, 2, 3, 4, 5], "956":[5, 6, 7, 8, 9, 10], "10258": [0, 2, 4, 7, 11], "33758": [2, 4, 5], "111":[5, 6, 7], "456": [2, 9, 11, 12, 16, 17]}
dataset = []

for ID, Coor in datasets.items():
	data = Kmeans.dataInfo(ID, np.array(Coor))
	dataset.append(data)
		
clusters = Kmeans.kmeans(dataset, 2)
for cluster in clusters:
    print("Cluster with a size of " + str(len(cluster)) + " starts here:")
    for d in cluster:
        print(d.ID, np.array(d.coor).tolist())
    print("Cluster ends here.")