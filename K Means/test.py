import Kmeans
import numpy as np
names = ["1", "2", "3", "4", "5"]
coors = np.array([[0, 1, 2, 3, 4, 5],
				[5, 5, 5, 5, 5, 5],
				[0, 0, 0, 0, 0, 0],
				[1, 2, 3, 4 ,2, 4],
				[5, 6, 6, 6, 7, 7]])
datasets = {"1025": [0, 1, 2, 3, 4, 5], "956":[5, 5, 5, 5, 5, 5], "10258": [0, 0, 0, 0, 0, 0], "33758": [1, 2, 3, 4 ,2, 4], "111": [5, 6, 6, 6, 7, 7]}
dataset = []

for ID, Coor in datasets.items(): 
	data = Kmeans.dataInfo(ID, np.array(Coor))
	dataset.append(data)
		
clusters = Kmeans.kmeans(dataset, 3)
for cluster in clusters:
    print("Cluster with a size of " + str(len(cluster)) + " starts here:")
    for d in cluster:
        print(d.ID, np.array(d.coor).tolist())
    print("Cluster ends here.")