import kmeans
import numpy as np

data = np.array([[0, 1, 2, 3, 4, 5],
				[5, 5, 5, 5, 5, 5],
				[0, 0, 0, 0, 0, 0],
				[1, 2, 3, 4 ,2, 4],
				[5, 6, 6, 6, 7, 7]])
		
cluster = kmeans.kmeans(data, 3)
print cluster