import numpy as np
np.random.seed(5566)

class DataGenerator(object):
	#'Generates data for Keras'
	def __init__(self,input_data_process, dim_x = 32, dim_y = 32, dim_x1 = 32, batch_size = 32, shuffle = True):
		#'Initialization'
		self.input_data_process = input_data_process
		self.dim_x = dim_x
		self.dim_y = dim_y
		self.dim_x1 = dim_x1
		self.batch_size = batch_size
		self.shuffle = shuffle

	def generate(self, labels, list_IDs):
		# Generates batches of samples
		# Infinite loop
		while 1:
			# Generate order of exploration of dataset
			indexes = self.__getExplorationOrder(list_IDs)
			# Generate batches
			imax = int(len(indexes)/self.batch_size)
			for i in xrange(imax):
				# Find list of IDs
				list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
				# Generate data
				X, X1, y = self.__dataGeneration(labels, list_IDs_temp)
				yield ([X, X1], y)

	def __getExplorationOrder(self, list_IDs):
		# Generates order of exploration
		# Find exploration order
		indexes = np.arange(len(list_IDs))
		if self.shuffle == True:
			np.random.shuffle(indexes)
			
		return indexes

	def __dataGeneration(self, labels, list_IDs_temp):
		# Generates data of batch_size samples
		# X : (n_samples, v_size, v_size, n_channels)
		# Initialization
		'''
		X = np.empty((self.batch_size, self.dim_x, self.dim_y, 1))
		X1 = np.empty((self.batch_size, self.dim_x1))
		y = np.empty((self.batch_size), dtype = int)
		'''
		# Generate data
		[X, X1, y] = self.input_data_process.genPassageAndLabels(list_IDs_temp, labels, self.batch_size)
		'''
		for i, ID in enumerate(list_IDs_temp):
			# Store volume
			#X[i, :, :, 0] = np.load(ID + '_psg.npy')
			#X1[i, :, :, 0] = np.load(ID + '_h.npy')
			# Store class
			y[i] = labels[ID]
		'''	
		return X, X1, self.__sparsify(y)

	def __sparsify(self, y):
		#'Returns labels in binary NumPy array'
		n_classes = 2# Enter number of classes
		return np.array([[1 if y[i] == j else 0 for j in range(n_classes)] for i in range(y.shape[0])])
