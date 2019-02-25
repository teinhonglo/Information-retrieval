import numpy as np
np.random.seed(5566)

class DataGenerator(object):
    #'Generates data for Keras'
    def __init__(self,input_data_process, 
                 dim_x = 32, dim_y = 32, 
                 batch_size = 32, shuffle = True):
        #'Initialization'
        self.input_data_process = input_data_process
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.type_rank = input_data_process.type_rank
        self.type_feat = input_data_process.type_feat

    def generate(self, labels, list_IDs):
        # Generates batches of samples
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__getExplorationOrder(list_IDs)
            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                # Generate data
                training_data = self.__dataGeneration(labels, list_IDs_temp)
                yield training_data

    def __getExplorationOrder(self, list_IDs):
        # Generates order of exploration
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)
        return indexes

    def __dataGeneration(self, labels, list_IDs_temp):
        # Generates data of batch_size samples
        # X : (n_samples, v_size)
        # Initialization
        # read input_data_process
        #----------------------------------------------------------        
        # Generate data
        # pointwise: dense, sparse, num of input = 1
        #######################################
        return (X, y)
        # pointwise: embeddings, num of input = 2 (the other is fixed weight)
        #######################################
	return ([X, W], y)
        # pairwise: dense, sparse, num of input = 2
        #######################################
        return ([X, X1], y)
        # pairwise: embeddings, num of input = 4 (word_emb * 2, fixed weight * 2)
        #-----------------------------------------------------------
        return ([X, W, X2, W1], y)

    def __sparsify(self, y):
        # Returns labels in binary NumPy array
        n_classes = 2# Enter number of classes
        return np.array([[1 if y[i] == j else 0 for j in range(n_classes)] for i in range(y.shape[0])])
