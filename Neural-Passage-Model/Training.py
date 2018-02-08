import numpy as np
np.random.seed(5566)
import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

import NPM
from SeqGenerator import DataGenerator
from Preprocess import InputDataProcess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                  inter_op_parallelism_threads = 1, intra_op_parallelism_threads = 1))

MAX_QRY_LENGTH = 1794
MAX_DOC_LENGTH = 2907
NUM_OF_FEATS = 10
PSG_SIZE = 50
NUM_OF_FILTERS = 8
batch_size = 4
tau = 1
optimizer = "Adam"
loss = "logcosh"
exp_path = "exp/basic_cnn" + optimizer + "_" + loss + "_weights-{epoch:02d}-{val_loss:.2f}.hdf5"

input_data_process = InputDataProcess(NUM_OF_FEATS, MAX_QRY_LENGTH, MAX_DOC_LENGTH)
# Parameters
params = {'input_data_process': input_data_process,
          'dim_x': MAX_QRY_LENGTH,
          'dim_y': MAX_DOC_LENGTH,
		  'dim_x1': NUM_OF_FEATS,
          'batch_size': batch_size,
          'shuffle': True}
'''
# Datasets
partition = # IDs
{'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
labels = # Labels
{'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}
'''
[partition, labels] = input_data_process.genTrainValidSet()

# Generators
training_generator = DataGenerator(**params).generate(labels, partition['train'])
validation_generator = DataGenerator(**params).generate(labels, partition['validation'])

# Design model
model = NPM.create_model(MAX_QRY_LENGTH, MAX_DOC_LENGTH, NUM_OF_FEATS, PSG_SIZE, NUM_OF_FILTERS, tau)
model.compile(optimizer = optimizer, loss = loss, metrics=["accuracy"])
# Model check point
checkpoint = ModelCheckpoint(exp_path, monitor='val_loss', verbose=0, save_best_only=False, mode='min')
callbacks_list = [checkpoint]
with tf.device('/gpu:0'):
	# Train model on dataset
	model.fit_generator(generator = training_generator,
						steps_per_epoch = len(partition['train']) / batch_size,
						validation_data = validation_generator,
						validation_steps = len(partition['validation']) / batch_size,
						callbacks = callbacks_list)
					
