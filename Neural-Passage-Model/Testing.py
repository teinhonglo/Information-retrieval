import numpy as np
np.random.seed(5566)
import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

import NPM
from SeqGenerator import DataGenerator
from preprocess import InputDataProcess

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
loss = "kullback_leibler_divergence"
exp_path = "exp/"

input_data_process = InputDataProcess(NUM_OF_FEATS, MAX_QRY_LENGTH, MAX_DOC_LENGTH)
# Parameters
params = {'input_data_process': input_data_process,
          'dim_x': MAX_QRY_LENGTH,
          'dim_y': MAX_DOC_LENGTH,
		  'dim_x1': NUM_OF_FEATS,
          'batch_size': batch_size,
          'shuffle': True}

[partition, labels] = input_data_process.genTrainValidSet()

# Generators
training_generator = DataGenerator(**params).generate(labels, partition['train'])
validation_generator = DataGenerator(**params).generate(labels, partition['validation'])

# Design model
model = model.load("exp_path/" + model_name + ".h5")
model.compile(optimizer = optimizer, loss = loss, metrics=["accuracy"])

with tf.device('/gpu:0'):
	# Train model on dataset
	model.predict_generator(generator = training_generator,	steps_per_epoch = len(partition['train']) / batch_size)
					
# import evaluation
# other