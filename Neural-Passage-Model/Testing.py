import numpy as np
np.random.seed(5566)
import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import NPM
from SeqGenerator import DataGenerator
from Preprocess import InputDataProcess
from Evaluate import EvaluateModel
from collections import defaultdict
import operator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                  inter_op_parallelism_threads = 1, intra_op_parallelism_threads = 1))

MAX_QRY_LENGTH = 200
MAX_DOC_LENGTH = 200
NUM_OF_FEATS = 4
batch_size = 4
percent = 100
optimizer = "Adam"
loss = "kullback_leibler_divergence"
exp_path = "exp/"
test_path = "../Corpus/TDT2/QUERY_WDID_NEW"
model_name = "basic_cnnAdam_logcosh_weights-04-0.04.hdf5"

input_data_process = InputDataProcess(NUM_OF_FEATS, MAX_QRY_LENGTH, MAX_DOC_LENGTH, test_path)
evaluate_model = EvaluateModel()
# Parameters
params = {'input_data_process': input_data_process,
          'dim_x': MAX_QRY_LENGTH,
          'dim_y': MAX_DOC_LENGTH,
		  'dim_x1': NUM_OF_FEATS,
          'batch_size': batch_size,
          'shuffle': False}

[partition, labels] = input_data_process.genTrainValidSet(percent)

# Generators
training_generator = DataGenerator(**params).generate(labels, partition['train'])

# Design model
model = load_model(exp_path + model_name)
model.compile(optimizer = optimizer, loss = loss, metrics=["accuracy"])
qry_doc = defaultdict(list)
with tf.device('/gpu:0'):
	# Train model on dataset
	pred = model.predict_generator(generator = training_generator,	steps = len(partition['train']) / batch_size)
	print pred.shape
	for idx, id in enumerate(partition['train']):
		q_id, d_id = id.split('_')
		qry_doc[q_id].append([d_id, pred[idx]])
	for id, docs_point in qry_doc.items():
		qry_doc[id] = sorted(docs_point, key=operator.itemgetter(1), reverse = True)
	mAP = evaluate_model.mAP(qry_doc)
	print mAP
# import evaluation
# other