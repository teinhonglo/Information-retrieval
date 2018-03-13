import numpy as np
np.random.seed(5566)
import os
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import keras.optimizers as optimizers

import NPM
from SeqGenerator import DataGenerator
from Preprocess import InputDataProcess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                  inter_op_parallelism_threads = 1, intra_op_parallelism_threads = 1))

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

MAX_QRY_LENGTH = 200
MAX_DOC_LENGTH = 200
NUM_OF_FEATS = 4
PSG_SIZE = [(50, 1), (150, 1), (MAX_QRY_LENGTH, MAX_DOC_LENGTH)]
NUM_OF_FILTERS = 1
tau = 1

optimizer = optimizers.Adam(lr=0.17)
loss = "categorical_crossentropy"
batch_size = 512
epochs = 500
exp_path = "exp/categorical_cnn_Adam_" + loss + "_weights-{epoch:02d}-{val_loss:.2f}.hdf5"

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
[partition, labels, partition_answer] = input_data_process.genTrainValidSet()
class_weight = class_weight.compute_class_weight('balanced', np.unique(partition_answer['train']), partition_answer['train'])
class_weight = {0:class_weight[0], 1:class_weight[1]}

print "Training: ", len(partition['train'])
print "Validation: ", len(partition['validation'])
print "Class Weight:", class_weight

# Generators
training_generator = DataGenerator(**params).generate(labels, partition['train'])
validation_generator = DataGenerator(**params).generate(labels, partition['validation'])

# Model check point
checkpoint = ModelCheckpoint(exp_path, monitor='val_loss', verbose=0, save_best_only=False, mode='min')
callbacks_list = [checkpoint]
with tf.device('/device:GPU:0'):
    # Train model on dataset
    # Design model
    model = NPM.create_model(MAX_QRY_LENGTH, MAX_DOC_LENGTH, NUM_OF_FEATS, PSG_SIZE, NUM_OF_FILTERS, tau)
    model.compile(optimizer = optimizer, loss = loss, metrics=['accuracy'])
    model.fit_generator(generator = training_generator,
                        steps_per_epoch = len(partition['train']) / batch_size,
                        epochs = epochs,
                        validation_data = validation_generator,
                        validation_steps = len(partition['validation']) / batch_size,
#                       class_weight = class_weight,
                        callbacks=callbacks_list)
                    
