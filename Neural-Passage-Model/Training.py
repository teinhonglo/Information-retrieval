import numpy as np
np.random.seed(5566)
import NPM
from SeqGenerator import DataGenerator

MAX_QRY_LENGTH = 1000
MAX_DOC_LENGTH = 2900
NUM_OF_FEATURE = 10
PSG_SIZE = 50
NUM_OF_FILTERS = 5
tau = 1

# Parameters
params = {'dim_x': MAX_QRY_LENGTH,
          'dim_y': MAX_DOC_LENGTH,
		  'dim_x1': NUM_OF_FEATURE,
          'batch_size': 64,
          'shuffle': True}

# Datasets
partition = # IDs
#{'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
labels = # Labels
#{'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}

# Generators
training_generator = DataGenerator(**params).generate(labels, partition['train'])
validation_generator = DataGenerator(**params).generate(labels, partition['validation'])

# Design model
model = NPM.create_model(MAX_QRY_LENGTH, MAX_DOC_LENGTH, NUM_OF_FEATURE, PSG_SIZE, NUM_OF_FILTERS, tau)
model.compile(optimizer = "adadelta", loss = "categorical_crossentropy", metrics=["accuracy"])

# Train model on dataset
model.fit_generator(generator = training_generator,
                    steps_per_epoch = len(partition['train'])//batch_size,
                    validation_data = validation_generator,
                    validation_steps = len(partition['validation'])//batch_size)