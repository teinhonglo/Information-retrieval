#-*- coding: utf-8 -*-
''' Import theano and numpy '''
import numpy as np
np.random.seed(1331)

import theano
import cPickle as pickle
'''
from keras.callbacks import ModelCheckpoint
modelCheckpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
'''
optimizer = ["Adagrad" , "Adam", "Nadam"]
losses = ["categorical_crossentropy", "kullback_leibler_divergence"]

#execfile('preprocess.py')
with open("query_model.pkl", "rb") as file: query_model = pickle.load(file)
X_train = query_model
with open("obj_func/rel_swlm_entropy_s_9.pkl", "rb") as file: query_relevance = pickle.load(file)
Y_train = query_relevance

''' set the size of mini-batch and number of epochs'''
batch_size = 16
nb_epoch = 55
vocabulary_size = 51253
embedding_dimensionality = 350


''' Import keras to build a DL model '''
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

print 'Building a model whose optimizer=adam, activation function=softmax'
model = Sequential()
model.add(Dense(embedding_dimensionality, input_dim = vocabulary_size))
model.add(Dense(vocabulary_size))
model.add(Activation('softmax'))


''' Setting optimizer as Adam '''
from keras.optimizers import Adam, SGD
model.compile(	loss= 'kullback_leibler_divergence',
		optimizer='Nadam',
		metrics=['accuracy'])

model.summary()
''' Fit models and use validation_split=0.1 '''
history_adam = model.fit(X_train, Y_train,
			batch_size=batch_size,
			nb_epoch=nb_epoch,
			verbose=1,
			shuffle=True,
			validation_split=0.1
			#callbacks = [modelCheckpoint]
			# earlyStopping callbacks
			#callbacks = [earlyStopping]
			)
''' Create a HDF5 file '''							
model.save('NN_Model/RLE_SSWLM_S.h5')

loss = history_adam.history.get('loss')
acc = history_adam.history.get('acc')
val_loss = history_adam.history.get('val_loss')
val_acc = history_adam.history.get('val_acc')

''' Visualize the loss and accuracy of both models'''
'''
import matplotlib.pyplot as plt
plt.figure(5)
plt.subplot(121)
plt.plot(range(len(loss)), loss,label='Training')
plt.plot(range(len(val_loss)), val_loss,label='Validation')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc)), acc,label='Training')
plt.plot(range(len(val_acc)), val_acc,label='Validation')
plt.title('Accuracy')
#plt.show()
plt.savefig('07_earlystopping.png',dpi=300,format='png')

print 'Result saved into 07_earlystopping.png'
'''
