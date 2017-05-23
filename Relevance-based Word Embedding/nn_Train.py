#-*- coding: utf-8 -*-
''' Import theano and numpy '''
import theano
import numpy as np


execfile('preprocess.py')
X_train = query_model
Y_train = query_relevance

''' set the size of mini-batch and number of epochs'''
batch_size = 16
nb_epoch = 30
embedding_dimensionality = 300
vocabulary_size = 51253

''' Import keras to build a DL model '''
from keras.models import Sequential
from keras.layers.core import Dense, Activation

print 'Building a model whose optimizer=adam, activation function=softmax'
model = Sequential()
model.add(Dense(embedding_dimensionality, input_dim = vocabulary_size))
model.add(Dense(vocabulary_size))
model.add(Activation('softmax'))

''' Setting optimizer as Adam '''
from keras.optimizers import Adam
model.compile(loss= 'categorical_crossentropy',
              		optimizer='Adam',
              		metrics=['accuracy'])

model.summary()
''' Fit models and use validation_split=0.1 '''
history_adam = model.fit(X_train, Y_train,
							batch_size=batch_size,
							nb_epoch=nb_epoch,
							verbose=1,
							shuffle=True,
                    		validation_split=0.1
                    		# earlyStopping callbacks
							#callbacks = [earlyStopping]
                    		)
''' Create a HDF5 file '''							
model.save('RLE.h5')

import cPickle as Pickle
# save as JSON
json_string = model.to_json()
Pickle.dump(json_string, open("RLE_model_structure.pkl", "wb"), True)

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
