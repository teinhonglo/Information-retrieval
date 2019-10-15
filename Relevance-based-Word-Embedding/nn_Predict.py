#-*- coding: utf-8 -*-
''' Import theano and numpy '''
import numpy as np

''' Import keras to build a DL model '''
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation

def predict(nn_model, qry_model):
    print 'Building a model whose ' + nn_model
    model = load_model(nn_model)
    rel_model = np.array(model.predict(qry_model))
    return rel_model
