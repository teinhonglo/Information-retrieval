import operator
import numpy as np
from numpy import inf
import ProcDoc
from math import log
import cPickle as Pickle

np.random.seed(1331)

q_tf = np.random.rand(3, 10)
print q_tf
print np.max(q_tf, axis=1)
q_tf = 0.5 + 0.5 * q_tf / np.max(q_tf, axis=1)[:, None]
tf = np.random.randint(2, size=(5, 10))
print tf
idf = np.log(tf.shape[0] / (tf != 0).sum(axis = 0))
idf[idf == -inf] = 0
print idf

# document term weights
doc_model = tf * idf[:,None]