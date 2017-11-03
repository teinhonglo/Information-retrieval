import operator
import numpy as np
import ProcDoc
from collections import defaultdict
from math import log
import cPickle as Pickle
import os

data = {}				# content of document (doc, content)
doc_model = {}	# word count of 2265 document (word, number of words)
query_model = {}				# query
vocabulary = np.zeros(51253)


with open("model/doc_model_s.pkl", "rb") as file: doc_model = Pickle.load(file)
with open("model/test_query_model_short.pkl", "rb") as file: query_model = Pickle.load(file)

doc_tf_log = np.log(doc_model + 1)
qry_tf_log = np.log(query_model + 1)

doc_idf = np.log((2265 + 0.1)/ ((1 * (doc_model != 0)).sum(axis = 0) + 0.1))

# re-scale
doc_min, doc_max = np.min(doc_model, axis = 0), np.max(doc_model, axis = 0)
new_doc_model = (doc_model - doc_min) / (doc_max - doc_min)
qry_min, qry_max = np.min(query_model, axis = 0), np.max(query_model, axis = 0)
new_query_model = (query_model - qry_min) / (qry_max - qry_min)

# filter nan
where_are_NaNs = np.isnan(doc_tf_log)
doc_tf_log[where_are_NaNs] = 0

# filter nan
where_are_NaNs = np.isnan(qry_tf_log)
qry_tf_log[where_are_NaNs] = 0

print (1 * np.isnan(doc_tf_log)).sum(axis = 1).sum(axis = 0)
print (1 * np.isnan(qry_tf_log)).sum(axis = 1).sum(axis = 0)


with open("model/log_test_query_model_short.pkl", "wb") as file: Pickle.dump(qry_tf_log, file, True)
with open("model/log_doc_model_s.pkl", "wb") as file: Pickle.dump(doc_tf_log, file, True)
