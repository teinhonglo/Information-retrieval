import csv
import random
import car
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize

# Load Data
[data, target]  = car.preprocess()

# Split Data into Train and Test
SPLIT_THRESHOLD = int(len(data) * 4 / 10)
x_train, x_test	= data	[0 : SPLIT_THRESHOLD], data	 [SPLIT_THRESHOLD:]
y_train, y_test	= target[0 : SPLIT_THRESHOLD], target[SPLIT_THRESHOLD:]

print "SVM"
# SVM
clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
			decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
			max_iter=-1, probability=False, random_state=False, shrinking=True,
			tol=0.001, verbose=False)
			
y_score = clf.fit(x_train, y_train).decision_function(x_test)

print ("mean_squared_error")
print (mean_squared_error(y_test, clf.predict(x_test)))
print

print "accuracy_score"
print (accuracy_score(y_test, clf.predict(x_test)))
print

print "confusion_matrix"
print (confusion_matrix(y_test, clf.predict(x_test)))
print

print "classification_report"
print (classification_report(y_test, clf.predict(x_test)))
print

import plot_precision_recall