import csv
import random
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

def preprocess():
	# data attributes
	buying_numeralization 	= {"low"	: 0	, "med" : 1	, "high" : 2	, "vhigh" : 3}
	maint_numeralization 	= {"low"	: 0	, "med" : 1	, "high" : 2	, "vhigh" : 3}
	doors_numeralization 	= {"2" 		: 2	, "3" 	: 3	, "4" 	 : 4	, "5more" : 5}
	persons_numeralization 	= {"2" 		: 2	, "4" 	: 4	, "more" : 5}
	lug_boot_numeralization = {"small" 	: 0	, "med"	: 1	, "big"  : 2}
	safety_numeralization 	= {"low" 	: 0	, "med"	: 1	, "high" : 2}
	# class
	class_numeralization 	= {"unacc" 	: 0	, "acc" : 1	, "good" : 2	, "vgood" : 3}
	# data and predict target
	data 	= []
	target 	= []

	# Read Data and Preprocess
	with open('car.data.csv') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			# Numeralization
			row['buying'] 	= buying_numeralization[row['buying']]
			row['maint'] 	= maint_numeralization[row['maint']]
			row['doors'] 	= doors_numeralization[row['doors']]
			row['persons'] 	= persons_numeralization[row['persons']]
			row['lug_boot'] = lug_boot_numeralization[row['lug_boot']]
			row['safety'] 	= safety_numeralization[row['safety']]
			row['class'] 	= class_numeralization[row['class']]
			
			# print(row['buying'], row['maint'], row['doors'], row['persons'], row['lug_boot'], row['safety'])
			data.append([row['buying'], row['maint'], row['doors'], row['persons'], row['lug_boot'], row['safety']])
			target.append(row['class'])
	
	# shuffle
	random.seed(5000)
	combine_shuffle = list(zip(data, target))
	random.shuffle(combine_shuffle)
	data, target	= zip(*combine_shuffle)
	data, target 	= np.array(data), np.array(target)
	
	# normalization
	'''
	attr_max = data.max(axis = 0)
	attr_min = data.min(axis = 0)
	data = (data - attr_min) / (1.0 * (attr_max - attr_min))
	'''
	return [data, target]

