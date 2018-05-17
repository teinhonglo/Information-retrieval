import os
import fileinput
import collections
import readAssessment
resultTraingSet_path = "ResultsTrainSet"
resultTraingSetDict = {}

def get_result():
	# 16 query documants
	for result_item in os.listdir(resultTraingSet_path):
		# join dir path and file name
		result_item_path = os.path.join(resultTraingSet_path, result_item)
		# check whether a file exists before read
		if os.path.isfile(result_item_path):
			with open(result_item_path, 'r') as f:
				# read content of query documant (doc, content)
				title = "query"
				count = 0
				rst_list = []
				for line in f.readlines():
					if "Query" in line.split():
						rst_list = []
						words = line.split()
						title = words[2]
						resultTraingSetDict[title] = ""
					else:
						words = line.split()
						if len(words) < 1: 
							resultTraingSetDict[title] = rst_list
							continue
						
						rst_list.append([words[0], words[1]])
	
	return resultTraingSetDict
	
results = get_result()
assessment = readAssessment.get_assessment()
precision_sum = 0
results = collections.OrderedDict(sorted(results.items()))
for key, value in results.items():
	print key
	precision_sum += readAssessment.precision(value , assessment[key])
print precision_sum	/ 16