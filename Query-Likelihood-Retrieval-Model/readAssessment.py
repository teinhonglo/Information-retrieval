import os
import fileinput
import collections
assessmentTraingSet_path = "../Corpus/TDT2/AssessmentTrainSet"
assessmentTraingSetDict = {}

def get_assessment():
	# 16 query documants
	for assessment_item in os.listdir(assessmentTraingSet_path):
		# join dir path and file name
		assessment_item_path = os.path.join(assessmentTraingSet_path, assessment_item)
		# check whether a file exists before read
		if os.path.isfile(assessment_item_path):
			with open(assessment_item_path, 'r') as f:
				# read content of query documant (doc, content)
				title = "query"
				for line in f.readlines():
					if "Query" in line.split():
						words = line.split()
						title = words[2]
						assessmentTraingSetDict[title] = ""
					else:
						assessmentTraingSetDict[title] += line
	
	return assessmentTraingSetDict

# result : list [(doc, point)]
# assessment_list : list [(doc)]
def precision(result, assessment_list):
	iterative = 0
	count = 0
	precision = 0
	assessment = assessment_list.split()
	for doc_name, point in result:
		iterative += 1
		if count == len(assessment): break
			
		if doc_name in assessment:
			count += 1
			precision += count * 1.0 / iterative
	
	precision /= len(assessment)
	return precision