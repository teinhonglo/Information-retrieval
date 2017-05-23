import os
import fileinput
import collections

class evaluate_model():
	def __init__(self):
		self.assessmentTraingSet_path = "../Corpus/AssessmentTrainSet"
		self.assessment = self.get_assessment()
		

	def get_assessment(self):
		assessmentTraingSetDict = {}
		# 16 query documants
		for assessment_item in os.listdir(self.assessmentTraingSet_path):
			# join dir path and file name
			assessment_item_path = os.path.join(self.assessmentTraingSet_path, assessment_item)
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
	def precision(self, result, assessment_list):
		iterative = 0
		count = 0
		precision = 0
		assessment = assessment_list.split()
		for doc_name in result:
			iterative += 1
			if count == len(assessment): break
				
			if doc_name in assessment:
				count += 1
				precision += count * 1.0 / iterative
		
		precision /= len(assessment)
		return precision
		
	def mean_average_precision(self, query_docs_point_dict):
		mAP = 0
		AP = 0
		assessment = self.assessment
		for q_key, docs_point_list in query_docs_point_dict.items():
			AP += self.precision(docs_point_list, assessment[q_key])
		mAP = AP / len(query_docs_point_dict.keys())
		return mAP