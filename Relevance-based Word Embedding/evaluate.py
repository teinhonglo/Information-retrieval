import os
import fileinput
import collections
from collections import defaultdict

class evaluate_model():
	def __init__(self):
		self.assessmentTraingSet_path = "../Corpus/Train/QDRelevanceTDT2_forHMMOutSideTrain"
		self.assessment = self.get_assessment()
		
	def get_assessment(self):
		assessmentTraingSetDict = defaultdict(list)
		assessmentTraingSet_path = self.assessmentTraingSet_path
		with open(assessmentTraingSet_path, 'r') as file:
			# read content of query documant (doc, content)
			title = ""
			for line in file.readlines():
				result = line.split()
				if len(result) > 1:
					title = result[1]
					continue
				assessmentTraingSetDict[title].append(result[0])
		
		return assessmentTraingSetDict

	# result : list [(doc, point)]
	# assessment_list : list [(doc)]
	def precision(self, result, q_key):
		iterative = 0
		count = 0
		precision = 0
		assessment = self.assessment[q_key]
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
		for q_key, docs_point_list in query_docs_point_dict.items():
			AP += self.precision(docs_point_list, q_key)
		mAP = AP / len(query_docs_point_dict.keys())
		return mAP

if __name__ == "__main__":
	eva = evaluate_model()