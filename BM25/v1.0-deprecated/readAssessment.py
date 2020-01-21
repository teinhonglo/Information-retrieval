import os
import fileinput
import collections
from collections import defaultdict
#assessmentTraingSet_path = "../Corpus/TDT2/Train/QDRelevanceTDT2_forHMMOutSideTrain"
assessmentTraingSet_path = "../../Corpus/TDT2/AssessmentTrainSet/AssessmentTrainSet.txt"
assessmentTraingSetDict = {}

# result : list [(doc, point)]
# assessment_list : list [(doc)]
def getAssessment(HMM = False):
        answerTraingSetDict = defaultdict(list)
        answerTraingSet_path = assessmentTraingSet_path
        with open(answerTraingSet_path, 'r') as f:
            # read content of query document (doc, content)
            title = ""
            for line in f.readlines():
                result = line.split()              
                if len(result) == 0:
                    continue
                if len(result) > 1:
                    if not HMM:
                        title = result[2]
                    else:
                        title = result[1]
                    continue

                answerTraingSetDict[title].append(result[0])
                
        return answerTraingSetDict

def precision(result, assessment):
    iterative = 0
    count = 0
    precision = 0
    for doc_name, point in result:
        iterative += 1
        if count == len(assessment): break
            
        if doc_name in assessment:
            count += 1
            precision += count * 1.0 / iterative
    
    precision /= len(assessment)
    return precision
    
def mean_average_precision(query_docs_point_dict, assessment):
    mAP = 0
    AP = 0
    for q_key, docs_point_list in query_docs_point_dict.items():
        AP += precision(docs_point_list, assessment[q_key])
    mAP = AP / len(query_docs_point_dict.keys())
    return mAP
