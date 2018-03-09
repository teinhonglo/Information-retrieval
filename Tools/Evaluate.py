import os
import fileinput
import collections
from collections import defaultdict

class EvaluateModel(object):
    def __init__(self, rel_set_path = None, HMM = False):
        # TDT2 path ../Corpus/TDT2/AssessmentTrainSet/AssessmentTrainSet.txt
        # HMMTrainingSet path ../Corpus/TDT2/Train/QDRelevanceTDT2_forHMMOutSideTrain
        if rel_set_path == None:
            self.assessmentTraingSet_path = "../Corpus/TDT2/AssessmentTrainSet/AssessmentTrainSet.txt"
        else:
            self.assessmentTraingSet_path = rel_set_path
        self.assessment = self.__getAssessment(HMM)
        
    def __getAssessment(self, HMM):
        assessmentTraingSetDict = defaultdict(list)
        assessmentTraingSet_path = self.assessmentTraingSet_path
        with open(assessmentTraingSet_path, 'r') as file:
            # read content of query document (doc, content)
            title = ""
            for line in file.readlines():
                result = line.split()
                
                if len(result) == 0:
                    continue
                if len(result) > 1:
                    if not HMM:
                        title = result[2]
                    else:
                        title = result[1]
                    continue

                assessmentTraingSetDict[title].append(result[0])
        return assessmentTraingSetDict

    # result : list [(doc, point)]
    # assessment_list : list [(doc)]
    def __avePrecision(self, result, q_key):
        iterative = 0
        count = 0
        precision = 0
        assessment = self.assessment[q_key]
        for doc_name, point in result:
            iterative += 1
            if count == len(assessment): break
                
            if doc_name in assessment:
                count += 1
                precision += count * 1.0 / iterative
        
        precision /= len(assessment)
        return precision
        
    def mAP(self, query_docs_point_dict):
        mAP = 0
        AP = 0
        for q_key, docs_point_list in query_docs_point_dict.items():
            AP += self.__avePrecision(docs_point_list, q_key)
        mAP = AP / len(query_docs_point_dict.keys())
        return mAP
    
    def NDCG(self, query_docs_point_dict):
        pass
        
if __name__ == "__main__":
    eva = evaluate_model()
