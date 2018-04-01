import matplotlib.pyplot as plt
from Evaluate import EvaluateModel
from collections import defaultdict
import numpy as np
import math

TraingSet_path = "../Corpus/ResultsTrainSet/ResultsTrainSet.txt"
TraingSetDict = defaultdict(list)
with open(TraingSet_path, 'r') as file:
    # read content of query document (doc, content)
    title = ""
    for line in file.readlines():
        result = line.split()
                
        if len(result) == 0:
            continue
        if len(result) > 2:
            title = result[2]
            continue
        TraingSetDict[title].append(result[0])

eval = EvaluateModel("../Corpus/TDT2/AssessmentTrainSet/AssessmentTrainSet.txt", False)
assess = eval.getAss()
P_R_table = np.zeros(11)
for q_key, results in TraingSetDict.items():
    p_max = 0
    num_correct = 0.
    start_recall = 0
    recall_acc = 0.
    for pos, doc_name in enumerate(results):
        t_pos = pos + 1
        if doc_name in assess[q_key]:
            num_correct += 1.
            precision = num_correct / t_pos
            recall_acc = num_correct / len(assess[q_key])
        if precision > p_max:
            p_max = precision
        isAssign = False
        end_recall = int(math.floor(recall_acc * 10))
        for i in range(start_recall, end_recall):
            P_R_table[start_recall] += p_max
            start_recall += 1
            isAssign = True
        
        if isAssign:
            if end_recall == 10:
                P_R_table[start_recall] += p_max
            p_max = 0

plt.plot(P_R_table / 16, '-o')
plt.show()