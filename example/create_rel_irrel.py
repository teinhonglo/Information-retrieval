# -*- coding: utf-8 -*-
#!/usr/bin/env python
import numpy as np
import sys
sys.path.append("../Tools")

import ProcDoc
import Evaluate
import os

is_training = True
is_short = False
is_spoken = False

output_dir = "data"
output_name = ""

if not os.path.isdir(output_dir):
    os.mkdir(output_dir, 0755)

if is_training:
    qry_path = "../Corpus/TDT2/Train/XinTrainQryTDT2/QUERY_WDID_NEW"
    rel_path = "../Corpus/TDT2/Train/QDRelevanceTDT2_forHMMOutSideTrain"
    output_name = "train"
else:
    if is_short:
        qry_path = "../Corpus/TDT2/QUERY_WDID_NEW_middle"
        output_name = "test_short"
    else:
        qry_path = "../Corpus/TDT2/QUERY_WDID_NEW"
        output_name = "test_long"
    rel_path = "../Corpus/TDT2/AssessmentTrainSet/AssessmentTrainSet.txt"

output_eval_name = output_name + ".all"

if is_spoken:
    doc_path = "../Corpus/TDT2/Spoken_Doc"
    output_name = output_name + ".spk.csv"
    output_eval_name = output_eval_name + ".spk.csv"
else:
    doc_path = "../Corpus/TDT2/SPLIT_DOC_WDID_NEW"
    output_name = output_name + ".csv"
    output_eval_name = output_eval_name + ".csv"

dict_path = "../Corpus/TDT2/LDC_Lexicon.txt"
ID_map = {}

def ID2Word(proc_dict, ID_map):
    for key, content in proc_dict.items():
        for i, ID in enumerate(content):
           content[i] = ID_map[ID]
    return proc_dict

# read relevant set for queries and documents
eval_mdl = Evaluate.EvaluateModel(rel_path, is_training)
rel_set = eval_mdl.getAset()

# read queris and documents
qry_file = ProcDoc.readFile(qry_path)
doc_file = ProcDoc.readFile(doc_path)

# preprocess + reserve postion infomation
qry_mdl_dict = ProcDoc.qryPreproc(qry_file, rel_set, True)
doc_mdl_dict = ProcDoc.docPreproc(doc_file, True)

# read dictionary (ID, Word)
import codecs
with codecs.open(dict_path, 'r', encoding='utf-8') as rf:
    for idx, line in enumerate(rf.readlines()):
        info = line.split("\r\n")[0].split(" ")
        ID_map[idx] = info[-1]

qry_mdl_dict = ID2Word(qry_mdl_dict, ID_map)
doc_mdl_dict = ID2Word(doc_mdl_dict, ID_map)
docs_list = doc_mdl_dict.keys()

print(output_dir + "/" + output_name, output_dir + "/" + output_eval_name)

# create relevant and irrelevant set
with codecs.open(output_dir + "/" + output_name, "w", encoding="utf-8") as wf:
    # iter the relevant set
    wf.write("qry,qry_content,doc,doc_content,rel\n")
    for qry, q_rel_docs in rel_set.items():
        for doc in q_rel_docs:
            # ===================REL DOCS====================
            # prepare prefix query
            wf.write(qry)
            wf.write(",")
            for qry_word in qry_mdl_dict[qry]:
                wf.write(qry_word)
            wf.write(",")
            wf.write(doc)
            wf.write(",")
            # irrelvant docs
            for doc_word in doc_mdl_dict[doc]:
                wf.write(doc_word)
            wf.write(",")
            wf.write("1\n")
            # ===================IRREL DOCS====================
            # prepare prefix query
            wf.write(qry)
            wf.write(",")
            for qry_word in qry_mdl_dict[qry]:
                wf.write(qry_word)
            wf.write(",")
            # irrelvant docs
            choice = -1
            while choice == -1 or docs_list[choice] in q_rel_docs:
                choice = np.random.randint(len(docs_list), size=1)[0]
            r_doc = docs_list[choice]
            wf.write(r_doc)
            wf.write(",")
            for doc_word in doc_mdl_dict[r_doc]:
                wf.write(doc_word)
            wf.write(",")
            wf.write("0\n")

if not is_training:
    with codecs.open(output_dir + "/" + output_eval_name, "w", encoding="utf-8") as wf:
        # iter the relevant set
        wf.write("qry,qry_content,doc,doc_content,rel\n")
        for qry, q_rel_docs in rel_set.items():
            for doc in docs_list:
                # prepare prefix query
                wf.write(qry)
                wf.write(",")
                for qry_word in qry_mdl_dict[qry]:
                    wf.write(qry_word)
                wf.write(",")
                wf.write(doc)
                wf.write(",")
                # irrelvant docs
                for doc_word in doc_mdl_dict[doc]:
                    wf.write(doc_word)
                wf.write(",")
                if doc in q_rel_docs:
                    wf.write("1\n")
                else:
                    wf.write("0\n")
