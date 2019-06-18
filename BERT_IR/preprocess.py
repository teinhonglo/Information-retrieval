# -*- coding: utf-8 -*-
#!/usr/bin/env python
import numpy as np
import sys
sys.path.append("../Tools")
np.random.seed(9)

import ProcDoc
import Evaluate
import os
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--output_dir",
                     default="data",
                     type=str)

parser.add_argument("--task_name",
                     default="TDT2",
                     type=str)

parser.add_argument("--is_training",
                     default=None,
                     type=str2bool,
                     required=True)

parser.add_argument("--is_short",
                     default=None,
                     type=str2bool,
                     required=True)

parser.add_argument("--is_spoken",
                     default=None,
                     type=str2bool,
                     required=True)

args = parser.parse_args()

is_training = args.is_training
is_short = args.is_short
is_spoken = args.is_spoken
task_name = args.task_name

output_name = ""
output_dir = args.output_dir + "/" + task_name

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

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


if is_spoken:
    doc_path = "../Corpus/TDT2/Spoken_Doc"
    output_eval_name = output_name + "_spk.all.csv"
    output_name = output_name + "_spk.csv"
else:
    doc_path = "../Corpus/TDT2/SPLIT_DOC_WDID_NEW"
    output_eval_name = output_name + ".all.csv"
    output_name = output_name + ".csv"

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

def write_content_to_file(wf, key, mdl_dict):
    for word in mdl_dict[key]:
        wf.write(word)

# create relevant and irrelevant set (pointwise)
with codecs.open(output_dir + "/" + output_name, "w", encoding="utf-8") as wf:
    # iter the relevant set
    wf.write("qry,qry_content,doc,doc_content,rel\n")
    for qry, q_rel_docs in rel_set.items():
        for doc in q_rel_docs:
            # ===================REL DOCS====================
            # prepare prefix query
            wf.write(qry + ",")
            for qry_word in qry_mdl_dict[qry]:
                wf.write(qry_word)
            wf.write("," + doc + ",")
            # relevant docs
            for doc_word in doc_mdl_dict[doc]:
                wf.write(doc_word)
            wf.write(",1\n")
            # ===================IRREL DOCS====================
            # prepare prefix query
            wf.write(qry + ",")
            for qry_word in qry_mdl_dict[qry]:
                wf.write(qry_word)
            wf.write(",")
            # irrelevant docs
            choice = -1
            while choice == -1 or docs_list[choice] in q_rel_docs:
                choice = np.random.randint(len(docs_list), size=1)[0]
            r_doc = docs_list[choice]
            wf.write(r_doc + ",")
            for doc_word in doc_mdl_dict[r_doc]:
                wf.write(doc_word)
            wf.write(",0\n") 
            
if not is_training:
    with codecs.open(output_dir + "/" + output_eval_name, "w", encoding="utf-8") as wf:
        # iter the relevant set
        wf.write("qry,qry_content,doc,doc_content,rel\n")
        for qry, q_rel_docs in rel_set.items():
            for doc in docs_list:
                # prepare prefix query
                wf.write(qry + ",")
                for qry_word in qry_mdl_dict[qry]:
                    wf.write(qry_word)
                wf.write("," + doc + ",")
                # irrelvant docs
                for doc_word in doc_mdl_dict[doc]:
                    wf.write(doc_word)
                wf.write(",")
                if doc in q_rel_docs:
                    wf.write("1\n")
                else:
                    wf.write("0\n")

'''
num_ir_docs = 3
num_ir_repeats = 3
# create relevant and irrelevant set (pointwise)
with codecs.open(output_dir + "/pairwise." + output_name, "w", encoding="utf-8") as wf:
    # iter the relevant set
    wf.write("tdt2_id,qry_content,doc_0,doc_1,doc_2,doc_3,label\n")
    for qry, q_rel_docs in rel_set.items():
        for doc in q_rel_docs:
            # prepare prefix query (doc = relevant doc)
            tdt2_id = qry + "#" + doc 
            # irrelvant docs
            ir_docs = []
            while len(ir_docs) < num_ir_docs * num_ir_repeats:
                choice = -1
                while (choice == -1) or (docs_list[choice] in q_rel_docs) or (choice in ir_docs):
                    choice = np.random.randint(len(docs_list), size=1)[0]
                ir_docs.append(choice)
            # preparing qry, doc, and irrel docs
            for start_idx in range(num_ir_repeats):
                start_offset = num_ir_repeats * start_idx
                tmp_tdt2_id = tdt2_id
                # a relevant document
                tmp_docs = [doc]
                # irrelvant documents
                for current_idx in range(num_ir_docs):
                   tmp_tdt2_id += "#" + docs_list[start_offset + current_idx]
                   tmp_docs.append(docs_list[start_offset + current_idx])
                
                wf.write(tmp_tdt2_id)
                wf.write(",")
                write_content_to_file(wf, qry, qry_mdl_dict)
                wf.write(",")
                for doc_name in tmp_docs:
                    write_content_to_file(wf, doc_name, doc_mdl_dict)
                    wf.write(",")
                wf.write("0\n")
'''
