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
from collections import defaultdict

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

parser.add_argument("--bm25_method",
		     default="BM25",
		     type=str,
		     required=False)

parser.add_argument("--topN",
                     default="10",
                     type=str,
                     required=False)

parser.add_argument("--att_type",
                     default="uni",
                     type=str,
                     required=False)

args = parser.parse_args()

is_training = args.is_training
is_short = args.is_short
is_spoken = args.is_spoken
task_name = args.task_name

BM25_method = args.bm25_method
topN = args.topN

att = args.att_type
att_types = ["uni", "conf"]
 
if att_types.index(att) == 0:
    att_prefix = "uni_"
elif att_types.index(att) == 1:
    att_prefix = "conf_"
    conf_path = "../Corpus/TDT2/SPLIT_AS0_WDID_NEW_POWER"

output_name = ""
pseudo_prefix = str(topN) + "_" + BM25_method + "_"

output_dir = args.output_dir + "/" + task_name

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

if is_training:
    qry_path = "../Corpus/TDT2/Train/XinTrainQryTDT2/QUERY_WDID_NEW"
    rel_path = "exp/BM25/pseudo_" + str(topN) + "_" + BM25_method
    if is_spoken:
        rel_path += "_spk.txt"
    else:
        rel_path += ".txt"
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

def ID2Word(proc_dict, ID_map, score_dict):
    att_dict = defaultdict(list)
    for key, content in proc_dict.items():
        for i, ID in enumerate(content):
           content[i] = ID_map[ID]
           if ID == -1: continue
           att_score = score_dict[key][ID]
           for j in range(len(ID_map[ID])):
              att_dict[key].append(att_score)
    return proc_dict, att_dict

# read relevant set for queries and documents
eval_mdl = Evaluate.EvaluateModel(rel_path, False)
rel_set = eval_mdl.getAset()

# read queris and documents
qry_file = ProcDoc.readFile(qry_path)
doc_file = ProcDoc.readFile(doc_path)

# preprocess + reserve postion infomation
qry_mdl_dict = ProcDoc.qryPreproc(qry_file, rel_set, True, True)
doc_mdl_dict = ProcDoc.docPreproc(doc_file, True, True)

# bag of word
qry_bow_dict = ProcDoc.qryPreproc(qry_file, rel_set)
doc_bow_dict = ProcDoc.docPreproc(doc_file)

# unigram
if att_types.index(att) == 0:
    qry_att_dict = ProcDoc.unigram(qry_bow_dict)
    doc_att_dict = ProcDoc.unigram(doc_bow_dict)
elif att_types.index(att) == 1:
    qry_att_dict = qry_bow_dict
    for q_key, q_cont in qry_att_dict.items():
        for q_w, q_w_uni in q_cont.items():
            qry_att_dict[q_key][q_w] = "1.0"
    if is_spoken:
        doc_conf_file = ProcDoc.readFile(conf_path)
        doc_att_dict = ProcDoc.confPreproc(doc_conf_file)
    else:
        doc_att_dict = doc_bow_dict
        for d_key, d_cont in doc_att_dict.items():
            for d_w, d_w_uni in d_cont.items():
                doc_att_dict[d_key][d_w] = "1.0"

# read dictionary (ID, Word)
import codecs
with codecs.open(dict_path, 'r', encoding='utf-8') as rf:
    for idx, line in enumerate(rf.readlines()):
        info = line.split("\r\n")[0].split(" ")
        ID_map[idx] = info[-1]
    ID_map[-1] = ":"

qry_mdl_dict, qry_att_dict = ID2Word(qry_mdl_dict, ID_map, qry_att_dict)
doc_mdl_dict, doc_att_dict = ID2Word(doc_mdl_dict, ID_map, doc_att_dict)
docs_list = doc_mdl_dict.keys()

print(output_dir + "/" + pseudo_prefix + output_name, output_dir + "/" + output_eval_name)

def write_content_to_file(wf, mdl_dict):
    for word in mdl_dict:
        wf.write(word)

def write_att_to_file(wf, att_dict):
    for idx, att in enumerate(att_dict):
        wf.write(str(att))
        if idx < len(att_dict) - 1:
            wf.write(":")
    return

# create relevant and irrelevant set (pointwise)
with codecs.open(output_dir + "/" + pseudo_prefix + output_name, "w", encoding="utf-8") as wf:
    # iter the relevant set
    wf.write("qry,qry_content,doc,doc_content,rel\n")
    for qry, q_rel_docs in rel_set.items():
        for doc in q_rel_docs:
            # ===================REL DOCS====================
            # prepare prefix query
            wf.write(qry + ",")
            write_content_to_file(wf, qry_mdl_dict[qry])
            wf.write("," + doc + ",")
            # relevant docs
            write_content_to_file(wf, doc_mdl_dict[doc])
            wf.write(",1\n")
            # ===================IRREL DOCS====================
            # prepare prefix query
            wf.write(qry + ",")
            write_content_to_file(wf, qry_mdl_dict[qry])
            wf.write(",")
            # irrelevant docs
            choice = -1
            while choice == -1 or docs_list[choice] in q_rel_docs:
                choice = np.random.randint(len(docs_list), size=1)[0]
            r_doc = docs_list[choice]
            wf.write(r_doc + ",")
            write_content_to_file(wf, doc_mdl_dict[r_doc])
            wf.write(",0\n") 
            
print(output_dir + "/" + pseudo_prefix + att_prefix + output_name, output_dir + "/" + att_prefix + output_eval_name)

# attention for query model and document model
# create relevant and irrelevant set (pointwise)
with codecs.open(output_dir + "/" + pseudo_prefix + att_prefix + output_name, "w", encoding="utf-8") as wf:
    # iter the relevant set
    wf.write("qry,qry_content,qry_att,doc,doc_content,doc_att,rel\n")
    for qry, q_rel_docs in rel_set.items():
        for doc in q_rel_docs:
            # ===================REL DOCS====================
            # prepare prefix query
            wf.write(qry + ",")
            write_content_to_file(wf, qry_mdl_dict[qry])
            wf.write(",")
            # attetion
            write_att_to_file(wf, qry_att_dict[qry])
            wf.write("," + doc + ",")
            # relevant docs
            write_content_to_file(wf, doc_mdl_dict[doc])
            wf.write(",")
            # attention
            write_att_to_file(wf, doc_att_dict[doc])
            wf.write(",1\n")
            # ===================IRREL DOCS====================
            # prepare prefix query
            wf.write(qry + ",")
            write_content_to_file(wf, qry_mdl_dict[qry])
            wf.write(",")
            # attention
            write_att_to_file(wf, qry_att_dict[qry])
            wf.write(",")
            # irrelevant docs
            choice = -1
            while choice == -1 or docs_list[choice] in q_rel_docs:
                choice = np.random.randint(len(docs_list), size=1)[0]
            r_doc = docs_list[choice]
            wf.write(r_doc + ",")
            write_content_to_file(wf, doc_mdl_dict[r_doc])
            wf.write(",")
            # attention
            write_att_to_file(wf, doc_att_dict[r_doc])
            wf.write(",0\n") 
            
