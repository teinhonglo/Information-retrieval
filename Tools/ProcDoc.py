# -*- coding: utf-8 -*-
import codecs
import io
import os
import fileinput
import collections
import numpy as np
import operator
import types
from math import exp
from webbrowser import BackgroundBrowser
from collections import defaultdict
from math import log

bg_modle_path = "../Information-retrieval/Corpus/background"
Cluster_path = "Topic"


# read file(query or document)
def readFile(filepath):
    data = {}                # content of document (doc, content)
    # list all files of a directory(Document)
    for dir_item in os.listdir(filepath):
        # join dir path and file name
        dir_item_path = os.path.join(filepath, dir_item)
        # check whether a file exists before read
        if os.path.isfile(dir_item_path):
            with open(dir_item_path, 'r') as f:
                # read content of document (doc, content)
                data[dir_item] = f.read()
    # data(dict)
    return data    

# read background model
def readRELdict(REL_PATH = None, isTraining = True):
    rel_dict = defaultdict(list)
    if REL_PATH == None: 
        REL_PATH = "../Corpus/TDT2/Train/QDRelevanceTDT2_forHMMOutSideTrain"
    with io.open(REL_PATH, 'r', encoding = 'utf8') as file:
        # read content of query document (doc, content)
        qry_name = ""
        for line in file.readlines():
            result = line.split()
            if len(result) == 0:
                continue
            if len(result) > 1:
                if isTraining:
                    qry_name = result[1]
                else:
                    qry_name = result[2]
                continue
            rel_dict[qry_name].append(result[0])
    # HMMTraingSetDict{word, probability}
    return rel_dict    

# read background model
def readBGdict():
    BGTraingSet = np.zeros(51253)
    # XIN1998
    for doc_item in os.listdir(bg_modle_path):
        # join dir path and file name
        doc_item_path = os.path.join(bg_modle_path, doc_item)
        # check whether a file exists before read
        if os.path.isfile(doc_item_path):
            with io.open(doc_item_path, 'r', encoding = 'utf8') as f:
                # read content of query document (doc, content)
                lines = f.readlines()
                for line in lines:
                    [id, prob] = line.split()
                    prob = exp(float(prob))
                    BGTraingSet[int(id)] = prob
    # Background{word, probability}
    return np.array([BGTraingSet])

# document preprocess
def docPreproc(dictionary, res_pos = False, topN = None):
    # parameter:
    # dictionary: doc {d_id: string content, ...}
    # rel_pos: bool, reserved position
    # topN: int, top N word
    # output : if res_pos == TRUE : return doc {d_id: [int wID0, int wID3 , int wID0], ...}
    #          if res_pos == FALSE : return qry {d_id: [int wID0:, int wID0_count, int wID3:, int wID3_count], ...}
    doc_new = {}
    for key, value in dictionary.items():
        content = ""
        temp_content = ""
        count = 0
        # split content by special character
        for line in dictionary[key].split('\n'):
            if count < 3:
                count += 1
                continue
            else:    
                for word in line.split('-1'):
                    temp_content += word + " "
        # delete double white space
        for word in temp_content.split():
            content += word + " "

        # content to int list
        int_rep = map(int, content.split())
        # topN
        if topN != None and len(int_rep) > topN:
            int_rep = int_rep[:topN]
        # replace old content    
        doc_new[key] = int_rep
        
    if not res_pos:
        doc_freq = {}    
        # term probability(word_count / word sum)    
        for doc_key, doc_content in doc_new.items():
            doc_words = wordCount(doc_content, {})
            doc_new[doc_key] = doc_words
        #dictionary = TFIDF(dictionary)    
        
    return doc_new

# query preprocess
def qryPreproc(dictionary, rel_set = None, res_pos = False, topN = None):
    # parameter:
    # dictionary: query {q_id: string content, ...}
    # rel_set: relevance {q_id: [d_id2, d_id3], ...}
    # rel_pos: bool, reserved position
    # topN: int, top N word
    # output : if res_pos == TRUE : return qry {q_id: [int wID0, int wID3 , int wID0], ...}
    #          if res_pos == FALSE : return qry {q_id: [int wID0:, int wID0_count, int wID3:, int wID3_count], ...}
    qry_new = {}
    for key, value in dictionary.items():
        if rel_set != None and len(rel_set[key]) == 0: continue
        content = ""
        temp_content = ""
        # split content by special character
        for line in dictionary[key].split('\n'):
            for word in line.split('-1'):
                temp_content += word + " "
        # delete double white space
        for word in temp_content.split():
            content += word + " "
        # content to int list
        int_rep = map(int, content.split())
        # topN
        if topN != None and len(int_rep) > topN: 
            int_rep = int_rep[:topN]
        # replace old content    
        qry_new[key] = int_rep
    if not res_pos:    
        qry_freq = {}    
        # term probability(word_count / word sum)    
        for qry_key, qry_content in qry_new.items():
            qry_words = wordCount(qry_content, {})
            qry_new[qry_key] = qry_words
        #dictionary = TFIDF(dictionary)    
    return qry_new

# word count
def wordCount(content, bg_word):
    for part in content:
        if part in bg_word:
            bg_word[part] += 1
        else:
            bg_word[part] = 1
    # return word count dictionary        
    return bg_word

def docFreq(doc, vocab_size = 51253):
    #0:docfreq 1:count
    corpus_dFreq_total = np.zeros((vocab_size, 2))
    for name, word_list in doc.items():
        temp_word_list = {}
        cont_type = type(word_list)
        # str to dict
        if isinstance(word_list, types.StringType):
            temp_word_list = word_count(word_list, {})
        # list to dict
        elif isinstance(word_list, types.ListType):
            temp_word_list = {}
            for part in word_list:
                if part in temp_word_list:
                    temp_word_list[part] += 1
                else:
                    temp_word_list[part] = 1
        elif isinstance(word_list, types.DictType):
            temp_word_list = dict(word_list)
        # assume type of word_list is dictionary
        for word, word_count in temp_word_list.items():
            corpus_dFreq_total[int(word), 0] += 1
            corpus_dFreq_total[int(word), 1] += word_count
    return corpus_dFreq_total

def rmStopWord(ori_content, corpus_dFreq_total, threshold = 0.1):
    weight_list = []
    corpus_length = corpus_dFreq_total[:, 1].sum(axis=0)
    cont_type = type(dict)
    for name, word_list in ori_content:
        cont_type = type(word_list)
        # str to dict
        if cont_type == type(str):
            temp_word_list = word_count(word_list, {})
        # list to dict
        elif cont_type == type(list):
            temp_word_list = {}
            for part in word_list.split():
                if part in temp_word_list:
                    temp_word_list[part] += 1
                else:
                    temp_word_list[part] = 1
        elif cont_type == type(dict):
            temp_word_list = dict(word_list)
        # assume type of word_list is dictionary
        cur_length = sum(temp_word_list.values())
        for word, word_count in temp_word_list.items():
            word_prob = word_count * 1.0 / cur_length
            corpus_word_prob = corpus_dFreq_total[int(word)][0] * 1.0 / corpus_length
            weight = word_prob * log(word_prob / corpus_word_prob)
            weight_list.append([name, word, weight])
    
    sorted(weight_list, key = lambda x : x[2])
    len_weight_list = len(weight_list)
    for i in xrange(len_weight_list * threshold):
        [name, word, weight] = weight_list[i]
        word_list = ori_content[name]
        # remove low weighted word(string)
        if cont_type == type(str):
            temp_list = word_list.replace(word + " ", "")
            word_list = temp_list.replace(" " + word, "")
        # remove low weighted word(list)
        elif cont_type == type(list):
            word_list = filter(lambda a: a != word, word_list)
        # remove low weighted word(dict)
        elif cont_type == type(dict):
            word_list.pop(word, None)
        # assign new value to name    
        ori_content[name] = word_list    
            
    return ori_content


# create unigram
def unigram(topic_wordcount_dict):
    topic_wordprob_dict = {}
    for topic, wordcount in topic_wordcount_dict.items():
        length = 1.0 * sum(wordcount.values())
        word_prob = {}
        for word, count in wordcount.items():
            word_prob[word] = count / length
        topic_wordprob_dict[topic] = word_prob
    topic_wordprob_dict = collections.OrderedDict(sorted(topic_wordprob_dict.items()))    
    return topic_wordprob_dict 

# modeling    
def smoothing(topic_wordprob_dict, background_model, alpha):
    modeling_dict = {}
    for topic, wordprob in topic_wordprob_dict.items():
        word_model = {}
        for word in wordprob.keys():
            word_model[word] = (1-alpha) * wordprob[word] + (alpha) * background_model[word]
        modeling_dict[topic] = dict(word_model)
    return modeling_dict

# softmax            
def softmax(model):
    model_word_sum  = 1.0 * sum(wordcount.values())
    model = {w: c / model_word_sum for w, c in dict(model).items()}
    return model    
