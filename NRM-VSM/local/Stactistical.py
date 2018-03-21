#!/usr/bin/env python3
import numpy as np

def TFIDF(qry, doc):
    doc_freq = docFreq(doc)
    num_docs = len(list(doc_new.keys()))
    qry_new = {q_id : [q_wc * np.log(num_docs / (1 + doc_freq(q_wid)))] q_wid, q_wc for q_content.items() 
                        for q_id, q_content in qry.items()}
    doc_new = {d_id : [d_wc * np.log(num_docs / (1 + doc_freq(q_wid)))] d_wid, d_wc for d_content.items() 
                        for d_id, d_content in doc.items()}
    return qry_new, doc_new

def docFreq(doc, vocab_size = 51253):
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

def docLen(doc):
    docs_len = {}
    for d_id, d_cont in doc.items():
        doc_len = 0
		for d_wid, d_wc in d_cont.items():
            doc_len += d_wc
        docs_len[d_id] = d_wc
	return docs_len    