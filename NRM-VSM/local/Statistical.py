#!/usr/bin/env python3
import numpy as np
import types

def TFIDF(qry, doc, qry_len, doc_len):
    doc_freq = docFreq(doc)
    num_docs = len(list(doc.keys())) + 1
    qry_new = {q_id : {q_wid : (.5 + .5 * np.log2(q_wc)) * np.log2(num_docs / (1 + doc_freq[q_wid][0]))
                for q_wid, q_wc in q_content.items()} for q_id, q_content in qry.items()}
    doc_new = {d_id : {d_wid : (d_wc) * np.log2(num_docs / (1 + doc_freq[d_wid][0]))
                for d_wid, d_wc in d_content.items()} for d_id, d_content in doc.items()}
    return qry_new, doc_new

def docFreq(doc, vocab_size = 51253):
    corpus_dFreq_total = np.zeros((vocab_size, 2))
    for name, word_list in doc.items():
        temp_word_list = {}
        # assume type of word_list is dictionary
        for word, word_count in word_list.items():
            corpus_dFreq_total[int(word), 0] += 1.0
            corpus_dFreq_total[int(word), 1] += word_count
    return corpus_dFreq_total

def compLenAcc(cmp_dict):
    dicts_len = {}
    for id, cont in cmp_dict.items():
        dict_len = 0.
        for wid, wc in cont.items():
            dict_len += wc
        dicts_len[id] = dict_len
    return dicts_len

def l2Normalize(cmp_np):
    calc_np = np.copy(cmp_np)
    l2_norms = np.sum(cmp_np ** 2, axis=1) ** (1. / 2)
    for idx, l2_norm in enumerate(l2_norms):
        calc_np[idx] = cmp_np[idx] / l2_norm
    
    return calc_np