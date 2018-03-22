#!/usr/bin/env python3
import numpy as np
import types

def TFIDF(qry, doc, doc_len):
    doc_freq = docFreq(doc)
    num_docs = len(list(doc.keys())) + 1
    qry_new = {q_id : {q_wid : (.5 + .5 * np.log2(q_wc)) * np.log2(num_docs / (1 + doc_freq[q_wid][0]))
                for q_wid, q_wc in q_content.items()} for q_id, q_content in qry.items()}
    doc_new = {d_id : {d_wid : (1 + np.log2(d_wc)) * np.log2(num_docs / (1 + doc_freq[d_wid][0])) / doc_len[d_id] 
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

def docLen(doc):
    docs_len = {}
    for d_id, d_cont in doc.items():
        doc_len = 0
        for d_wid, d_wc in d_cont.items():
            doc_len += d_wc ** 2
        docs_len[d_id] = np.sqrt(doc_len)
    return docs_len    
