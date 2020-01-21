#!/usr/bin/env python3
import numpy as np
import numpy.ma as ma
import types

def TFIDF(qry, doc):
    doc_freq = docFreq(doc)
    num_docs = doc.shape[0] + 1
    #qry_new = {q_id : {q_wid : (.5 + .5 * np.log2(q_wc)) * np.log2(num_docs / (1 + doc_freq[q_wid][0]))
    #            for q_wid, q_wc in q_content.items()} for q_id, q_content in qry.items()}
    #doc_new = {d_id : {d_wid : (d_wc) * np.log2(num_docs / (1 + doc_freq[d_wid][0]))
    #            for d_wid, d_wc in d_content.items()} for d_id, d_content in doc.items()}
    qry_tfidf = np.zeros((qry.shape[0], qry.shape[1]))
    doc_tfidf = np.zeros((doc.shape[0], doc.shape[1]))
    for qi, qvec in enumerate(qry):
        zero_idx = np.where(qry[qi] == 0)
        qry_tfidf[qi] = (0.5 + 0.5 * ma.log2(qvec).filled(0)) * np.log2(1 + num_docs / (1 + doc_freq[:,0]))
        qry_tfidf[qi][zero_idx] = 0
    
    for di, dvec in enumerate(doc):
        zero_idx = np.where(doc[di] == 0)
        doc_tfidf[di] = dvec * np.log2(1 + num_docs / (1 + doc_freq[:,0]))
        doc_tfidf[di][zero_idx] = 0

    return [qry_tfidf, doc_tfidf]

def docFreq(doc, vocab_size = 51253):
    corpus_dFreq_total = np.zeros((doc.shape[1], 2))
    # document frequency
    corpus_dFreq_total[:, 0] = np.count_nonzero(doc, axis=0)
    corpus_dFreq_total[:, 1] = np.sum(doc, axis = 0)
    return corpus_dFreq_total

def l2Norm(cmp_np):
    calc_np = np.copy(cmp_np)
    l2_norms = np.sum(cmp_np ** 2, axis=1) ** (1. / 2)
    for idx, l2_norm in enumerate(l2_norms):
        if l2_norm != 0:
            calc_np[idx] = cmp_np[idx] / l2_norm
        else:
            print(idx)
    return calc_np

def compLenAcc(comp_dict):
    dicts_len = {}
    for id, cont in comp_dict.items():
        dict_len = 0.
        for wid, wc in cont.items():
            dict_len += wc
        dicts_len[id] = dict_len
    return dicts_len

if __name__ == "__main__":
    qry = np.array([[1, 2, 0], [3, 4, 9]])
    doc = np.array([[5, 0, 0], [7, 8, 1]])
    TFIDF(qry, doc)
    

