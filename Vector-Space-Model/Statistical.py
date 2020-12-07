#!/usr/bin/env python3
import numpy as np
import numpy.ma as ma
import types

def TFIDFPairs(tf_vec, df_vec, num_docs, weighted_type=[0, 0]):
    # tf
    #eps = 0
    eps = np.finfo(np.float32).eps
    tf = np.copy(tf_vec)
    zero_idx = np.where(tf == 0)
    
    if weighted_type[0] == 0:
        tf = 1 * (tf_vec > 0)
    elif weighted_type[0] == 1:
        tf = tf_vec.copy()
    elif weighted_type[0] == 2:
        tf = 1 + tf_vec
    elif weighted_type[0] == 3:
        tf = np.log2(1 + tf_vec)
    elif weighted_type[0] == 4:
        if np.max(tf_vec) == 0:
            tf = 0.5 + 0.5 * tf_vec
        else:
            tf = 0.5 + 0.5 * tf_vec / np.max(tf_vec)
    elif weighted_type[0] == 5:
        tf = 1 + ma.log2(tf_vec).filled(0)
    elif weighted_type[0] == 6:
        tf = 1 + np.log2(1 + tf_vec)
    tf[zero_idx] = eps
    
    # idf
    if weighted_type[1] == 0:
        idf = 1
    elif weighted_type[1] == 1:
        idf = np.log2(num_docs / df_vec)
    elif weighted_type[1] == 2:
        idf = np.log2(1 + num_docs / df_vec)
    elif weighted_type[1] == 3:
        idf = np.log2(1 + (num_docs - df_vec + 0.5) / (df_vec + 0.5))
    elif weighted_type[1] == 4:
        idf = np.log2((num_docs - df_vec + 0.5) / (df_vec + 0.5))
    elif weighted_type[1] == 5:
        idf = np.log2(1 + np.max(df_vec) / df_vec)
    elif weighted_type[1] == 6:
        idf = np.log2(1 + (num_docs - df_vec) / df_vec)
    elif weighted_type[1] == 7:
        idf = np.log2(1 + (np.max(df_vec) - df_vec + 0.5) / (df_vec + 0.5))
    elif weighted_type[1] == 8:
        idf = np.log2((np.max(df_vec) - df_vec + 0.5) / (df_vec + 0.5))
        
    return [tf, idf]

def TFIDF(qry, doc, weighted_type={"qry":[0, 0], "doc":[0, 0]}, dtype=np.float32):
    doc_freq = docFreq(doc)
    num_docs = doc.shape[0]
    
    qry_tfidf = np.zeros((qry.shape[0], qry.shape[1]), dtype=dtype)
    doc_tfidf = np.zeros((doc.shape[0], doc.shape[1]), dtype=dtype)
    for qi, qvec in enumerate(qry):
        [tf, idf] = TFIDFPairs(qvec, doc_freq[:,0], num_docs, weighted_type["qry"])
        qry_tfidf[qi] = tf * idf
    
    for di, dvec in enumerate(doc):
        [tf, idf] = TFIDFPairs(dvec, doc_freq[:,0], num_docs, weighted_type["doc"]) 
        doc_tfidf[di] = tf * idf

    return [qry_tfidf, doc_tfidf]

def docFreq(doc, dtype=np.float32):
    corpus_dFreq_total = np.zeros((doc.shape[1], 2), dtype=dtype)
    # document frequency
    corpus_dFreq_total[:, 0] = np.count_nonzero(doc, axis=0)
    corpus_dFreq_total[:, 1] = np.sum(doc, axis = 0)
    return corpus_dFreq_total

def l2Norm(cmp_np):
    calc_np = np.copy(cmp_np)
    l2_norms = np.sqrt(np.sum(cmp_np ** 2, axis=1))
    for idx, l2_norm in enumerate(l2_norms):
        if l2_norm != 0:
            calc_np[idx] = cmp_np[idx] / l2_norm
        else:
            calc_np[idx] = 0
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
    qry = np.array([[1, 2, 0], [3, 4, 9], [0,0,0]])
    doc = np.array([[5, 0, 0], [7, 8, 1], [0,0,0]])
    qry_pairs = []
    doc_pairs = []
    num_tfs = 7
    num_idfs = 6
    for tf in range(num_tfs):
        for idf in range(num_idfs):
            qry_pairs.append([tf, idf])
            doc_pairs.append([tf, idf])
    for q_tf, q_idf in qry_pairs:
        for d_tf, d_idf in doc_pairs:
            [qry_tfidf, doc_tfidf] = TFIDF(qry, doc, {"qry":[q_tf, q_idf], "doc": [d_tf, d_idf]})
            qry_isinf = 1 * np.isinf(qry_tfidf)
            doc_isinf = 1 * np.isinf(doc_tfidf)
            if (np.sum(qry_isinf) != 0) or (np.sum(doc_isinf) != 0):
                print(qry_tfidf, doc_isinf)
                print(q_tf, q_idf, d_tf, d_idf)
    

