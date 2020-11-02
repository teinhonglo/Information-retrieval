#!/usr/bin/env python3
import numpy as np
import numpy.ma as ma
import types

def IDF(doc, weighted_type=0):
    df_vec = docFreq(doc)[:,0]
    num_docs = doc.shape[0]
    # idf
    if weighted_type == 0:
        idf = np.log2((num_docs - df_vec + 0.5) / (df_vec + 0.5))    
    elif weighted_type == 1:
        idf = np.log2(num_docs / df_vec)
    elif weighted_type == 2:
        idf = np.log2(1 + num_docs / df_vec)
    elif weighted_type == 3:
        idf = np.log2(1 + (num_docs - df_vec + 0.5) / (df_vec + 0.5))
    elif weighted_type == 4:
        idf = np.log2(1 + np.max(df_vec) / df_vec)
    elif weighted_type == 5:
        idf = np.log2(1 + (num_docs - df_vec) / df_vec)
    elif weighted_type == 6:
        idf = np.log2(1 + (np.max(df_vec) - df_vec + 0.5) / (df_vec + 0.5))
    elif weighted_type == 7:
        idf = np.log2((np.max(df_vec) - df_vec + 0.5) / (df_vec + 0.5))
        
    return idf

def avgLen(doc):
    num_docs = doc.shape[0]
    doc_len = np.sum(doc, axis = 1)
    avg_len = np.sum(doc_len) / num_docs
    return avg_len

def BM25(qry, doc, idf, avg_len, b=0.75, k1=1.75, k3=1.75, delta=1.25):
    qry_bm25 = np.zeros((qry.shape[0], qry.shape[1]))
    doc_bm25 = np.zeros((doc.shape[0], doc.shape[1]))
    
    for qi, qvec in enumerate(qry):
        zero_idx = np.where(qvec == 0)
        qry_bm25[qi] = idf * (((k3 + 1) * qvec) / (k3 + qvec))
    
    for di, dvec in enumerate(doc):
        doc_len = np.sum(dvec)
        tf_vec = dvec / ((1-b) + b * (doc_len / avg_len))
        zero_idx = np.where(tf_vec == 0)
        doc_bm25[di] = ((k1 + 1) * tf_vec + delta) / (k1 + tf_vec + delta)
        doc_bm25[di][zero_idx] = 0

    return [qry_bm25, doc_bm25]

def docFreq(doc):
    corpus_dFreq_total = np.zeros((doc.shape[1], 2))
    # document frequency
    corpus_dFreq_total[:, 0] = np.count_nonzero(doc, axis=0)
    corpus_dFreq_total[:, 1] = np.sum(doc, axis = 0)
    return corpus_dFreq_total

if __name__ == "__main__":
    qry = np.array([[1, 2, 0], [3, 4, 9]])
    doc = np.array([[5, 0, 0], [7, 8, 1], [1,1,1], [0,5,0]])
    
    print(avgLen(doc))
    

