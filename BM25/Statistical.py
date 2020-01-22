#!/usr/bin/env python3
import numpy as np
import numpy.ma as ma
import types

def IDF(doc):
    # inverse document frequency
    doc_freq = docFreq(doc)
    num_docs = doc.shape[0]
    idf = np.zeros(doc.shape[1])
    idf = np.log2((num_docs - doc_freq[:,0] + 0.5) / (doc_freq[:,0] + 0.5))
    return idf

def docFreq(doc):
    vocab_size = doc.shape[1]
    corpus_dFreq_total = np.zeros((vocab_size, 2))
    # document frequency
    corpus_dFreq_total[:, 0] = np.count_nonzero(doc, axis=0)
    corpus_dFreq_total[:, 1] = np.sum(doc, axis = 0)
    return corpus_dFreq_total

def compLenAve(doc):
    num_docs = doc.shape[0]
    doc_len = np.sum(doc, axis = 1)
    doc_len_ave = 1.0 * np.sum(doc_len, axis = 0) / num_docs
    doc_len = doc_len / doc_len_ave
    return doc_len

if __name__ == "__main__":
    qry = np.array([[1, 2, 0], [3, 4, 9]])
    doc = np.array([[5, 0, 0], [7, 8, 1], [1, 3, 0]])
    compLenAve(doc)
    

