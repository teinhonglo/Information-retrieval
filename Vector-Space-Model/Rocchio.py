#!/usr/bin/env python
import numpy as np
import logging

def feedback(results, qry_mdl, doc_mdl, alpha=0.5, topN = 10, dtype=np.float32):
    logging.debug("Rocchio feedback")
    ''' Initialize '''
    num_qries, num_vocab = qry_mdl.shape
    Rocchio = np.zeros((num_qries, num_vocab), dtype=dtype)
    vocab_size = doc_mdl.shape[1]
    ''' relevance model '''
    for q_idx in range(num_qries):
        Rocchio[q_idx] = np.copy(qry_mdl[q_idx])
        # Relevant top-M documents
        for i, doc_idx in enumerate(results[q_idx][:topN]):
            Rocchio[q_idx] += doc_mdl[doc_idx] / topN
        Rocchio[q_idx] = (1-alpha) * qry_mdl[q_idx] + alpha * Rocchio[q_idx]
    return Rocchio

