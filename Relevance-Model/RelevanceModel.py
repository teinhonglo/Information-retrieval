'''
    parameter                type    
    query_docs_ranking         dict        {q_key:[d_key...], ...}
    qry_IDs_list              list        [q_key, ....]
    qry_mdl             numpy        [[query_unigram], ....]
    doc_IDs_list                list        [d_key, .....]
    doc_mdl                numpy        [[doc_unigram]]
    
'''

import numpy as np
import ProcDoc
from math import exp
import logging

def feedback(qry_IDs_list, qry_mdl, doc_IDs_list, doc_mdl, bg_mdl, query_docs_ranking, topM = 9, smoothing = 0.0):
    logging.debug("RM3 feedback")
    ''' Initialize '''
    RM3 = np.zeros((qry_mdl.shape[0], qry_mdl.shape[1]))
    vocabulary_size = doc_mdl.shape[1]
    ''' inverted key '''
    doc_IDs = {doc_ID:int(idx) for idx, doc_ID in enumerate(doc_IDs_list)}
    ''' smoothing '''
    for d_idx, doc_vec in enumerate(doc_mdl):
        doc_mdl[d_idx] = (1 - smoothing) * doc_vec + smoothing * bg_mdl
    ''' relevance model '''
    for q_idx, q_key in enumerate(qry_IDs_list):
        q_vec = qry_mdl[q_idx]
        # Relevant top-M document
        q_t_d = np.zeros(len(query_docs_ranking[q_key][:topM]))
        w_d = np.zeros(vocabulary_size)
        for rank_idx, doc_key in enumerate(query_docs_ranking[q_key][:topM]):
            doc_idx = doc_IDs[doc_key]
            doc_vec = doc_mdl[doc_idx]
            # P(q_t|D)
            q_non_zero, = np.where(q_vec != 0)
            # product
            # q_t_d[rank_idx] = (np.prod(doc_vec[q_non_zero]) + 0.1)
            # logadd
            if q_t_d[rank_idx] == 0.:
                for q_t in np.log(doc_vec[q_non_zero]):
                    q_t_d[rank_idx] += q_t
            #print exp(q_t_d[rank_idx])
            w_d += doc_vec * q_t_d[rank_idx]
        # relevance model
        w_d /= q_t_d.sum(axis = 0)
        RM3[q_idx] = w_d
    return RM3
