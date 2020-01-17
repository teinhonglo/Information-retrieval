'''
    parameter                type    
    query_docs_ranking         dict        {q_key:[d_key...], ...}
    query_list              list        [q_key, ....]
    query_model             numpy        [[query_unigram], ....]
    doc_list                list        [d_key, .....]
    doc_model                numpy        [[doc_unigram]]
    
'''

import numpy as np
import ProcDoc
from math import exp

def feedback(query_list, query_model, doc_list, doc_model, background_model, query_docs_ranking, topM = 9, smoothing = 0.0):
    ''' inverted key '''
    doc_IDs = {doc_ID:int(idx) for idx, doc_ID in enumerate(doc_list)}
    ''' smoothing '''
    vocabulary_size = doc_model.shape[1]
    for d_idx, doc_vec in enumerate(doc_model):
        doc_model[d_idx] = (1 - smoothing) * doc_vec + smoothing * background_model

    ''' relevance model '''
    for q_idx, q_key in enumerate(query_list):
        q_vec = query_model[q_idx]
        # Relevant top-M document
        q_t_d = np.zeros(len(query_docs_ranking[q_key][:topM]))
        w_d = np.zeros(vocabulary_size)
        for rank_idx, doc_key in enumerate(query_docs_ranking[q_key][:topM]):
            doc_idx = doc_IDs[doc_key]
            doc_vec = doc_model[doc_idx]
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
        query_model[q_idx] = w_d
    return query_model 
