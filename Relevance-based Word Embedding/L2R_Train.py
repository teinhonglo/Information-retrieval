#-*- coding: utf-8 -*-
''' Import theano and numpy '''
import numpy as np
import cPickle as Pickle
import L2R_Model
np.random.seed(1331)
import random
from random import shuffle
from keras.callbacks import ModelCheckpoint

def preprocess(qry_train_set, LM_score, query_list, query_model, doc_list, doc_model):
    random.seed(1331)
    shuffle(qry_train_set)
    qry_vec = []
    d1_vec = []
    d2_vec = []
    answer = []

    for idx, trn in enumerate(qry_train_set):
        qry_idx = query_list.index(trn[0])
        qry_vec.append(query_model[qry_idx])
        [d1_n, d2_n] = trn[1:]
        d1_idx = doc_list.index(d1_n)
        d2_idx = doc_list.index(d2_n)
        d1_vec.append(doc_model[d1_idx])
        d2_vec.append(doc_model[d2_idx])
        answer.append(LM_score[qry_idx][d1_idx] - LM_score[qry_idx][d2_idx])
    return [qry_vec, d1_vec, d2_vec, answer]

def generate_arrays(qry_vec, d1_vec, d2_vec, answer, batch_size):
    total_size = len(qry_vec)
    while 1:
        for i in xrange(total_size / batch_size + 1):
            st = i * batch_size
            ed = st + batch_size
            if ed >= total_size:
                ed = None
            q = np.vstack(qry_vec[st:ed])
            d1 = np.vstack(d1_vec[st:ed])
            d2 = np.vstack(d2_vec[st:ed])
            a = np.vstack(answer[st:ed])
            x = [q, d1, d2]
            y = np.array(a)
            yield (x, y)

def train(qry, pos_doc, neg_doc, rel_dist):
    steps_per_epoch = 180018
    batch_size = 16 
    epochs = 55
    Epochs_filepath="NN_Model/TDT2/L2R/Epochs/weights-{epoch:02d}-{loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(Epochs_filepath, monitor='loss', verbose=0, save_best_only=False, mode='min')
    callbacks_list = [checkpoint]
    model = L2R_Model.create_model()
    model.fit_generator(generate_arrays(qry, pos_doc, neg_doc, rel_dist, batch_size),
                    steps_per_epoch=(steps_per_epoch / batch_size + 1), 
                    callbacks=callbacks_list,
                    epochs = epochs)

    ''' Create a HDF5 file '''                            
    model.save('NN_Model/TDT2/L2R/RLE_L2R_KL.h5')

def main():
    model_path = "../Corpus/model/TDT2/UM/"
    pairwise_path = "../Corpus/rel_irrel/TDT2/"
    qry = []
    pos_doc = []
    neg_doc = []
    rel_dist = []

    with open(pairwise_path+"L2R_qry_train_set.pkl", "rb") as f: qry_train_set = Pickle.load(f)
    with open(model_path+"LM_score.pkl", "rb") as f: LM_score = Pickle.load(f)
    with open(model_path+"query_model.pkl", "rb") as f: query_model = Pickle.load(f)
    with open(model_path+"query_list.pkl", "rb") as f: query_list = Pickle.load(f)
    with open(model_path+"doc_model.pkl", "rb") as f: doc_model = Pickle.load(f)
    with open(model_path+"doc_list.pkl", "rb") as f: doc_list = Pickle.load(f)
    [qry, pos_doc, neg_doc, rel_dist] =  preprocess(qry_train_set, LM_score, query_list, query_model, doc_list, doc_model)
    train(qry, pos_doc, neg_doc, rel_dist)
if __name__ == "__main__":
    main()
