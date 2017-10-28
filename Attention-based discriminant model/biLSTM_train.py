#-*- coding: utf-8 -*-
''' Import theano and numpy '''
import numpy as np
import cPickle as Pickle
import biLSTM as TestModel
np.random.seed(1331)
import random
from random import shuffle
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

def preprocess(qry_emb, doc_emb, pointwise_list):
    q = []
    d = []
    r = []
    for idx, data in enumerate(pointwise_list):
        [q_i, d_i, rel] = data
        q.append(qry_emb[q_i])
        d.append(doc_emb[d_i])
        r.append(rel)

    return [q, d, r] 

def generate_arrays(qry, doc, rel, batch_size):
    total_size = len(rel)
    while 1:
        for cur_batch in xrange(total_size / batch_size + 1):
            st = cur_batch * batch_size
            ed = st + batch_size
            if ed >= total_size: ed = None
            # numpy array
            q = np.array(qry[st:ed])
            d = np.array(doc[st:ed])
            r = np.array(rel[st:ed])
            #q = pad_sequences(q, dtype='float32', padding='post')
            #d = pad_sequences(d, dtype='float32', padding='post')
            # zero padding
            x = [q, d]
            y = r
            yield (x, y)

def train(qry_emb, doc_emb, pointwise_list):
    num_of_train_data = len(pointwise_list)
    batch_size = 32 
    epochs = 55
    Epochs_filepath="NN_Model/TDT2/L2R/Epochs_lstm_300/weights-{epoch:02d}-{loss:.2f}_adadelta_learn_post.hdf5"
    checkpoint = ModelCheckpoint(Epochs_filepath, monitor='loss', verbose=0, save_best_only=False, mode='min')
    callbacks_list = [checkpoint]
    [qry, doc, rel] = preprocess(qry_emb, doc_emb, pointwise_list)
    model = TestModel.create_emb_model(2907, 300)
    model.fit_generator(generate_arrays(qry, doc, rel, batch_size), steps_per_epoch=(num_of_train_data / batch_size + 1), callbacks=callbacks_list, epochs = epochs)

    ''' Create a HDF5 file '''                            
    model.save('NN_Model/TDT2/L2R/LSTM_Emb_300_adadelta_learn_post.h5')

def main():
    model_path = "../Corpus/model/TDT2/UM/"
    L2R_path = "../Corpus/rel_irrel/TDT2/"
    qry = []
    pos_doc = []
    neg_doc = []
    rel_dist = []

    with open(L2R_path+"pointwise_list_small.pkl", "rb") as f: qry_doc_rel = Pickle.load(f)
    qry_emb = np.load(model_path+"qry_id_fix_pad.npy")    
    doc_emb = np.load(model_path+"doc_id_fix_pad.npy")
    train(qry_emb, doc_emb, qry_doc_rel)
if __name__ == "__main__":
    main()
