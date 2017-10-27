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
            q = qry[st:ed] 
            d = doc[st:ed]
            r = rel[st:ed]
            # numpy array
            q = np.array(q)
            d = np.array(d)
            r = np.array(r)
            q = pad_sequences(q, dtype='float32')
            d = pad_sequences(d, dtype='float32')
            ''' 
            for data in pointwise_list[st:ed]:
                [q_idx, d_idx, rel] = data
                q.append(qry_emb[q_idx])
                d.append(doc_emb[d_idx])
                r.append(rel)
            # list to numpy
            q = np.array(q)
            d = np.array(d)
            r = np.array(r)
            q = pad_sequences(q, maxlen=2907, dtype='float32')
            d = pad_sequences(q, maxlen=2907, dtype='float32')
            #with open(batch_path+str(cur_batch+1)+".pkl", "rb") as bFile : [q, d, r] = Pickle.load(bFile)
            #data = np.load(batch_path+str(cur_batch+1)+".npy")
            #r = np.load(batch_path+"a"+str(cur_batch+1)+".npy")
            #q = data[0]
            #d = data[1]
            '''
            # zero padding
            x = [q, d]
            y = r
            yield (x, y)

def train(qry_emb, doc_emb, pointwise_list):
    num_of_train_data = len(pointwise_list)
    batch_size = 16 
    epochs = 55
    Epochs_filepath="NN_Model/TDT2/L2R/Epochs_lstm_100/weights-{epoch:02d}-{loss:.2f}_nonconv_adadelta.hdf5"
    checkpoint = ModelCheckpoint(Epochs_filepath, monitor='loss', verbose=0, save_best_only=False, mode='min')
    callbacks_list = [checkpoint]
    [qry, doc, rel] = preprocess(qry_emb, doc_emb, pointwise_list)
    model = TestModel.create_single_model(2907, 100)
    model.fit_generator(generate_arrays(qry, doc, rel, batch_size), steps_per_epoch=(num_of_train_data / batch_size + 1), callbacks=callbacks_list, epochs = epochs)

    ''' Create a HDF5 file '''                            
    model.save('NN_Model/TDT2/L2R/LSTM_Emb_100_nonconv_adadelta.h5')

def main():
    model_path = "../Corpus/model/TDT2/UM/"
    L2R_path = "../Corpus/rel_irrel/TDT2/"
    qry = []
    pos_doc = []
    neg_doc = []
    rel_dist = []

    with open(L2R_path+"pointwise_list_small.pkl", "rb") as f: qry_doc_rel = Pickle.load(f)
    qry_emb = np.load(model_path+"qry_emb_fix_100.npy")    
    doc_emb = np.load(model_path+"doc_emb_fix_100.npy")
    train(qry_emb, doc_emb, qry_doc_rel)
if __name__ == "__main__":
    main()
