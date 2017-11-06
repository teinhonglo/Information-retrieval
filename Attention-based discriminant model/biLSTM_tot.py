#-*- coding: utf-8 -*-
''' Import theano and numpy '''
import numpy as np
import cPickle as Pickle
np.random.seed(1331)
import theano
#import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
#                  inter_op_parallelism_threads = 1, intra_op_parallelism_threads = 1))
import biLSTM_flatten as TestModel
import random
from random import shuffle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


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
            q = np.asarray(qry[st:ed])
            d = np.asarray(doc[st:ed])
            r = np.asarray(rel[st:ed])
            #q = pad_sequences(q, dtype='float32', padding='post')
            #d = pad_sequences(d, dtype='float32', padding='post')
            # zero padding
            x = [q, d]
            y = r
            yield (x, y)

def predict(qry_emb, doc_emb, pointwise_list):
     num_of_train_data = len(pointwise_list)
     batch_size = 32 
     epochs = 55
     l2r_path = "NN_Model/TDT2/L2R/Epochs_lstm_100/"
     [qry, doc, rel] = preprocess(qry_emb, doc_emb, pointwise_list)
     model = load_model(l2r_path + "weights-09-0.37_Emb_adadelta_learn_100_post.hdf5")
     results = model.predict_generator(generate_arrays(qry, doc, rel, batch_size), steps=(num_of_train_data / batch_size + 1), verbose=1)
     print(results.shape)
     print(results)
     np.save("tot.batch.npy", results)
     ''' Create a HDF5 file '''                            
     model.save('NN_Model/TDT2/L2R/LSTM_Flat_Emb_adadelta_learn_100_post.h5')

def train_obj(qry_emb, qry_rel): 
    pass
    '''
    batch_size = 32 
    epochs = 55
    #Epochs_filepath="NN_Model/TDT2/EMB/Epochs_lstm_300/weights-{epoch:02d}-{loss:.2f}_ce_local_learn_post.hdf5"
    qry_emb = np.asarray(qry_emb)
    checkpoint = ModelCheckpoint(Epochs_filepath, monitor='loss', verbose=0, save_best_only=False, mode='min')
    callbacks_list = [checkpoint]
    # doc_length = 2907
    # query length = 1794
    model = TestModel.create_rep_model(1794, 300)
    history_adam = model.fit(qry_emb, qry_rel,
    		batch_size=batch_size,
    		epochs=epochs,
    		verbose=1,
    		shuffle=True,
    		validation_split=0.1,
    		callbacks = callbacks_list)
    '''
    ''' Create a HDF5 file '''
    #model.save('NN_Model/TDT2/EMB/LSTM_Emb_300_ce_local_learn_post.h5')

def main():
    model_path = "../Corpus/model/TDT2/UM/"
    obj_path = "obj_func/TDT2/"
    L2R_path = "../Corpus/rel_irrel/TDT2/"
    qry = []
    pos_doc = []
    neg_doc = []
    rel_dist = []
    
    with open(L2R_path+"pointwise_list_small.pkl", "rb") as f: qry_doc_rel = Pickle.load(f)
    qry_emb = np.load(model_path+"qry_id_fix_pad.npy")    
    #with open(obj_path + "rel_supervised_swlm_entropy.pkl", "rb") as rFile: qry_rel = Pickle.load(rFile)
    doc_emb = np.load(model_path+"doc_id_fix_pad.npy")
    #predict(qry_emb, doc_emb, qry_doc_rel)
    results = np.load("tot.batch.npy")
    pos_val = 0
    neg_val = 0
    pos_n = 0
    neg_n = 0
    for idx, data in enumerate(qry_doc_rel):
        [qry_id, doc_id, rel] = data
        pred_rel = results[idx]
        if rel == 1 and pred_rel < 0.5:
            print rel, pred_rel
        if rel == 0 and pred_rel >= 0.5:
            print rel, pred_rel
        if pred_rel >= 0.5:
            pos_val += pred_rel
            pos_n += 1
        else:
            neg_val += pred_rel
            neg_n += 1
    print pos_val / pos_n
    print neg_val / neg_n
    #train_obj(qry_emb, qry_rel)

if __name__ == "__main__":
    main()
