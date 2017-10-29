#!/usr/bin/python 
import numpy as np
np.random.seed(5566)
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, Reshape, Activation, Masking
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot

def nn_Test():
    import cPickle as Pickle
    import evaluate
    from collections import defaultdict 
    from keras.preprocessing.sequence import pad_sequences

    model_path = "../Corpus/model/TDT2/UM/"
    emb_path = "NN_Model/TDT2/EMB/Epochs_lstm_300/"
    result_path = "NN_Result/TDT2/EMB/"
    isTrainOnTest = False
    isTDT3 = False

    with open(model_path+"test_query_list.pkl", "rb") as qFile: qry_list = Pickle.load(qFile)
    with open(model_path+"doc_list.pkl", "rb") as dFile: doc_list = Pickle.load(dFile)
    with open(model_path+"doc_model.pkl", "rb") as dFile: doc_model = Pickle.load(dFile)
    qry_emb = np.load(model_path + "tstQry_id_fix_pad.npy")
    
    # create ranking model
    #rank_model = create_single_model(2907, 100)
    rank_model = load_model(emb_path + "weights-51-5.42_ce_learn_post.hdf5")
    qry_emb = rank_model.predict_on_batch(qry_emb)
    with open(result_path+"LSTM_A_KL.pkl", "wb") as rFile: Pickle.dump(qry_emb, rFile, True)
    # evaluate search result

if __name__ == "__main__":
    nn_Test()
    '''
    # how to use it
    batch_size = 17
    maxlen = 100
    samples = 10
    word_rep = 300
    model = create_single_model(maxlen, word_rep)
    qry_train = np.random.rand(samples, maxlen, word_rep) 
    doc_train = np.random.rand(samples, maxlen, word_rep)
    Y_train = np.random.randint(2, size=(samples,))
    print Y_train
    print Y_train.shape
    model.fit([qry_train, doc_train], Y_train, epochs=4, batch_size=batch_size)
    '''

