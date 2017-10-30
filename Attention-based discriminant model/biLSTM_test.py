#!/usr/bin/python 
import numpy as np
np.random.seed(0)
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, Reshape, Activation, Masking
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot

def create_rep_model(maxlen, word_rep =100):
    #maxlen = None
    num_u = word_rep
    vocabulary_size = 51253
    reduction_d_a = 64
    reduction_r = 16
    embedding_size = 300
    # Embedding network
    word_emb = Embedding(input_dim=vocabulary_size + 1, output_dim=word_rep, input_length=maxlen, mask_zero=False)
    biLSTM_H = Bidirectional(LSTM(num_u, return_sequences=True, name="LSTM"), merge_mode='concat', name = "Bidirectional_LSTM")
    # multi-layer perceptron, using attention
    mlp_hid_1 = Dense(reduction_d_a, activation = "tanh", name="mlp_tanh")
    mlp_hid_2 = Dense(reduction_r, activation="softmax", name="mlp_softmax")
    # learning to rank architecture
    Conv1D_Feature = Convolution1D(1, 2 * num_u, padding = "same", input_shape = (2 * num_u, reduction_r), activation = "linear", use_bias = False, name="position_aware")
    encoded_hid = Dense(embedding_size, activation="linear", name="encoded_hid")
    rep_layer = Dense(vocabulary_size, activation="softmax", name="representaion_layer")

    qry = Input(shape=(maxlen,), name="qry_input")
    #qry_rep = Masking(mask_value=.0)(qry)
    #doc_rep = Masking(mask_value=.0)(doc)
    qry_rep = word_emb(qry)
    # query feature map
    qry_H = biLSTM_H(qry_rep)
    q_h1 = mlp_hid_1(qry_H)
    q_A = mlp_hid_2(q_h1)
    M = dot([qry_H, q_A], axes = 1, normalize="False", name="with_Attention_qry")
    M = Reshape((2 * num_u, reduction_r))(M)
    #M = Reshape((2 * num_u, ))(M)
    # doc feature map
    #M_d = Reshape((2 * num_u, ))(M_d)
    # ranking task
    conv_qry = Conv1D_Feature(M)
    conv_qry = Reshape((2 * num_u,))(conv_qry)
    encoded_qry = encoded_hid(conv_qry)

    #similarity = dot([M, M_d], axes=1, normalize="True", name="cosine_similarity")
    predict = rep_layer(encoded_qry)

    model = Model(inputs = qry, outputs = M)
    model.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics=['accuracy', 'categorical_accuracy'])
    model.summary()
    
    return model

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

    qry_emb = np.load(model_path + "qry_id_fix_pad.npy")
    print qry_emb.shape
    len_qry = qry_emb.shape[0]
    qry_emb_reshape = np.zeros((qry_emb.shape[0], 51253))
    # create ranking model
    # rank_model = create_single_model(2907, 100)
    rank_model = create_rep_model(2907, 300)
    rank_model.load_weights(emb_path + "weights-05-5.93_ce_learn_post.hdf5", by_name=True)
    #rank_model = load_model(emb_path + "weights-05-5.93_ce_learn_post.hdf5")
    batch_size = 16
    for cur_batch in xrange(len_qry / batch_size):
        st = cur_batch * batch_size
        ed = st + batch_size
        if ed > len_qry: ed = None
        print st, ed
        #qry_emb_reshape[:2] = 
        print np.nonzero(rank_model.predict_on_batch(qry_emb[:2])[0] == 0)
        raw_input()
    print qry_emb_reshape
    with open(result_path+"LSTM_A_KL.pkl", "wb") as rFile: Pickle.dump(qry_emb_reshape, rFile, True)
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

