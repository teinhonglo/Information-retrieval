#!/usr/bin/python 
import numpy as np
np.random.seed(5566)
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
    word_emb = Embedding(input_dim=vocabulary_size, output_dim=word_rep, input_length=maxlen, mask_zero=True)
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

    model = Model(inputs = qry, outputs = predict)
    model.compile(optimizer = "Adam", loss = "kullback_leibler_divergence", metrics=['accuracy'])
    model.summary()
    
    return model

def create_emb_model(maxlen, word_rep =100):
    #maxlen = None
    num_u = word_rep
    vocabulary_size = 51253
    reduction_d_a = 64
    reduction_r = 16
    # Embedding network
    word_emb = Embedding(input_dim=vocabulary_size, output_dim=word_rep, input_length=maxlen, mask_zero=True)
    biLSTM_H = Bidirectional(LSTM(num_u, return_sequences=True, name="LSTM"), merge_mode='concat', name = "Bidirectional_LSTM")
    # multi-layer perceptron, using attention
    mlp_hid_1 = Dense(reduction_d_a, activation = "tanh", name="mlp_tanh")
    mlp_hid_2 = Dense(reduction_r, activation="softmax", name="mlp_softmax")
    # learning to rank architecture
    Conv1D_Feature = Convolution1D(1, 2 * num_u, padding = "same", input_shape = (2 * num_u, reduction_r), activation = "linear", use_bias = False, name="position_aware")

    qry = Input(shape=(maxlen,), name="qry_input")
    doc = Input(shape=(maxlen,), name="doc_input")
    #qry_rep = Masking(mask_value=.0)(qry)
    #doc_rep = Masking(mask_value=.0)(doc)
    qry_rep = word_emb(qry)
    doc_rep = word_emb(doc)
    # query feature map
    qry_H = biLSTM_H(qry_rep)
    q_h1 = mlp_hid_1(qry_H)
    q_A = mlp_hid_2(q_h1)
    M = dot([qry_H, q_A], axes = 1, normalize="False", name="with_Attention_qry")
    M = Reshape((2 * num_u, reduction_r))(M)
    #M = Reshape((2 * num_u, ))(M)
    # doc feature map
    doc_H = biLSTM_H(doc_rep)
    d_h1 = mlp_hid_1(doc_H)
    d_A = mlp_hid_2(d_h1)
    M_d = dot([doc_H, d_A], axes = 1, normalize="False", name="with_Attention_doc")
    M_d = Reshape((2 * num_u, reduction_r))(M_d)
    #M_d = Reshape((2 * num_u, ))(M_d)
    # ranking task
    conv_qry = Conv1D_Feature(M)
    conv_qry = Reshape((2 * num_u,))(conv_qry)
    ref_qry = Activation("tanh")(conv_qry)

    conv_doc = Conv1D_Feature(M_d)
    conv_doc = Reshape((2 * num_u,))(conv_doc)
    ref_doc = Activation("tanh")(conv_doc)
    # similarity
    similarity = dot([ref_qry, ref_doc], axes=1, normalize="True", name="cosine_similarity")
    #similarity = dot([M, M_d], axes=1, normalize="True", name="cosine_similarity")
    predict = Activation("sigmoid", name="predict_layer")(similarity)

    model = Model(inputs = [qry, doc], outputs = predict)
    model.compile(optimizer = "adadelta", loss = "binary_crossentropy", metrics=['accuracy'])
    model.summary()
    
    return model

def create_single_model(maxlen, word_rep =100):
    #maxlen = None
    num_u = word_rep
    vocabulary_size = 51253
    reduction_d_a = 64
    reduction_r = 16
    # Embedding network
    #word_emb = Embedding(input_dim=vocabulary_size, output_dim=word_rep, input_length=maxlen, mask_zero=True)
    biLSTM_H = Bidirectional(LSTM(num_u, return_sequences=True, name="LSTM"), merge_mode='concat', name = "Bidirectional_LSTM")
    # multi-layer perceptron, using attention
    mlp_hid_1 = Dense(reduction_d_a, activation = "tanh", name="mlp_tanh")
    mlp_hid_2 = Dense(reduction_r, activation="softmax", name="mlp_softmax")
    # learning to rank architecture
    Conv1D_Feature = Convolution1D(1, 2 * num_u, padding = "same", input_shape = (2 * num_u, reduction_r), activation = "linear", use_bias = False, name="position_aware")

    qry = Input(shape=(maxlen, word_rep), name="qry_input")
    doc = Input(shape=(maxlen, word_rep), name="doc_input")
    qry_rep = Masking(mask_value=.0)(qry)
    doc_rep = Masking(mask_value=.0)(doc)
    #qry_rep = word_emb(qry)
    #doc_rep = word_emb(doc)
    # query feature map
    qry_H = biLSTM_H(qry_rep)
    q_h1 = mlp_hid_1(qry_H)
    q_A = mlp_hid_2(q_h1)
    M = dot([qry_H, q_A], axes = 1, normalize="False", name="with_Attention_qry")
    M = Reshape((2 * num_u, reduction_r))(M)
    #M = Reshape((2 * num_u, ))(M)
    # doc feature map
    doc_H = biLSTM_H(doc_rep)
    d_h1 = mlp_hid_1(doc_H)
    d_A = mlp_hid_2(d_h1)
    M_d = dot([doc_H, d_A], axes = 1, normalize="False", name="with_Attention_doc")
    M_d = Reshape((2 * num_u, reduction_r))(M_d)
    #M_d = Reshape((2 * num_u, ))(M_d)
    # ranking task
    conv_qry = Conv1D_Feature(M)
    conv_qry = Reshape((2 * num_u,))(conv_qry)
    ref_qry = Activation("tanh")(conv_qry)

    conv_doc = Conv1D_Feature(M_d)
    conv_doc = Reshape((2 * num_u,))(conv_doc)
    ref_doc = Activation("tanh")(conv_doc)
    # similarity
    similarity = dot([ref_qry, ref_doc], axes=1, normalize="True", name="cosine_similarity")
    #similarity = dot([M, M_d], axes=1, normalize="True", name="cosine_similarity")
    predict = Activation("sigmoid", name="predict_layer")(similarity)

    model = Model(inputs = [qry, doc], outputs = predict)
    model.compile(optimizer = "adadelta", loss = "binary_crossentropy", metrics=['accuracy'])
    model.summary()
    '''
    # visualization
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
    '''
    return model

def create_double_model(maxlen, word_rep):
    maxlen = None
    num_u = 96 
    reduction_d_a = 64
    reduction_r = 16
    # Embedding network
    biLSTM_H = Bidirectional(LSTM(num_u, return_sequences=True, name="LSTM"), merge_mode='concat', name = "qry_Bidirectional_LSTM")
    # multi-layer perceptron, using attention
    mlp_hid_1 = Dense(reduction_d_a, activation = "tanh", name="qry_mlp_tanh")
    mlp_hid_2 = Dense(reduction_r, activation="softmax", name="qry_mlp_softmax")
    # learning to rank architecture
    Conv1D_Feature = Convolution1D(1, 2 * num_u, padding = "same", input_shape = (2 * num_u, reduction_r), activation = "linear", use_bias = False, name="qry_position_aware")
        
    # Embedding network
    biLSTM_H_d = Bidirectional(LSTM(num_u, return_sequences=True, name="LSTM"), merge_mode='concat', name = "doc_Bidirectional_LSTM")
    # multi-layer perceptron, using in attention
    mlp_hid_1_d = Dense(reduction_d_a, activation = "tanh", name="doc_mlp_tanh")
    mlp_hid_2_d = Dense(reduction_r, activation="softmax", name="doc_mlp_softmax")
    # learning to rank architecture
    Conv1D_Feature_d = Convolution1D(1, 2 * num_u, padding = "same", input_shape = (2 * num_u, reduction_r), activation = "linear", use_bias = False, name="doc_position_aware")

    qry = Input(shape=(maxlen, word_rep), name="qry_input")
    doc = Input(shape=(maxlen, word_rep), name="doc_input")
    qry_Mask = Masking(mask_value=.0)(qry)
    doc_Mask = Masking(mask_value=.0)(doc)
    # query feature map
    qry_H = biLSTM_H(qry_Mask)
    q_h1 = mlp_hid_1(qry_H)
    q_A = mlp_hid_2(q_h1)
    M = dot([qry_H, q_A], axes = 1, normalize="False", name="with_Attention_qry")
    M = Reshape((2 * num_u, reduction_r))(M)
    # doc feature map
    doc_H = biLSTM_H_d(doc_Mask)
    d_h1 = mlp_hid_1_d(doc_H)
    d_A = mlp_hid_2_d(d_h1)
    M_d = dot([doc_H, d_A], axes = 1, normalize="False", name="with_Attention_doc")
    M_d = Reshape((2 * num_u, reduction_r))(M_d)
    # ranking task
    conv_qry = Conv1D_Feature(M)
    conv_qry = Reshape((2 * num_u,))(conv_qry)
    ref_qry = Activation("tanh")(conv_qry)

    conv_doc = Conv1D_Feature_d(M_d)
    conv_doc = Reshape((2 * num_u,))(conv_doc)
    ref_doc = Activation("tanh")(conv_doc)
    # similarity
    similarity = dot([ref_qry, ref_doc], axes=1, normalize="True", name="cosine_similarity")
    predict = Activation("sigmoid", name="predict_layer")(similarity)

    model = Model(inputs = [qry, doc], outputs = predict)
    model.compile(optimizer = "adadelta", loss = "binary_crossentropy", metrics=['accuracy'])
    model.summary()
    '''
    # visualization
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
    '''
    return model

def nn_Test():
    import cPickle as Pickle
    import evaluate
    from collections import defaultdict 
    from keras.preprocessing.sequence import pad_sequences

    model_path = "../Corpus/model/TDT2/UM/"
    l2r_path = "NN_Model/TDT2/L2R/Epochs_lstm_100/"
    isTrainOnTest = False
    isTDT3 = False

    with open(model_path+"test_query_list.pkl", "rb") as qFile: qry_list = Pickle.load(qFile)
    with open(model_path+"doc_list.pkl", "rb") as dFile: doc_list = Pickle.load(dFile)
    qry_emb = np.load(model_path + "tstQry_emb_fix_100.npy")
    doc_emb = np.load(model_path + "doc_emb_fix_100.npy")
    # create ranking model
    #rank_model = create_single_model(2907, 100)
    rank_model = load_model(l2r_path + "weights-01-0.39_nonconv_adadelta.hdf5")
    # evaluate search result
    doc_length = doc_emb.shape[0] - 1
    print doc_length
    
    eva = evaluate.evaluate_model(isTrainOnTest, isTDT3)
    
    query_docs_ranking = defaultdict(list)
    for qry_idx, qry_vec in enumerate(qry_emb):
        query_ranking = []
        qry_test = []
        doc_test = []
        results = []
        # prepare data
        for doc_idx, doc_vec in enumerate(doc_emb):
            qry_test.append(qry_vec)
            doc_test.append(doc_vec)
            if (len(qry_test) % 16 == 0) or (doc_idx == doc_length):
                qry_test = pad_sequences(np.array(qry_test), dtype='float32', padding='post')
                doc_test = pad_sequences(np.array(doc_test), dtype='float32', padding='post')
                print qry_test.shape
                print doc_test.shape
                # predict relevance result
                batch_results = rank_model.predict_on_batch([qry_test, doc_test])
                #print batch_results
                results += batch_results.reshape(-1).tolist()
                #print batch_results.reshape(-1).tolist()
                #print results
                print qry_idx, len(results)
                qry_test = []
                doc_test = []
        #print results
        results = np.argsort(-np.array(results))
        print results
        for doc_idx in results:
            #print doc_idx
            query_docs_ranking[qry_list[qry_idx]].append(doc_list[doc_idx])
    
    #with open("test_query_ranking.pkl", "wb") as tmpRkF: Pickle.dump(query_docs_ranking, tmpRkF, True)
    with open("test_query_ranking.pkl", "rb") as tmpRkF: query_docs_ranking = Pickle.load(tmpRkF)
    # evaluate 
    mAP = eva.mean_average_precision(query_docs_ranking)
    print mAP

if __name__ == "__main__":
    create_rep_model(2907, 100)
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

