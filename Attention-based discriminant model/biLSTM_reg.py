#!/usr/bin/python 
import numpy as np
np.random.seed(5566)
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, Reshape, Activation, Masking, Lambda
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot

def penalization_l2(attention):
    H = dot([attention, attention], axes=1, normalize=False) - K.eye(16)
    # activity_regularizer = penalization_l2
    return 0.1 * K.sum(K.square(attention))

def p_shape(attention):
    return (attention[0], attention[1], 1)

def create_emb_model(maxlen, word_rep =100):
    #maxlen = None
    num_u = word_rep
    vocabulary_size = 51253
    reduction_d_a = 64
    reduction_r = 16
    # Embedding network
    word_emb = Embedding(input_dim=vocabulary_size + 1, output_dim=word_rep, input_length=maxlen, mask_zero=True)
    biLSTM_H = Bidirectional(LSTM(num_u, return_sequences=True, name="LSTM"), merge_mode='concat', name = "Bidirectional_LSTM")
    # multi-layer perceptron, using attention
    mlp_hid_1 = Dense(reduction_d_a, activation = "tanh", name="mlp_tanh")
    mlp_hid_2 = Dense(reduction_r, activation="softmax", name="mlp_softmax")
    penal_layer = Lambda(penalization_l2, output_shape=p_shape, name="penal_layer")
    # learning to rank architecture
    Conv1D_Feature = Convolution1D(1, 2 * num_u, padding = "same", input_shape = (2 * num_u, reduction_r), activation = "linear", use_bias = False, name="position_aware")

    qry = Input(shape=(maxlen,), name="qry_input")
    doc = Input(shape=(maxlen,), name="doc_input")
    #qry_rep = Masking(mask_value=.0)(qry)
    #doc_rep = Masking(mask_value=.0)(doc)
    qry_rep = Masking(mask_value=.0)(word_emb(qry))
    doc_rep = Masking(mask_value=.0)(word_emb(doc))
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

if __name__ == "__main__":
    # how to use it
    create_emb_model(2907) 
    '''
    batch_size = 17
    maxlen = 100
    samples = 10
    word_rep = 300
    model = create_emb_model(maxlen, word_rep)
    qry_train = np.random.randint(2, size=(samples, maxlen)) 
    doc_train = np.random.randint(2, size=(samples, maxlen))
    Y_train = np.random.randint(2, size=(samples,))
    print Y_train
    print Y_train.shape
    model.fit([qry_train, doc_train], Y_train, epochs=4, batch_size=batch_size)
    '''

