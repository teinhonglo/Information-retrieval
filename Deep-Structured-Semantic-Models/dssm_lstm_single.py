# Tien-Hong, Lo(teinhonglo@gmail.com)
# paper: Learning Deep Structured Semantic Models for Web Search using Clickthrough Data[1]
# url: https://www.microsoft.com/en-us/research/publication/learning-deep-
# structured-semantic-models-for-web-search-using-clickthrough-data/
# An implementation of the Deep Semantic Similarity Model (DSSM) found in [1].

import numpy as np

from keras import backend
from keras.layers import Activation, Input, LSTM, Masking
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot
from keras.models import Model

def create_model(J=3, WORD_DEPTH=100, K=300, L=128):
    #LETTER_GRAM_SIZE = 3 # See section 3.2.
    #WINDOW_SIZE = 3 # See section 3.2.
    #TOTAL_LETTER_GRAMS = int(3 * 1e4) # Determined from data. See section 3.2.
    #K = 300 # Dimensionality of the max-pooling layer. See section 3.4.
    #L = 128 # Dimensionality of latent semantic space. See section 3.5.
    #J = 3 # Number of random unclicked documents serving as negative examples for a query. See section 4.
    FILTER_LENGTH = 1 # We only consider one time step for convolutions.

    query = Input(shape = (None, WORD_DEPTH))
    pos_doc = Input(shape = (None, WORD_DEPTH))
    neg_docs = [Input(shape = (None, WORD_DEPTH)) for j in range(J)]
    # Input tensors holding the query, positive (clicked) document, and negative (unclicked) documents.
    # The first dimension is None because the queries and documents can vary in length.

    # The document equivalent of the above query model.
    qd_lstm = LSTM(K, input_shape=(None, WORD_DEPTH), name="LSTM")
    qd_output = Dense(K, name="l_output", activation="tanh")
    qd_proj = Dense(K, name="d_proj_1", activation="tanh")
    qd_proj_2 = Dense(K, name="d_proj_2", activation="tanh")
    qd_sem = Dense(L, name="d_sem", activation = "tanh")
        
    query_conv = qd_lstm(Masking(mask_value=.0)(query))
    # Next, we apply a max-pooling layer to the convolved query matrix. 
    query_max = qd_output(query_conv) # See section 3.4.
    query_proj = qd_proj(query_max) # See section 3.4.
    query_proj_2 = qd_proj_2(query_proj) # See section 3.4.
    query_sem = qd_sem(query_proj_2) # See section 3.5.

    pos_doc_conv = qd_lstm(Masking(mask_value=.0)(pos_doc))
    neg_doc_convs = [qd_lstm(Masking(mask_value=.0)(neg_doc)) for neg_doc in neg_docs]        
    
    pos_doc_max = qd_output(pos_doc_conv)
    neg_doc_maxes = [qd_output(neg_doc_conv) for neg_doc_conv in neg_doc_convs]

    pos_doc_proj = qd_proj(pos_doc_max)
    neg_doc_projs = [qd_proj(neg_doc_max) for neg_doc_max in neg_doc_maxes]

    pos_doc_proj2 = qd_proj_2(pos_doc_proj)
    neg_doc_proj2s = [qd_proj_2(neg_doc_proj) for neg_doc_proj in neg_doc_projs]

    pos_doc_sem = qd_sem(pos_doc_proj2)
    neg_doc_sems = [qd_sem(neg_doc_proj2) for neg_doc_proj2 in neg_doc_proj2s]

    # This layer calculates the cosine similarity between the semantic representations of
    # a query and a document.
    R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True, name="pos_cos") # See equation (4).
    R_Q_D_ns = [dot([query_sem, neg_doc_sem], axes = 1, normalize = True, name="neg_cos_" + str(i)) for i, neg_doc_sem in enumerate(neg_doc_sems)] # See equation (4).

    concat_Rs = concatenate([R_Q_D_p] + R_Q_D_ns)
    concat_Rs = Reshape((J + 1, 1))(concat_Rs)

    # In this step, we multiply each R(Q, D) value by gamma. In the paper, gamma is
    # described as a smoothing factor for the softmax function, and it's set empirically
    # on a held-out data set. We're going to learn gamma's value by pretending it's
    # a single 1 x 1 kernel.
    weight = np.full((1, 1, 1), 1)
    with_gamma = Convolution1D(1, 1, padding = "same", input_shape = (J + 1, 1), activation = "linear", use_bias = False, weights = [weight])(concat_Rs)
    with_gamma = Reshape((J + 1, ))(with_gamma)

    # Finally, we use the softmax function to calculate P(D+|Q).
    prob = Activation("softmax")(with_gamma) # See equation (5).

    # We now have everything we need to define our model.
    model = Model(inputs = [query, pos_doc] + neg_docs, outputs = prob)
    model.compile(optimizer = "adadelta", loss = "categorical_crossentropy", metrics=["accuracy"])

    model.summary()
    '''
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
    '''
    return model

if __name__ == "__main__":
    create_model()
