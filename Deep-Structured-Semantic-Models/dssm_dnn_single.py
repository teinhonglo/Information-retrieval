# Tien-Hong, Lo(teinhonglo@gmail.com)
# paper: Learning Deep Structured Semantic Models for Web Search using Clickthrough Data[1]
# url: https://www.microsoft.com/en-us/research/publication/learning-deep-
# structured-semantic-models-for-web-search-using-clickthrough-data/
# An implementation of the Deep Semantic Similarity Model (DSSM) found in [1].

import numpy as np
np.random.seed(1331)

from keras import backend
from keras.layers import Activation, Input
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Dense, Lambda, Reshape, Dropout
from keras.layers.merge import concatenate, dot
from keras.models import Model
import cPickle as Pickle
from keras import backend as K

def abs_acc(y_true, y_pred):
    return (.5 - K.mean(K.abs(y_pred - y_true), axis=-1)) / .5

def create_model():

    WORD_DEPTH = 51253
    K = 300 # Dimensionality of the projetion layer. See section 3.1.
    L = 128 # Dimensionality of latent semantic space. See section 3.1.
    J = 3 # Number of random unclicked documents serving as negative examples for a query. See section 3.

    # Input tensors holding the query, positive (clicked) document, and negative (unclicked) documents.
    # The first dimension is None because the queries and documents can vary in length.
    query = Input(shape = (WORD_DEPTH,), name="query")
    pos_doc = Input(shape = (WORD_DEPTH,), name="pos_doc")
    neg_docs = [Input(shape = (WORD_DEPTH,), name="neg_doc_" + str(j)) for j in range(J)]

    # Latent Semantic Model
    # projection high dimension to low.
    proj = Dense(K, name="proj_1", activation="tanh")
    proj_2 = Dense(K, name="proj_2", activation="tanh")
    sem = Dense(L, name="sem", activation = "tanh")


    query_proj = proj(query)
    pos_doc_proj = proj(pos_doc)
    neg_doc_projs = [proj(neg_doc) for neg_doc in neg_docs]
    
    query_proj2 = proj_2(query_proj)
    pos_doc_proj2 = proj_2(pos_doc_proj)
    neg_doc_proj2s = [proj_2(neg_doc_proj) for neg_doc_proj in neg_doc_projs]
    
    query_sem = sem(query_proj2)
    pos_doc_sem = sem(pos_doc_proj2)
    neg_doc_sems = [sem(neg_doc_proj2) for neg_doc_proj2 in neg_doc_proj2s]

    # This layer calculates the cosine similarity between the semantic representations of
    # a query and a document.
    R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True, name="pos_cos") # See equation (5).
    R_Q_D_ns = [dot([query_sem, neg_doc_sem], axes = 1, normalize = True, name="neg_cos_" + str(i)) for i, neg_doc_sem in enumerate(neg_doc_sems)] # See equation (5).

    concat_Rs = concatenate([R_Q_D_p] + R_Q_D_ns, name="concat_without_gamma")
    concat_Rs = Reshape((J + 1, 1))(concat_Rs)

    # In this step, we multiply each R(Q, D) value by gamma. In the paper, gamma is
    # described as a smoothing factor for the softmax function, and it's set empirically
    # on a held-out data set.
    weight = np.full((1, 1, 1), 1)

    # We're also going to learn gamma's value by pretending it's a single 1 x 1 kernel.
    with_gamma = Convolution1D(1, 1, padding = "same", input_shape = (J + 1, 1), activation = "linear", use_bias = False, weights = [weight])(concat_Rs) # See equation (5).
    with_gamma = Reshape((J + 1, ))(with_gamma)

    # Finally, we use the softmax function to calculate P(D+|Q).
    prob = Activation("softmax")(with_gamma) # See equation (5).

    # We now have everything we need to define our model.
    model = Model(inputs = [query, pos_doc] + neg_docs, outputs = prob)
    model.compile(optimizer = "adadelta", loss = "categorical_crossentropy", metrics=['accuracy'])
    model.summary()
    # visulization
    '''
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
    '''
    
    return model

if __name__ == "__main__":
   create_model()

