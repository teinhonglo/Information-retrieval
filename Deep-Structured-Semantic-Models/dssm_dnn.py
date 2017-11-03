# Tien-Hong, Lo(teinhonglo@gmail.com)
# paper: Learning Deep Structured Semantic Models for Web Search using Clickthrough Data[1]
# url: https://www.microsoft.com/en-us/research/publication/learning-deep-
# structured-semantic-models-for-web-search-using-clickthrough-data/
# An implementation of the Deep Semantic Similarity Model (DSSM) found in [1].

import numpy as np

from keras import backend
from keras.layers import Activation, Input
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot
from keras.models import Model

def create_model():
	LETTER_GRAM_SIZE = 3 # See section 3.2.
	WINDOW_SIZE = 3 # See section 3.2.
	TOTAL_LETTER_GRAMS = int(3 * 1e4) # Determined from data. See section 3.2.
	WORD_DEPTH = 51253 # See equation (1).
	K = 300 # Dimensionality of the max-pooling layer. See section 3.4.
	L = 128 # Dimensionality of latent semantic space. See section 3.5.
	J = 3 # Number of random unclicked documents serving as negative examples for a query. See section 4.
	FILTER_LENGTH = 1 # We only consider one time step for convolutions.

	# Input tensors holding the query, positive (clicked) document, and negative (unclicked) documents.
	# The first dimension is None because the queries and documents can vary in length.
	query = Input(shape = (WORD_DEPTH,), name="query")
	pos_doc = Input(shape =(WORD_DEPTH,), name="pos_doc")
	neg_docs = [Input(shape = (WORD_DEPTH,), name="neg_doc_" + str(j)) for j in range(J)]

	# Next, we apply a projection layer to the convolved query matrix.
	query_proj = Dense(K, name="q_proj_1", activation="tanh")(query) # See section 3.4.
	query_proj_2 = Dense(K, name="q_proj_2", activation="tanh")(query_proj) # See section 3.4.

	# In this step, we generate the semantic vector represenation of the query. 
	query_sem = Dense(L, name="q_sem", activation = "tanh")(query_proj_2) # See section 3.5.

	# The document equivalent of the above query model.
	doc_proj = Dense(K, name="d_proj_1", activation="tanh")
	doc_proj_2 = Dense(K, name="d_proj_2", activation="tanh")
	doc_sem = Dense(L, name="d_sem", activation = "tanh")

	pos_doc_proj = doc_proj(pos_doc)
	neg_doc_projs = [doc_proj(neg_doc) for neg_doc in neg_docs]

	pos_doc_proj2 = doc_proj_2(pos_doc_proj)
	neg_doc_proj2s = [doc_proj_2(neg_doc_proj) for neg_doc_proj in neg_doc_projs]

	pos_doc_sem = doc_sem(pos_doc_proj2)
	neg_doc_sems = [doc_sem(neg_doc_proj2) for neg_doc_proj2 in neg_doc_proj2s]

	# This layer calculates the cosine similarity between the semantic representations of
	# a query and a document.
	R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True, name="pos_cos") # See equation (4).
	R_Q_D_ns = [dot([query_sem, neg_doc_sem], axes = 1, normalize = True, name="neg_cos_" + str(i)) for i, neg_doc_sem in enumerate(neg_doc_sems)] # See equation (4).

	concat_Rs = concatenate([R_Q_D_p] + R_Q_D_ns)
	concat_Rs = Reshape((J + 1, 1))(concat_Rs)

	# We're also going to learn gamma's value by pretending it's a single 1 x 1 kernel.
	weight = np.full((1, 1, 1), 1)
	with_gamma = Convolution1D(1, 1, padding = "same", input_shape = (J + 1, 1), activation = "linear", use_bias = False, weights = [weight])(concat_Rs)
	with_gamma = Reshape((J + 1, ))(with_gamma)

	# Finally, we use the softmax function to calculate P(D+|Q).
	prob = Activation("softmax")(with_gamma) # See equation (5).

	# We now have everything we need to define our model.
	model = Model(inputs = [query, pos_doc] + neg_docs, outputs = prob)
	model.compile(optimizer = "adadelta", loss = "categorical_crossentropy", metrics=["accuracy"])

	model.summary()
	from keras.utils import plot_model
	plot_model(model, to_file='model.png')
	return model

if __name__ == "__main__":
    create_model()
