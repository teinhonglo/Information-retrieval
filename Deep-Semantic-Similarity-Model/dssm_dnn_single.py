# Michael A. Alcorn (malcorn@redhat.com)
# paper: Learning Deep Structured Semantic Models for Web Search using Clickthrough Data
# url: https://www.microsoft.com/en-us/research/publication/learning-deep-
# structured-semantic-models-for-web-search-using-clickthrough-data/
# An implementation of the Deep Semantic Similarity Model (DSSM) found in.

import numpy as np

from keras import backend
from keras.layers import Activation, Input
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.merge import concatenate, dot
from keras.models import Model

WORD_DEPTH = 51253
K = 300 # Dimensionality of the projetion layer. See section 3.1.
L = 128 # Dimensionality of latent semantic space. See section 3.1.
J = 40 # Number of random unclicked documents serving as negative examples for a query. See section 3.

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
#concat_Rs = Reshape((J + 1, 1))(concat_Rs)

# In this step, we multiply each R(Q, D) value by gamma. In the paper, gamma is
# described as a smoothing factor for the softmax function, and it's set empirically
# on a held-out data set.
weight = np.full((1, 640, 1), 1)

# We're also going to learn gamma's value by pretending it's a single 1 x 1 kernel.
#with_gamma = Convolution1D(1, 1, padding = "same", input_shape = (J + 1, 1), activation = "linear", use_bias = False, weights = [weight])(concat_Rs) # See equation (5).
#with_gamma = Reshape((J + 1, ))(with_gamma)

# Finally, we use the softmax function to calculate P(D+|Q).
prob = Activation("softmax")(concat_Rs) # See equation (5).

# We now have everything we need to define our model.
model = Model(inputs = [query, pos_doc] + neg_docs, outputs = prob)
model.compile(optimizer = "adadelta", loss = "categorical_crossentropy")

# Build a random data set.
'''
sample_size = 10
l_Qs = []
pos_l_Ds = []

# Variable length input must be handled differently from padded input.
BATCH = True

(query_len, doc_len) = (5, 100)

for i in range(sample_size):
    
    if BATCH:
        l_Q = np.random.rand(query_len, WORD_DEPTH)
        l_Qs.append(l_Q)
        
        l_D = np.random.rand(doc_len, WORD_DEPTH)
        pos_l_Ds.append(l_D)
    else:
        query_len = np.random.randint(1, 10)
        l_Q = np.random.rand(1, query_len, WORD_DEPTH)
        l_Qs.append(l_Q)
        
        doc_len = np.random.randint(50, 500)
        l_D = np.random.rand(1, doc_len, WORD_DEPTH)
        pos_l_Ds.append(l_D)

neg_l_Ds = [[] for j in range(J)]
for i in range(sample_size):
    possibilities = list(range(sample_size))
    possibilities.remove(i)
    negatives = np.random.choice(possibilities, J, replace = False)
    for j in range(J):
        negative = negatives[j]
        neg_l_Ds[j].append(pos_l_Ds[negative])
'''
model.summary()
from keras.utils import plot_model
plot_model(model, to_file='model.png')

'''
if BATCH:
    y = np.zeros((sample_size, J + 1))
    y[:, 0] = 1
    
    l_Qs = np.array(l_Qs)
    pos_l_Ds = np.array(pos_l_Ds)
    for j in range(J):
        neg_l_Ds[j] = np.array(neg_l_Ds[j])
    
    history = model.fit([l_Qs, pos_l_Ds] + [neg_l_Ds[j] for j in range(J)], y, epochs = 1, verbose = 0)
else:
    y = np.zeros(J + 1).reshape(1, J + 1)
    y[0, 0] = 1
    
    for i in range(sample_size):
        history = model.fit([l_Qs[i], pos_l_Ds[i]] + [neg_l_Ds[j][i] for j in range(J)], y, epochs = 1, verbose = 0)

# Here, I walk through how to define a function for calculating output from the
# computational graph. Let's define a function that calculates R(Q, D+) for a given
# query and clicked document. The function depends on two inputs, query and pos_doc.
# That is, if you start at the point in the graph where R(Q, D+) is calculated
# and then work backwards as far as possible, you'll end up at two different starting
# points: query and pos_doc. As a result, we supply those inputs in a list to the
# function. This particular function only calculates a single output, but multiple
# outputs are possible (see the next example).
get_R_Q_D_p = backend.function([query, pos_doc], [R_Q_D_p])
if BATCH:
    get_R_Q_D_p([l_Qs, pos_l_Ds])
else:
    get_R_Q_D_p([l_Qs[0], pos_l_Ds[0]])

# A slightly more complex function. Notice that both neg_docs and the output are
# lists.
get_R_Q_D_ns = backend.function([query] + neg_docs, R_Q_D_ns)
if BATCH:
    get_R_Q_D_ns([l_Qs] + [neg_l_Ds[j] for j in range(J)])
else:
    get_R_Q_D_ns([l_Qs[0]] + neg_l_Ds[0])



if BATCH:
    y = np.zeros((sample_size, J + 1))
    y[:, 0] = 1
    
    l_Qs = np.array(l_Qs)
    pos_l_Ds = np.array(pos_l_Ds)
    for j in range(J):
        neg_l_Ds[j] = np.array(neg_l_Ds[j])
    
    history = model.fit([l_Qs, pos_l_Ds] + [neg_l_Ds[j] for j in range(J)], y, epochs = 1, verbose = 0)
else:
    y = np.zeros(J + 1).reshape(1, J + 1)
    y[0, 0] = 1
    
    for i in range(sample_size):
        history = model.fit([l_Qs[i], pos_l_Ds[i]] + [neg_l_Ds[j][i] for j in range(J)], y, epochs = 1, verbose = 1)

# Here, I walk through how to define a function for calculating output from the
# computational graph. Let's define a function that calculates R(Q, D+) for a given
# query and clicked document. The function depends on two inputs, query and pos_doc.
# That is, if you start at the point in the graph where R(Q, D+) is calculated
# and then work backwards as far as possible, you'll end up at two different starting
# points: query and pos_doc. As a result, we supply those inputs in a list to the
# function. This particular function only calculates a single output, but multiple
# outputs are possible (see the next example).
get_R_Q_D_p = backend.function([query, pos_doc], [R_Q_D_p])
if BATCH:
    get_R_Q_D_p([l_Qs, pos_l_Ds])
else:
    get_R_Q_D_p([l_Qs[0], pos_l_Ds[0]])

# A slightly more complex function. Notice that both neg_docs and the output are
# lists.
get_R_Q_D_ns = backend.function([query] + neg_docs, R_Q_D_ns)
if BATCH:
    get_R_Q_D_ns([l_Qs] + [neg_l_Ds[j] for j in range(J)])
else:
    get_R_Q_D_ns([l_Qs[0]] + neg_l_Ds[0])
'''
