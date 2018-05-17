import numpy as np
import time
np.random.seed(1331)
from gensim.models import Word2Vec
import cPickle as pickle

fname="data/word2vec.pickle"
model = Word2Vec.load(fname)
wv  = model.wv
vocab = model.wv.vocab
wv_dict = {}
for key in vocab:
    wv_dict[key] = wv[key]
    
with open("data/word2vec_dict.pkl", "wb") as f_dict: pickle.dump(wv_dict, f_dict, True)
with open("data/word2vec_dict.pkl", "rb") as f_dict: wv_dict = pickle.load(f_dict)
print wv_dict.keys()[1289], wv_dict[wv_dict.keys()[1289]]
