import word2vec_model
import cPickle as Pickle
import plot_diagram
import operator
import numpy as np

word_model = word2vec_model.word2vec_model()
wv = word_model.getWord2Vec()
vocab = wv.vocab
w1 = vocab.keys()[0]
word_sim_dict = {}

for v in vocab:
	word_sim_dict[v] = word_model.getWordSimilarity(wv[w1] / np.sqrt((wv[w1] ** 2 ).sum(axis = 0)), wv[v] / np.sqrt((wv[v] ** 2 ).sum(axis = 0)))
word_list = sorted(word_sim_dict.items(), key=operator.itemgetter(1), reverse = True)
word_rank = [i[1] for i in word_list]

import matplotlib.pyplot as plt
plt.figure(8)
plt.plot(range(1000), word_rank[:1000],label = "a = 50")

word_model.setAlpha(20)
for v in vocab:
	word_sim_dict[v] = word_model.getWordSimilarity(wv[w1] / np.sqrt((wv[w1] ** 2 ).sum(axis = 0)), wv[v] / np.sqrt((wv[v] ** 2 ).sum(axis = 0)))
word_list = sorted(word_sim_dict.items(), key=operator.itemgetter(1), reverse = True)
word_rank = [i[1] for i in word_list]

plt.plot(range(1000), word_rank[:1000],label = "a = 20")

word_model.setAlpha(10)
for v in vocab:
	word_sim_dict[v] = word_model.getWordSimilarity(wv[w1] / np.sqrt((wv[w1] ** 2 ).sum(axis = 0)), wv[v] / np.sqrt((wv[v] ** 2 ).sum(axis = 0)))
word_list = sorted(word_sim_dict.items(), key=operator.itemgetter(1), reverse = True)
word_rank = [i[1] for i in word_list]

plt.plot(range(1000), word_rank[:1000],label = "a = 10")

for v in vocab:
	word_sim_dict[v] = word_model.getWordSimilarityCosine(wv[w1] / np.sqrt((wv[w1] ** 2 ).sum(axis = 0)), wv[v] / np.sqrt((wv[v] ** 2 ).sum(axis = 0)))
word_list = sorted(word_sim_dict.items(), key=operator.itemgetter(1), reverse = True)
word_rank = [i[1] for i in word_list]

plt.plot(range(1000), word_rank[:1000],label = "cosine_measure")

plt.title('Similiarity')
plt.legend(loc='upper left')
plt.title("Word " + w1)
plt.show()

