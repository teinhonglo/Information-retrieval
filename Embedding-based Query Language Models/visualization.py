import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
import word2vec_model
from sklearn.manifold import TSNE
 
 
def main():
	word_model = word2vec_model.word2vec_model()
	wv, vocabulary = load_embeddings(word_model)

	tsne = TSNE(n_components=2, random_state=0)
	np.set_printoptions(suppress=True)
	print wv[:300,:]
	Y = tsne.fit_transform(wv[:300,:])
 
	plt.scatter(Y[:, 0], Y[:, 1])
	for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
		plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
	plt.show()
 
 
def load_embeddings(word_model):
	wv = []
	vocabulary = []
	test_vec = word_model.getWord2Vec()
	test_vocab = test_vec.vocab.keys()
	for v in test_vocab:
		vocabulary.append(v)
		wv.append(test_vec[v])
	wv = np.array(wv)	
	return wv, vocabulary
 
if __name__ == '__main__':
    main()