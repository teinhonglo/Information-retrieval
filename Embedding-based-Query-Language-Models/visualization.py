import sys
import codecs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import word2vec_model
from sklearn.manifold import TSNE
 
 
def main():
    word_model = word2vec_model.word2vec_model()
    wv, vocabulary = load_embeddings(word_model)

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(wv[:1000,:])
    # area = np.pi * similarity
    plt.figure(figsize=(18, 18))
    plt.scatter(Y[:, 0], Y[:, 1])
    
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()

def visualization(collection, similiarity, filename):
    wv = []
    vocabulary = []
    wv_similarity = []
    for word, word_similarity in similiarity[:50]:
        wv.append(collection[word])
        vocabulary.append(word)
        wv_similarity.append(word_similarity)
    # numpy array
    wv = np.array(wv)
    wv_similarity = np.array(wv_similarity)
    
    # dimension reduction
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(wv[:,:])
    # area
    area = np.pi * wv_similarity * 10000
    
    plt.figure(figsize=(8, 8))
    plt.scatter(Y[:, 0], Y[:, 1], s = area)
    
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.savefig(filename, format='png')
    plt.close()
    
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