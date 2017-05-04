import word2vec_model

word2vec = word2vec_model.word2vec_model()
w2v = word2vec.getWordSimilarity("35236", "508")
print w2v
w2v = word2vec.getWordSimilarity("508", "35236")
print w2v