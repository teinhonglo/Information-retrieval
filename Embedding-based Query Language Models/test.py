import word2vec_model
import cPickle as Pickle

word2vec = word2vec_model.word2vec_model()
a = {1:"3", 4:"4", 2:"5"}
with open("test.pkl", "wb") as output:
	Pickle.dump(a, output, True)

with open("collection_total_similarity.pkl", "rb") as file:
	query_model = Pickle.load(file)
print (query_model)	