import cPickle as Pickle
import numpy as np

vocabulary_size = 51253

with open("query_list.pkl", "rb") as file: query_list = Pickle.load(file)
with open("rel_swlm_entropy_S_9.pkl", "rb") as file: significant_word = Pickle.load(file)

rel_model = []

for qry_key in query_list:
	vocabulary = np.zeros(vocabulary_size)
	for word, count in significant_word[qry_key].items():
		vocabulary[int(word)] = count
	vocabulary /= vocabulary.sum(axis = 0)	
	rel_model.append(np.copy(vocabulary))
	
rel_model = np.array(rel_model)
with open("rel_swlm_entropy_s_9.pkl", "wb") as file: Pickle.dump(rel_model, file, True)