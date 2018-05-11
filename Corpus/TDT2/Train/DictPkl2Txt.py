import cPickle as Pickle

with open("QD_rel_with_DSSM.pkl", "rb") as f : rel_dict = Pickle.load(f)

for q_id, doc_list in rel_dict.items():
	print q_id
	for doc, score in doc_list:
		print doc, score
		raw_input()	