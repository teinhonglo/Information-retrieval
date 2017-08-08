import cPickle as Pickle

with open("qry_val_set.pkl", "rb") as file: qry_val_set = Pickle.load(file)
print len(qry_val_set)

