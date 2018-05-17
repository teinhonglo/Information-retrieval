import numpy as np
import ProcDoc
import cPickle as Pickle

corpus = "TDT2"
model_path = "../Corpus/model/"+corpus+"/UM/"

with open(model_path + "query_model.pkl", "rb") as f: qry_model = Pickle.load(f)
with open(model_path + "doc_model.pkl", "rb") as f: doc_model = Pickle.load(f)
background = ProcDoc.read_background_dict()
qry_smooth_alpha = 0.
doc_smooth_alpha = 0.8

background_model = ProcDoc.read_background_dict()
print background_model.shape


for idx, vec in enumerate(doc_model):
    doc_model[idx] = (1 - doc_smooth_alpha) * vec + doc_smooth_alpha * background

for idx, vec in enumerate(qry_model):
    qry_model[idx] = (1 - qry_smooth_alpha) * vec + qry_smooth_alpha * background

LM_score = np.dot(qry_model, np.log(doc_model).T)
with open("LM_score.pkl", "wb") as f: Pickle.dump(LM_score, f, True)


