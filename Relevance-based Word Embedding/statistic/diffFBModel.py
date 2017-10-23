import numpy as np
import cPickle as Pickle
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt

qry_idx = 10
print "read file"
with open("query_model.pkl", "rb") as f: query_model = Pickle.load(f)
with open("doc_model.pkl", "rb") as f: doc_model = Pickle.load(f)
with open("query_list.pkl", "rb") as f: query_list = Pickle.load(f)
with open("UM.pkl", "rb") as f: uni_model = Pickle.load(f)
with open("RM.pkl", "rb") as f: rm_model = Pickle.load(f)
with open("SWM.pkl", "rb") as f: swm_model = Pickle.load(f)
print str(qry_idx) + "-th query"

qry_name = query_list[qry_idx]
qry_md = query_model[qry_idx]
print qry_md.shape
uni_md = uni_model[qry_idx]
rm_md = rm_model[qry_idx]
swm_md = swm_model[qry_idx]
bg = np.sum(doc_model, axis = 0)
print bg.shape
term_ranking = np.argsort(-bg)
print term_ranking.shape


qry_md_reRank = []
uni_md_reRank = []
rm_md_reRank = []
swm_md_reRank = []
print "term re-rank"
for idx in term_ranking:
	qry_md_reRank.append(qry_md[idx])
	uni_md_reRank.append(uni_md[idx])
	rm_md_reRank.append(rm_md[idx])
	swm_md_reRank.append(swm_md[idx])
'''
qry_md_reRank = np.vstack(qry_md_reRank)
uni_md_reRank = np.vstack(uni_md_reRank)
rm_md_reRank = np.vstack(rm_md_reRank)
swm_md_reRank = np.vstack(swm_md_reRank)


import csv

with open("diffMD.csv", 'wb') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
	wr.writerow(qry_md_reRank)
	wr.writerow(uni_md_reRank)
	wr.writerow(rm_md_reRank)
	wr.writerow(swm_md_reRank)
'''
print "Plot..."
plt.figure(8)
print "origin"
plt.plot(range(len(qry_md_reRank)), qry_md_reRank, label='orgin')
print "unigram"
plt.plot(range(len(uni_md_reRank)), uni_md_reRank, label='um')
#print "relevacne model"
#plt.plot(range(len(rm_md_reRank)), rm_md_reRank, label='rm')
#print "significant word model"
#plt.plot(range(len(swm_md_reRank)), swm_md_reRank, label='swm')
plt.title('Pseudo Relevance Feedback')
plt.legend(loc='upper right')
print "down"
plt.savefig('swm.png',dpi=300,format='png')

	