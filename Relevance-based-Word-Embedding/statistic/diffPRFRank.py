import numpy as np
import cPickle as Pickle
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt


print "read file"
with open("query_list.pkl", "rb") as f: query_list = Pickle.load(f)
with open("um_rank.pkl", "rb") as f: uni_rank = Pickle.load(f)
with open("rm_rank.pkl", "rb") as f: rm_rank = Pickle.load(f)
with open("swm_rank.pkl", "rb") as f: swm_rank = Pickle.load(f)
with open("bg_rank.pkl", "rb") as f: bg_rank = Pickle.load(f)

uni_mAP = []
rm_mAP = []
swm_mAP = []
bg_mAP = []
print "term re-rank"
for qry_name in query_list:
	uni_mAP.append(uni_rank[qry_name])
	rm_mAP.append(rm_rank[qry_name])
	swm_mAP.append(swm_rank[qry_name])
	bg_mAP.append(bg_rank[qry_name])

import csv

with open("diffMD.csv", 'wb') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
	wr.writerow(uni_mAP)
	wr.writerow(rm_mAP)
	wr.writerow(swm_mAP)
	wr.writerow(bg_mAP)

print "Plot..."
plt.figure(figsize=(18,18))
print "unigram"
plt.plot(range(len(uni_mAP)), uni_mAP, label='um')
print "relevacne model"
plt.plot(range(len(rm_mAP)), rm_mAP, label='rm')
print "significant word model"
plt.plot(range(len(swm_mAP)), swm_mAP, label='swm')
plt.title('mAP')
plt.legend(loc='upper right')
print "down"
#plt.show()
plt.savefig('ranking results.png',format='png')

	