# -*- coding: utf-8 -*-
import ProcDoc
from gensim import corpora, models, matutils
from sklearn.cluster import KMeans
documents = ProcDoc.read_doc()
			 
# remove common words and tokenize
texts = [[word for word in document.lower().split()] for document in documents]

texts = [[token for token in text] for text in texts]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

print "TFIDF:"
corpus_tfidf = matutils.corpus2csc(corpus_tfidf).transpose()
print corpus_tfidf
print "__________________________________________"

num_of_clusters = 60
kmeans = KMeans(n_clusters = num_of_clusters)
doc_cluster = kmeans.fit_predict(corpus_tfidf)
clusters = [[] for i in num_of_clusters]

doc_index = 0
for cluster in doc_cluster:
	clusters[cluster].append(doc_index)
	doc_index += 1
print clusters

cluster_Num = 0

with open("cluster.txt", 'w') as outfile:
	for cluster in clusters:
		cluster_Name = "cluster" + str(cluster_Num) + ", "
		outfile.write(cluster_Name)
		docName = ""
		for doc in cluster:
			docName += " " + str(doc)
		outfile.write(docName)
		outfile.write("\n")	
		cluster_Num += 1
