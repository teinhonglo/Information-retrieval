from collections import defaultdict
documents = ["Human machine interface for lab abc computer applications",
			"A survey of user opinion of computer system response time",
			"The EPS user interface management system",
			"System and human system engineering testing of EPS",
			"Relation of user perceived response time to error measurement",
			"The generation of random binary unordered trees",
			"The intersection graph of paths in trees",
			"Graph minors IV Widths of trees and well quasi ordering",
			"Graph minors A survey"]
			
texts = [[word for word in document.lower().split()] for document in documents]

inverted_index = defaultdict(list)

for index in range(len(texts)):
	for word in texts[index]:
		inverted_index[word].append(index)
		
from pprint import pprint
pprint(inverted_index)