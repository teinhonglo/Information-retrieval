import ProcDoc
import codecs

clusters = ProcDoc.read_clusters()		# read cluster [[cluster ID, Docs]]
doc_wordCount = ProcDoc.read_doc_dict()  # read document (Doc No.,Doc content)  
bg_word = {}  # background (word, count)

# calculate word of the background
# convert (Doc No.,Doc content) to (Doc_No, {word, count})
for docName, content in doc_wordCount.items():
	temp_dict = ProcDoc.word_count(content, {})
	doc_wordCount[docName] = temp_dict
	for word, frequency in temp_dict.items():
		if word in bg_word:
			bg_word[word] += int(frequency)
		else:
			bg_word[word] = int(frequency)
		
with codecs.open("background_word_count.txt", 'w', "utf-8") as outfile:
	out_str = ""
	for word, frequency in bg_word.items():
		out_str += word + " " + str(frequency) + "\n"
	outfile.write(out_str)	

with codecs.open("cluster_word_prob.txt", 'w', "utf-8") as outfile:	
	# clusters			
	for c_ID, docs in clusters:
		cluster_word_count = {}
		# documents in the same cluster
		# word count of the cluster
		for docID in docs.split():
			# word count of the document
			for word, frequency in doc_wordCount[str(docID)].items():
				if word in cluster_word_count:
					cluster_word_count[word] += int(frequency)
				else:
					cluster_word_count[word] = int(frequency)
		# sum of word of cluster
		cluster_word_sum = ProcDoc.word_sum(cluster_word_count) * 1.0
		#probability of word
		cluster_word_probability = {}
		out_str = c_ID + "#"
		for word, num_of_word in bg_word.items():
			if word in cluster_word_count:
				word_probability = cluster_word_count[word] / cluster_word_sum
			else:
				word_probability = 0.0
			cluster_word_probability[word] = word_probability
			out_str += word + " " + str(word_probability) + ","
		outfile.write(out_str)
		outfile.write("\n")