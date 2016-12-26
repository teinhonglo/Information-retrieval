import ProcDoc
import codecs

INIT_PROBABILITY = 1.0 / 60
topic_word_prob_dict = ProcDoc.read_clusters()		# read cluster P(W|T), {T: {W:Prob}}
doc_topic_prob_dict = {}							# P(T|D),{D:{T:Prob}} 
doc_wc_dict = ProcDoc.read_doc_dict()  				# read document (Doc No.,Doc content)  
bg_word = {}  										# background (word, count)

# calculate word of the background
# convert (Doc No.,Doc content) to (Doc_No, {word, count})
for docName, content in doc_wc_dict.items():
	temp_dict = ProcDoc.word_count(content, {})
	doc_wc_dict[docName] = temp_dict
	for word, frequency in temp_dict.items():
		if word in bg_word:
			bg_word[word] += int(frequency)
		else:
			bg_word[word] = int(frequency)

for docName, wordCount in doc_wc_dict.items():
	for topic, wordProb in topic_word_prob_dict.items():