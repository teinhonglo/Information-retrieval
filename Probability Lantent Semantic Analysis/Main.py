import ProcDoc
import PLSA
import codecs

INIT_PROBABILITY = 1.0 / 60
topic_word_prob_dict = ProcDoc.read_clusters()		# read cluster P(W|T), {T: {W:Prob}}
doc_topic_prob_dict = {}							# P(T|D),{D:{T:Prob}} 
doc_word_topic_prob_dict = {}						# P(T| w, D), {D: {word:{T:prob}}}
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

# initialize P(T|D)
print "Initialize P(T|D)"
for docName, wordCount in doc_wc_dict.items():
	topic_prob = {}
	for topic, wordProb in topic_word_prob_dict.items():
		topic_prob[topic] = INIT_PROBABILITY
	doc_topic_prob_dict[docName] = topic_prob

print "Initialize P(T| w, D)"
for docName, wordCount in doc_wc_dict.items():	
	word_list = {}
	for word, frequency in wordCount.items():	
		topic_prob = {}
		for topic, wordProb in topic_word_prob_dict.items():
			topic_prob[topic] = 0.0
		word_list[word] = topic_prob
	doc_word_topic_prob_dict[docName] = word_list

print "start PLSA"	
[topic_word_prob_dict, doc_topic_prob_dict, doc_wc_dict] = PLSA.Probability_LSA(bg_word, doc_wc_dict, doc_topic_prob_dict, topic_word_prob_dict, doc_word_topic_prob_dict)
	