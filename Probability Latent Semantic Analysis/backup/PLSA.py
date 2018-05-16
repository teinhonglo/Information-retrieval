from collections import defaultdict
import ProcDoc

def Probability_LSA(doc_wc_dict, doc_topic_prob_dict, topic_word_prob_dict, doc_word_topic_prob_dict):
	topic_word_prob_dict = topic_word_prob_dict
	doc_topic_prob_dict = doc_topic_prob_dict
	doc_word_topic_prob_dict = doc_word_topic_prob_dict
	doc_wc_dict = doc_wc_dict
	interative = 0
	while has_converged(interative):
		EStep(doc_wc_dict, doc_topic_prob_dict, topic_word_prob_dict, doc_word_topic_prob_dict)
		MStep(doc_wc_dict, doc_topic_prob_dict, topic_word_prob_dict, doc_word_topic_prob_dict)
		interative += 1
		print interative
	return [topic_word_prob_dict, doc_topic_prob_dict]	
	
def EStep(doc_wc_dict, doc_topic_prob_dict, topic_word_prob_dict, doc_word_topic_prob_dict):
	# P(T| D, w)
	for doc_name, word_count in doc_wc_dict.items():	
		for word, count in word_count.items():
			denominator = 0.0
			for topic, prob in doc_topic_prob_dict[doc_name].items():
				w_t = topic_word_prob_dict[topic][word]
				t_d = doc_topic_prob_dict[doc_name][topic]	
				denominator += w_t * t_d
			
			word_topic_list = defaultdict(dict)
			for topic, prob in doc_topic_prob_dict[doc_name].items():
				w_t = topic_word_prob_dict[topic][word]
				t_d = doc_topic_prob_dict[doc_name][topic]
				word_topic_list[word][topic] = w_t * t_d / denominator
			
			doc_word_topic_prob_dict[doc_name] = word_topic_list[word][topic]

def MStep(doc_wc_dict, doc_topic_prob_dict, topic_word_prob_dict, doc_word_topic_prob_dict):
	# P(w | T)
	for tp, w_prob_list in topic_word_prob_dict.items():	
		for word, word_prob in w_prob_list.items():
			denominator = 0.0
			for w, w_p in w_prob_list.items():
				for doc_name, doc_wc_list in doc_wc_dict.items():
					try:
						d_w_c = doc_wc_list[w]
						d_w_t_p = doc_word_topic_prob_dict[doc_name][w][tp]
						denominator += d_w_c * d_w_t_p
					except KeyError:
						pass
							
			molecellur = 0.0		
			for doc_name, doc_wc_list in doc_wc_dict.items():
				try:
					d_w_c = doc_wc_list[word]
					d_w_t_p = doc_word_topic_prob_dict[doc_name][word][tp]
					molecellur += d_w_c * d_w_t_p
				except KeyError:
					pass
			
			if denominator != 0.0:	
				topic_word_prob_dict[tp][word] = molecellur / denominator
		
	# P(T| D)
	for doc_name, topic_list in doc_topic_prob_dict.items():
		denominator = ProcDoc.word_sum(doc_wc_dict[doc_name]) * 1.0
		for tp, tp_prob in topic_list.items():
			molecellur = 0.0
			for d_w, doc_wc in doc_wc_dict[doc_name].items():
				try:
					d_w_c = doc_wc
					d_w_t_p = doc_word_topic_prob_dict[doc_name][d_w][tp]
					molecellur += d_w_c * d_w_t_p / denominator
				except KeyError:
					pass
			doc_topic_prob_dict[doc_name][tp] = molecellur
			
	
def has_converged(interative):
	return interative < 100