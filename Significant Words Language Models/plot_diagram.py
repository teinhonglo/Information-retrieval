import matplotlib.pyplot as plt
import ProcDoc
import operator

def plotModel(general_model, specific_model, significant_model, feedback_doc_wc, feedback_doc_unigram):
	
	general_model_softmax = {}					
	general_list = []
	specific_list = []
	significant_list = []
	unigram_list = []
	feedback_wc = {}
	feedback_wu = {}
	
	for doc, wc in feedback_doc_wc.items():
		total_word_sum = ProcDoc.word_sum(wc)
		for word, count in wc.items():
			if word in feedback_wc:
				feedback_wc[word] += count
				feedback_wu[word] += total_word_sum * feedback_doc_unigram[doc][word]
			else:
				feedback_wc[word] = count
				feedback_wu[word] = total_word_sum * feedback_doc_unigram[doc][word]
			
			
	feedback_wc = sorted(feedback_wc.items(), key=operator.itemgetter(1), reverse = True)
	total_word_sum = ProcDoc.word_sum(dict(feedback_wc))
	for word, count in feedback_wc:
		general_list.append(count)
		specific_list.append(total_word_sum * specific_model[word])
		significant_list.append(total_word_sum * significant_model[word])
		unigram_list.append(feedback_wu[word])

	import matplotlib.pyplot as plt
	plt.figure(8)
	plt.plot(range(len(general_list)), general_list,label='general')
	plt.plot(range(len(specific_list)), specific_list,label='specific')
	#plt.plot(range(len(significant_list)), significant_list,label='significant')
	#plt.plot(range(len(unigram_list)), unigram_list,label='unigram')
	plt.title('Loss')
	plt.legend(loc='upper left')
	plt.title('Accuracy')
	plt.show()
	r = raw_input()