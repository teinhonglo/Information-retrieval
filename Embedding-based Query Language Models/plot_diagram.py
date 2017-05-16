import matplotlib.pyplot as plt
import numpy as np
import ProcDoc
import operator
import cPickle as Pickle

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
	# plt.plot(range(len(significant_list)), significant_list, label='significant')
	# plt.plot(range(len(unigram_list)), unigram_list, label='unigram')
	plt.title('Loss')
	plt.legend(loc='upper left')
	plt.title('Accuracy')
	plt.show()
	r = raw_input()
	
def plotList(x_axis, objList, title, curve):
	draw_list = np.array(objList)
	maxVal = 1.0 * draw_list.max(axis = 0)
	plt.figure(8)
	plt.plot(x_axis, objList,label = curve)
	plt.title('Loss')
	plt.legend(loc='upper left')
	plt.title(title)
	plt.show()
	#r = raw_input()

def main():
	a_list = np.linspace(10, 50, num=5)
	m_list = np.linspace(10, 80, num=71)
	line_style = ["g", "r", "y", "c", "m"]
	eqe1 = []
	eqe2 = []
	for a in a_list:
		with open("model/mAP_list_a" + str(int(a)) + "_EQE1.pkl", "rb") as file:
			eqe1.append(Pickle.load(file))
		with open("model/mAP_list_a" + str(int(a)) + "_EQE2.pkl", "rb") as file:
			eqe2.append(Pickle.load(file))
			
	plt.figure(8)
	plt.title("Conditional Independence of Query Terms")
	plt.xlabel("m")
	plt.ylabel("mAP")
	for a_val in a_list:
		a_idx = int(a_val / 10 - 1)
		plt.plot(m_list, eqe1[a_idx], line_style[a_idx], label = "a = " + str(a_val))
	plt.legend(loc='upper right')	
	plt.show()
	
	plt.figure(9)	
	plt.title("Query-Independent Term Similarities")
	plt.xlabel("m")
	plt.ylabel("mAP")
	for	a_val in a_list:
		a_idx = int(a_val / 10 - 1)
		plt.plot(m_list, eqe2[a_idx], line_style[a_idx], label = "a = " + str(a_val))
	plt.legend(loc='upper right')	
	plt.show()	
	
	
if __name__ == "__main__":
	main()
	