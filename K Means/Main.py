# -*- coding: utf-8 -*- 
import ProcDoc

# read document. (Doc No.,Doc content)
docDict = ProcDoc.read_doc()
bg_word = {}

for docName, content in docDict.items():
	temp_dict = ProcDoc.word_count(content, {})
	docDict[docName] = temp_dict
	for word, frequency in temp_dict.items():
		if word in bg_word:
			bg_word[word] += int(frequency)
		else:
			bg_word[word] = int(frequency)

for docName, word_dict in docDict.items():
	word_vector = []
	for word, frequency in bg_word.items():
		if word in word_dict:
			word_vector.append(1)
		else:
			word_vector.append(0)	
	docDict[docName] = word_vector

ProcDoc.printDict(docDict)
ProcDoc.printDict(bg_word)