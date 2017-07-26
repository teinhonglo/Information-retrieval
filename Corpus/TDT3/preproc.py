import os
import re

q_pat = "Query.*[0-9]+"
d_pat = "VOM[0-9]+[\.][0-9]+[\.][0-9]+[\r\n]"
wb_file_str = ""

with open("Assessment3371TDT3.txt", "rb") as file: 
	content = file.read()
	query_list = re.findall(q_pat, content)
	doc_list = re.split(q_pat, content)
	for ql, dl in zip(query_list, doc_list[1:]):
		docs = re.findall(d_pat, dl)
		wb_file_str += ql + "\n"
		for doc in docs:
			wb_file_str += doc + "\n"
		wb_file_str += "\n"	

with open("Assessment3371TDT3_clean.txt", "wb") as file: 
	file.write(wb_file_str)