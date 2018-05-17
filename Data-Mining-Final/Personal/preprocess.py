#-*- coding: utf-8 -*-
import re
import codecs
import time
import io
import itertools
import numpy as np
attr_names = []

with io.open("animate.txt", 'r', encoding = 'utf8') as f:
	# read content
	content = f.readlines()
	content = [x.strip() for x in content] 
	for anime in content:
		anime_info = anime.split(",")
		anime_name = anime_info[0]
		anime_attr_name = anime_info[1::2]
		anime_attr_value = anime_info[2::2]
		'''
		print anime_name
		print anime_attr_name
		print anime_attr_value
		'''
		attr_names += anime_attr_name
		
attr_names.sort()
attr_names = list(attr_names for attr_names,_ in itertools.groupby(attr_names))
print attr_names

new_anime = {}
with io.open("animate.txt", 'r', encoding = 'utf8') as f:
	# read content
	content = f.readlines()
	content = [x.strip() for x in content] 
	for anime in content:
		anime_info = anime.split(",")
		anime_name = anime_info[0]
		anime_attr_name = anime_info[1::2]
		anime_attr_value = anime_info[2::2]
		
		
		#print anime_name
		anime_attr = [0] * len(attr_names)
		
		for an in anime_attr_name:
			
			if anime_attr_value[anime_attr_name.index(an)] != '--':
				anime_attr[attr_names.index(an)] = float(anime_attr_value[anime_attr_name.index(an)])
			else:	
				anime_attr[attr_names.index(an)] = float(7)
			
		new_anime[anime_name] = anime_attr
		
large_matrix = []		

for an, av in new_anime.items():
	print an
	print av
	large_matrix.append(av)

max_col =  np.array(large_matrix).max(axis = 0)
min_col =  np.array(large_matrix).min(axis = 0)
print max_col

re_anime_item = {}
for an, av in new_anime.items():	
	av = (np.array(av) - min_col)/ (max_col- min_col)
	# av = np.array(av) / max_col
	re_anime_item[an] = av

for an, av in re_anime_item.items():
	print an
	print av

with io.open("animate_re.txt", 'w', encoding = 'utf8') as output:
	
	title = ""
	for an in attr_names:
		title += "," + an
	
	output.write(title + "\n")	
	for an, av in re_anime_item.items():
		output.write(an)
		anime_value = ""
		for v in av:
			anime_value += "," + str(v)
		anime_value = unicode(anime_value, "utf-8")	
		output.write(anime_value + "\n")