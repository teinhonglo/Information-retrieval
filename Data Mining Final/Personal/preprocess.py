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
				anime_attr[attr_names.index(an)] = float(0)
			
		new_anime[anime_name] = anime_attr
large_matrix = []		
for an, av in new_anime.items():
	print an
	print av
	large_matrix.append(av)

max_col =  np.array(large_matrix).max(axis = 0)
print max_col

re_anime_item = {}
for an, av in new_anime.items():	
	av = np.array(av) / max_col
	re_anime_item[an] = av

for an, av in re_anime_item.items():
	print an
	print av

with io.open("animate_re.txt", 'r', encoding = 'utf8') as f:
	for an, av in re_anime_item.items():
		io.write(an)
		animate_value = ""
		for v in av:
			