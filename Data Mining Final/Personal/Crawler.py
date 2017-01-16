#-*- coding: utf-8 -*-
import urllib,urllib2
import re
import codecs
import time
import io
from BeautifulSoup import *
id_list = []

with io.open("animate_list.txt", 'r', encoding = 'utf8') as f:
	# read content
	id_list = f.read().split()

delay_num = 0
skip_animate = []

with codecs.open("animate.txt", 'a', "utf-8") as outfile:
	for id in id_list:
		if delay_num >= 10: 
			delay_num = 0
			skip_animate.append(id)
			continue
		url = "https://acg.gamer.com.tw/acgDetail.php?s=" + id
		#url = 'file:' + urllib.pathname2url("D:\Information-retrieval/Data Mining Final/Personal/getAnimate79000.php")
		is_next = False
		while not is_next:
			
			while True:
				#url = raw_input('Enter - ')
				
				result = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', url)
				print result
				
				if len(result) > 0:
					break
			
			
			request = urllib2.Request(url) 
			request.add_header("User-Agent","Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.1; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; MS-RTC LM 8; InfoPath.3; .NET4.0C; .NET4.0E) chromeframe/8.0.552.224")
			response = urllib2.urlopen(request)  
			
			
			encoding = response.headers['content-type'].split('charset=')[-1]
			
			html = response.read()
			try:
				html = unicode(html, 'utf8', errors="replace")
				soup = BeautifulSoup(html)
				title = soup.h1.string
				print title.encode(encoding="utf-8", errors="strict")
				# ACG score 
				ACG_score = soup.findAll("div", { "class" : "ACG-score" })
				score = str(re.split("<|>", str(ACG_score))[2])
				people = re.split("<|>", str(ACG_score))[4]
				people =  str(re.findall("[0-9]+", people)[0])
				
				# ACG data
				ACG_data = soup.findAll("div", { "class" : "ACG-data" })
				ACG_data =  unicode(str(ACG_data), 'utf-8')
				attr = re.split("<li>|</li>|<ul>|</ul>|<p>|</p>|<span>|</span>", ACG_data)
				attr = filter(None, attr)
				animate_attr = {}
				for a in range(1, 6):
					animate_attr[attr[a]] = str(re.findall("[0-9]+", attr[2 * a + 6])[0])
					
				
				outfile.write(title)
				outfile.write(",score," + score)
				outfile.write(",people," + people)
				for an, av in animate_attr.items():
					outfile.write("," + an + "," + av)
					
				outfile.write("\n")	
				is_next = True
				
				if delay_num > 0:
					delay_num = delay_num / 2
					
			except Exception as e: 
				print str(e)
				is_next = False
				time.sleep(10 + delay_num * 2)
				if delay_num < 20:
					delay_num += 1
					
				print is_next
			
print skip_animate