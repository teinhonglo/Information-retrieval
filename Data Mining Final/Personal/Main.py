# coding=utf-8

import requests
from bs4 import BeautifulSoup

res = requests.get("https://news.google.com")
soup = BeautifulSoup(res.text)
print soup.select(".esc-body")

count = 1

for item in soup.select(".esc-body"):
    print '======[',count,']========='
    news_title = item.select(".esc-lead-article-title")[0].text
    news_url = item.select(".esc-lead-article-title")[0].find('a')['href']
    print news_title
    print news_url
    count += 1