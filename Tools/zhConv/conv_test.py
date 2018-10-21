#coding: utf-8
from langconv import Converter
sentence = "测试"
line = Converter('zh-hant').convert(sentence.decode('utf-8'))
line = line.encode('utf-8')
print ("簡轉繁(原): " + sentence)
print ("簡轉繁(轉): " + line)
