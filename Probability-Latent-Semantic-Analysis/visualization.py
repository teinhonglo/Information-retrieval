# -*- coding: utf-8 -*-
import io
from wordcloud import WordCloud
import matplotlib
import cPickle as pickle
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
total_vocab = []
comp_vocab = []
prob_word_topic = []
topN = 20
top_len = 16

with io.open("../Corpus/TDT2/LDC_Lexicon.txt", "r") as ldc_file:
    for line in ldc_file.readlines():
        w_ID, word =  line.split()
        total_vocab.append(word)


with open("exp/w_IDs.pkl", "rb") as f: comp_vocab = pickle.load(f)
with open("exp/pwz.pkl", "rb") as f: prob_word_topics = pickle.load(f)

font = 'Fonts/simfang.ttf'

for topic_idx, prob_word in enumerate(prob_word_topics):
    text = u" "
    high_freq_word_idx = np.argsort(prob_word)[-topN:-(topN - top_len)]
    high_freq_word = np.sort(prob_word)[-topN:-(topN - top_len)]
    print high_freq_word_idx
    min_freq = high_freq_word[0] 
    print min_freq
    word_freq = np.round(high_freq_word / min_freq)
    for idx, freq in enumerate(word_freq):
        w_ID = comp_vocab[high_freq_word_idx[idx]]
        word = total_vocab[w_ID]
        for it in xrange(int(freq)):
            text += word + " "
    # the font from github: https://github.com/adobe-fonts
    wc = WordCloud(collocations=False, font_path=font, width=1400, height=1400, margin=2).generate(text)
    plt.imshow(wc)
    plt.axis("off")
    wc.to_file('exp/topic_' + str(topic_idx) + '.png')  # save wordcloud
