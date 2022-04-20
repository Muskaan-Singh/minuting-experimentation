### Baseline
import numpy as np
from summarizer import SingleModel
import bs4 as bs  
import urllib.request  
import re
import argparse
import nltk
import logging
import heapq
import os

path = "../../../Transcript/"
summary_path ="../../../submissions/" 

if "Baseline" not in os.listdir(summary_path):
  os.mkdir(summary_path+"Baseline")

print('Welcome to the Baseline Model Summarizer!\n')


# text file input option
# reading in text file
for document in os.listdir(path):
# reading in text file
  if ".ipynb" not in document:
    with open(path+document, 'r',encoding="utf-8") as d:
        text_data = d.read()

    # text clean up
    text_data = re.sub(r'\[[0-9]*\]', ' ', text_data)  
    text_data = re.sub(r'\s+', ' ', text_data)  

    processed_article = re.sub('[^a-zA-Z]', ' ', text_data )  
    processed_article = re.sub(r'\s+', ' ', processed_article)

    # sentence-level tokenization of full text
    sentence_list = nltk.sent_tokenize(text_data)  

    # NLTK stopword list
    stopwords = nltk.corpus.stopwords.words('english')

    # creating term frequency dict
    word_frequencies = {}  
    for word in nltk.word_tokenize(processed_article):  
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequency = max(word_frequencies.values())

    # adding term frequency ratios as dict values
    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

    # ranking sentences for summary inclusion
    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    # creating final summary with default 4 highest-scoring sentences
    summary_sentences = heapq.nlargest(50, sentence_scores, key=sentence_scores.get)
    summary_sentences = ''.join(summary_sentences)
    summary_file = summary_sentences

    # printing summary and full-text output for comparison

    # appending summary to text file
    with open("{}Baseline/{}".format(summary_path,document), 'w',encoding="utf-8") as summary_output:
        for line in summary_file:
            summary_output.write(line.replace(".",".\n"))
