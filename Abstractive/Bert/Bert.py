import nltk
nltk.download('punkt')
nltk.download('stopwords')


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

if "Bert" not in os.listdir(summary_path):
  os.mkdir(summary_path+"Bert")

print('Welcome to the BERT Summarizer!\n')


for document in os.listdir(path):
# reading in text file
  if ".txt"  in document:
    # print(document)
    with open(path+document, 'r',encoding="utf-8") as d:
        text_data = d.read()
    
    # importing model and passing in full text
    model = SingleModel()
    m = model(text_data)

    # creating final summary with a ratio of 0.13
    summary_file = m

    # appending summary output to text file
    with open("{}Bert/{}".format(summary_path,document), 'w',encoding="utf-8") as summary_output:
        for line in summary_file:
            summary_output.write(line)

