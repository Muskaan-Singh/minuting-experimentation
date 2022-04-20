import os
from sumeval.metrics.rouge import RougeCalculator
import csv
from bert_score import score
import numpy as np
from moverscore_v2 import get_idf_dict, word_mover_score 
import sys
# import pandas as pd

## Usage : python eval.py model/folder name


# checking if folder name is specified or not
if(len(sys.argv) < 2):
  print("\n\nPlease specify folder/model name")
  exit(0)



model = sys.argv[1]

rouge = RougeCalculator(stopwords=True, lang="en")

# path to folder
path = "../submissions/"



## it will store all the summaries with the key as file name
Dict_all_summaries = {}

# list of all file_names
file_names = []
# looping over all the files in Original folder
for file in os.listdir(path+"Original"):
  # checking if file is not .ipynb checkpoint
  if ".ipyn" not in file:
    # initializing the row of dictionary as a dict
    row = {}
    newPath = path+"Original/"

    # storing the original summary of file
    row["Original_Summary"] = open(newPath+file,"r",encoding="utf8").read()

    # storing the row at corresponding file name key.
    Dict_all_summaries[file] = row

    # storing file name in array
    file_names.append(file)
 

# the path to the summaries of the specified folder/model 
newPath = path + model + "/"

# print("h1")
# looping over all files in the model folder
for file in os.listdir(newPath):
  # checking if file is not .ipynb checkpoint
  if ".ipyn" not in file:
    # storing the summary from model with the original summary for the same file
    # Example: for file_name "1", original summary and model summary can be accessed using Dict_all_summaries["1"]
    Dict_all_summaries[file][model] = open(newPath+file,"r",encoding="utf8").read()

# print("h2")
# checking if results .csv exists or not
if "Results.csv" not in os.listdir():
  # if it doesnot exist, create Results.csv and a heading row
  file_csv = open('Results.csv', 'w', newline='')
  writer = csv.writer(file_csv)
  writer.writerow(["Model","Rouge_1", "Rouge_2", "Rouge_l","Rouge_be","Word Mover Distance","Bert_F1"])


else:
  file_csv = open('Results.csv', 'a', newline='')
  writer = csv.writer(file_csv)

# print("h3")
#initializing the variables which will store the total sum of values
rouge_1_total = 0
rouge_2_total = 0
rouge_l_total = 0
rouge_be_total = 0
Word_mover_total = 0
Bert_F1_total = 0


# print("h4")
for count,file in enumerate(file_names):

  # To print progress
  print("{}/{}".format(count+1,len(file_names)))

  # getting the original summary for file
  Original_summary = Dict_all_summaries[file]["Original_Summary"]

  # creating 2 copies of the summary generate by the model
  gen = Dict_all_summaries[file][model]
  gen1 = Dict_all_summaries[file][model]

  # calculating Rouge Score
  rouge_1_total += rouge.rouge_n(Original_summary,gen,n=1)
  rouge_2_total += rouge.rouge_n(Original_summary,gen,n=2)
  rouge_l_total += rouge.rouge_l(Original_summary,gen)
  rouge_be_total += rouge.rouge_be(Original_summary,gen)


  # creating a copy of original summary
  orig = Original_summary

  # if length of original summary is less, we pad it with " " charachter to make lengths equal
  if len(orig) < len(gen):
    orig = orig + (" "*(len(gen)-len(orig)))

  # if length of generate summary is less, we pad it with " " charachter to make lengths equal
  elif (len(orig) > len(gen)):
    gen = gen + (" "*(len(orig)-len(gen)))

  # calculating Bert-Score
  P_curr, R_curr, F1_curr = score(gen, orig, lang="en", verbose=False)

  # Adding the mean of each value from current document to calculate total for full dev-set 
  Bert_F1_total += F1_curr.mean()


  # # creating list of sentences from generated and Original summary
  text = gen1.split("\n")
  orig = Original_summary.split("\n")
   
  # # removing sentences whose length is less than 5 (i.e. no. of charachters is less than 5 as suh sentences are empty and give wrong score)
  
  
  orig = [sent for sent in orig if len(sent) > 5]
  text = [sent for sent in text if len(sent) > 5]
   
  if len(text) < len(orig):
    orig = orig[:len(text)]
  elif len(orig) < len(text):
    text = text[:len(orig)]
  
  
  # creating idf dict for both the original summary and sentence
  idf_dict_hyp = get_idf_dict(text) 
  idf_dict_ref = get_idf_dict(orig) 

  
  # calcuating score
  document_score = word_mover_score(orig, text, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True)
 
  # adding the document score to dev set score
  Word_mover_total += np.mean(document_score)



length = len(file_names)

# calculating averages
rouge_1_average = rouge_1_total/length
rouge_2_average = rouge_2_total/length
rouge_l_average = rouge_l_total/length
rouge_be_average = rouge_be_total/length
Word_mover_average = Word_mover_total/length
Bert_F1_average = Bert_F1_total/length

#Storing values in csv
writer.writerow([model,rouge_1_average,rouge_2_average,rouge_l_average,rouge_be_average,Word_mover_average,float(Bert_F1_average)])

file_csv.close()
