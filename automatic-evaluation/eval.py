# %%capture
import os
from sumeval.metrics.rouge import RougeCalculator
import csv
from bert_score import score
import numpy as np
import sys
from collections import defaultdict
from moverscore_v2 import get_idf_dict, word_mover_score 
import sys
# import pandas as pd

path_to_submission_folder = sys.argv[0] # path to submission folder

## Usage : python eval.py model/folder name

# checking if folder name is specified or not
if(len(sys.argv) < 2):
  print("\n\nPlease specify folder/model name")
  exit(0)

models = [model for model in os.listdir(path_to_submission_folder)]

for model in models:
    
    rouge = RougeCalculator(stopwords=True, lang="en")

    # path to folder
    path = path_to_submission_folder

    # ["Dev_Transcripts", "testNew_Transcripts", "Test_Transcripts"]
    data = "Test"

    ## it will store all the summaries with the key as file name
    Dict_all_summaries = defaultdict(list)

    # current directory
    current_directory = os.getcwd()
    path_to_minuting_experimentation_directory = os.path.join(current_directory, "minuting-experimentation/")

    # minutes_files = ['testNew_Minutes', 'Test_Minutes', 'Dev_Minutes']
    minutes_files = ['Test_Minutes']
    path_to_minutes_directory = [os.path.join(path_to_minuting_experimentation_directory, folder) for folder in minutes_files]

    # list of all file_names
    file_names = []
    for minutes_path in path_to_minutes_directory:
        for minutes_folder in os.listdir(minutes_path):
            if minutes_folder == "meeting_en_test_005":
                continue
            path_to_minutes_folder = os.path.join(minutes_path, minutes_folder)
            for file in os.listdir(path_to_minutes_folder):
                # checking if file is not .ipynb checkpoint
                if ".ipynb" not in file:
                    # initializing the row of dictionary as a dict
                    row = {}
                    newPath = os.path.join(path_to_minutes_folder, file)

                    # storing the original summary of file                
                    # storing the row at corresponding file name key.
                    Dict_all_summaries[minutes_folder].append({file: open(newPath, "r", encoding="unicode_escape").read()})

                    # storing file name in array
                    file_names.append(minutes_folder)

    # the path to the summaries of the specified folder/model 
    path_to_model = os.path.join(path, model)
    newPath = os.path.join(path_to_model, minutes_files[0].split('_')[0])
    print(newPath)

    # print("h1")
    # looping over all files in the model folder
    for minutes_folder in os.listdir(newPath):
        path_to_minutes_folder = os.path.join(newPath, minutes_folder)
        for file in os.listdir(path_to_minutes_folder):
            # checking if file is not .ipynb checkpoint
            if ".ipynb" not in file:
                # storing the summary from model with the original summary for the same file
                # Example: for file_name "1", original summary and model summary can be accessed using Dict_all_summaries["1"]
                Dict_all_summaries[minutes_folder].append({model: open(os.path.join(path_to_minutes_folder, file), "r", encoding="utf8").read()})

    # print("h2")
    # checking if results .csv exists or not
    if f"{data}_results.csv" not in os.listdir():
        # if it doesnot exist, create Results.csv and a heading row
        file_csv = open(f'{data}_results.csv', 'w', newline='')
        writer = csv.writer(file_csv)
        writer.writerow(["Model", "file_name", "Rouge_1", "Rouge_2", "Rouge_l", "Rouge_be", "Word Mover Distance", "Bert_F1"])
    else:
        file_csv = open(f'{data}_results.csv', 'a', newline='')
        writer = csv.writer(file_csv)
    print(os.path.join(os.getcwd(), f"{data}_results.csv"))

    # print("h3")
    #initializing the variables which will store the total sum of values
    rouge_1_total = 0
    rouge_2_total = 0
    rouge_l_total = 0
    rouge_be_total = 0
    Word_mover_total = 0
    Bert_F1_total = 0

    file_names = list(set(file_names))

    for count, file in enumerate(file_names):
        print(file)
        # To print progress
        print("{}/{}".format(count+1,len(file_names)))

        processed_dict = dict()

        for instance in Dict_all_summaries[file]:
            key = list(instance.keys())[0]
            value = list(instance.values())[0]
            processed_dict[key] = value
        
        original_summaries_filenames = [key for key in processed_dict.keys() if ".txt" in key]

        for file_name in original_summaries_filenames:
            # getting the original summary for file
            Original_summary = processed_dict[file_name]
            # creating 2 copies of the summary generate by the model
            gen = processed_dict[model]
            gen1 = processed_dict[model]

            # # calculating Rouge Score
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

            # creating list of sentences from generated and Original summary
            text = gen1.split("\n")
            orig = Original_summary.split("\n")

            # removing sentences whose length is less than 5 (i.e. no. of charachters is less than 5 as suh sentences are empty and give wrong score)
            orig = [sent for sent in orig if len(sent) > 5]
            text = [sent for sent in text if len(sent) > 5]

            if len(text) < len(orig):
                orig = orig[:len(text)]
            elif len(orig) < len(text):
                text = text[:len(orig)]

            # creating idf dict for both the original summary and sentence
            idf_dict_hyp = get_idf_dict(text) 
            idf_dict_ref = get_idf_dict(orig) 

            # # calcuating score
            document_score = word_mover_score(text, orig, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True)
            
            # # adding the document score to dev set score
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