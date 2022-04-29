from transformers import pipeline
import os
import sys
from tqdm import tqdm 
from summarizer import Summarizer
from utils import segment_text

## BART.py
from transformers import pipeline
import os

sum_path = sys.argv[0] # path to submission folder
path_to_transcript_folders = sys.argv[1] # path to transcript folder

summarizer = Summarizer('distilbert-base-uncased', hidden=[-1,-2], hidden_concat=True)

model_name = "BertSUM"
if "BertSUM" not in os.listdir(sum_path):
  os.mkdir("{}/BertSUM".format(sum_path))

for trans_path in tqdm(path_to_transcript_folders, total=len(path_to_transcript_folders)):
    transcript_files = os.listdir(trans_path)
    folder_name = trans_path.split("/")[-1].split('_')[0]
    path_to_model_folder = os.path.join(sum_path, model_name)
    os.mkdir(os.path.join(path_to_model_folder, folder_name))
    path_to_data_folder = os.path.join(path_to_model_folder, folder_name)
    for index, transcript_file in tqdm(enumerate(transcript_files), total=len(transcript_files)):
        if transcript_file == "meeting_en_test_005":
            continue
        path_to_transcript = os.path.join(trans_path, transcript_file)
        os.mkdir(os.path.join(path_to_data_folder, transcript_file))
        for id, file in enumerate(os.listdir(path_to_transcript)):
            print(file)
            if ".txt" in file:
                transcript = open(os.path.join(path_to_transcript, file),"r",encoding="utf-8").read()
                final_summary = summarizer(transcript, num_sentences=30)
                final_summary = final_summary.replace(". ",".\n")
                open("{}/{}/{}/{}/{}".format(sum_path,model_name,folder_name,transcript_file,file),"w",encoding="utf-8").write(final_summary)