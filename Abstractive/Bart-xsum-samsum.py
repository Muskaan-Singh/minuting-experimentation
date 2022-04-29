from transformers import pipeline
import os
import sys
from tqdm import tqdm 
from utils import segment_text
from transformers import pipeline
import os

sum_path = sys.argv[0] # path to submission folder
path_to_transcript_folders = sys.argv[1] # path to transcript folder

summarizer = pipeline("summarization", device=0, model="lidiya/bart-large-xsum-samsum") 

model_name = "Bart-xsum-samsum"
if "Bart-xsum-samsum" not in os.listdir(sum_path):
  os.mkdir("{}/Bart-xsum-samsum".format(sum_path))

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
                segments_refined = segment_text(transcript,450)
                summary = []

                for ind,segment in enumerate(segments_refined):
                    print("{}/{} segments".format(ind+1,len(segments_refined)))
                    summary.append(summarizer(segment,min_length=0,))

                final_summary = ""
                for i,summ in enumerate(summary):
                    text = summ[0]["summary_text"]
                    final_summary += text

                final_summary = final_summary.replace(". ",".\n")
                open("{}/{}/{}/{}/{}".format(sum_path,model_name,folder_name,transcript_file,file),"w",encoding="utf-8").write(final_summary)