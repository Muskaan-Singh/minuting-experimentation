# Roberta2Roberta.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import segement_text
from tqdm import tqdm
import sys
import os

sum_path = sys.argv[0] # path to submission folder
path_to_transcript_folders = sys.argv[1] # path to transcript folder.

tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_cnn_daily_mail")
model = AutoModelForSeq2SeqLM.from_pretrained("google/roberta2roberta_L-24_cnn_daily_mail").to("cuda")

model_name = "Roberta2Roberta"
if "Roberta2Roberta" not in os.listdir(sum_path):
  os.mkdir("{}/Roberta2Roberta".format(sum_path))

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
        for id, file in enumerate(os.listdir(path_to_transcript)):  
            path_to_transcript = os.path.join(trans_path, transcript_file)
            os.mkdir(os.path.join(path_to_data_folder, transcript_file))
            if ".txt" in file:
                transcript = open(os.path.join(path_to_transcript, file),"r",encoding="utf-8").read()
                # print("{}files".format(id+1))
                segments_refined = segment_text(transcript,200)

                summary = []
                for ind,segment in enumerate(segments_refined):
                    # print("{}/{} segments".format(ind+1,len(segments_refined)))
                    input_ids = tokenizer(segment, return_tensors="pt").input_ids.to("cuda").long()
                    output_ids = model.generate(input_ids)[0]
                    summary.append(tokenizer.decode(output_ids, skip_special_tokens=True))

                final_summary = ""
                for i,summ in enumerate(summary):
                    text = summ
                    final_summary += text

                final_summary = final_summary.replace(". ",".\n")
                open("{}/{}/{}/{}/{}".format(sum_path,model_name,folder_name,transcript_file,file),"w",encoding="utf-8").write(final_summary)