# LSA.py
import os
import sys
from tqdm import tqdm
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
import nltk
nltk.download('punkt')

summarizer_lsa = LsaSummarizer(Stemmer("english"))
summarizer_lsa.stop_words = get_stop_words("english")    

sum_path = sys.argv[0] # path to submission folder
path_to_transcript_folders = sys.argv[1] # path to transcript folder

Transcripts = {}
for trans_path in tqdm(path_to_transcript_folders, total=len(path_to_transcript_folders)):
    transcript_files = os.listdir(trans_path)
    for index, transcript_file in tqdm(enumerate(transcript_files), total=len(transcript_files)):
        path_to_transcript = os.path.join(trans_path, transcript_file)
        for file in os.listdir(path_to_transcript):
            path_to_transcript = os.path.join(trans_path, transcript_file)
            if ".txt" in file:
                Transcripts[f"{transcript_file}/{file}"] = PlaintextParser.from_file("{}/{}".format(path_to_transcript,file),Tokenizer("english"))

model_name = "LSA"
if "LSA" not in os.listdir(sum_path):
  os.mkdir("{}/LSA".format(sum_path))

# ["Dev_Transcripts", "testNew_Transcripts", "Test_Transcripts"]
folder_name = "Test"
if folder_name not in os.listdir(os.path.join(sum_path, model_name)):
  os.mkdir(f"{sum_path}/{model_name}/{folder_name}")

for key,value in Transcripts.items():
  summaries = summarizer_lsa(value.document,50)

  final_summary = ""
  for sent in summaries:
    final_summary += str(sent) + "\n"
  transcript_file, file = key.split("/")
  os.mkdir(os.path.join(f"{sum_path}/{model_name}/{folder_name}", transcript_file))
  open("{}/{}/{}/{}/{}".format(sum_path,model_name,folder_name,transcript_file,file),"w",encoding="utf-8").write(final_summary)