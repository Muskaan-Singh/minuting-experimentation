from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

summarizer_lsa = LsaSummarizer(Stemmer("english"))
summarizer_lsa.stop_words = get_stop_words("english")    

# Load Pkgs
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
import os

import nltk
nltk.download('punkt')

sum_path = "../../submissions"
trans_path = "../../Transcript"

Transcripts = {}
for file in os.listdir(trans_path):
  if ".txt" in file:
    Transcripts[file] = PlaintextParser.from_file("{}/{}".format(trans_path,file),Tokenizer("english"))

if "LSA" not in os.listdir(sum_path):
  os.mkdir("{}/LSA".sum_path)

for key,value in Transcripts.items():
  summaries = summarizer_lsa(value.document,50)

  final_summary = ""
  for sent in summaries:
    final_summary += str(sent) + "\n"
  open("{}/LSA/{}".format(sum_path,key),"w",encoding="utf-8").write(final_summary)
