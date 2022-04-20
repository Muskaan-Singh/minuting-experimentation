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

from sumy.summarizers.luhn import LuhnSummarizer

if "Luhn" not in os.listdir(sum_path):
  os.mkdir("{}/Luhn".format(sum_path))

luhn_summarizer = LuhnSummarizer()
for key,value in Transcripts.items():
  summaries = (luhn_summarizer(value.document,50))

  final_summary = ""
  for sent in summaries:
    final_summary += str(sent) + "\n"
  open("{}/Luhn/{}".format(sum_path,key),"w",encoding="utf-8").write(final_summary)
