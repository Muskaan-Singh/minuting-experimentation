from transformers import pipeline
import os
from utils import segment_text

sum_path = "../../submissions"
trans_path = "../../Transcript"

summarizer = pipeline("summarization",device=0)


if "Bart" not in os.listdir(sum_path):
  os.mkdir("{}/Bart".format(sum_path))

for id,file in enumerate(os.listdir(trans_path)):	
	if ".txt" in file:
		transcript = open("{}/{}".format(trans_path,file),"r",encoding="utf-8").read()
		print("{}files".format(id+1))
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
		open("{}/Bart/{}".format(sum_path,file),"w",encoding="utf-8").write(final_summary)