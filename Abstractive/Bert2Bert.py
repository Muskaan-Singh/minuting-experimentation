import os
from utils import segment_text
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sum_path = "../../submissions"
trans_path = "../../Transcript"
  
tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
model = AutoModelForSeq2SeqLM.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail").to("cuda")

if "Bert2Bert" not in os.listdir(sum_path):
  os.mkdir("{}/Bert2Bert".format(sum_path))


for id,file in enumerate(os.listdir(trans_path)):  
  if ".txt" in file:
    transcript = open("{}/{}".format(trans_path,file),"r",encoding="utf-8").read()
    print("{}files".format(id+1))
    segments_refined = segment_text(transcript,250)

    summary = []
    for ind,segment in enumerate(segments_refined):
      print("{}/{} segments".format(ind+1,len(segments_refined)))
      input_ids = tokenizer(segment, return_tensors="pt").input_ids.to("cuda")
      output_ids = model.generate(input_ids)[0]
      summary.append(tokenizer.decode(output_ids, skip_special_tokens=True))

    final_summary = ""
    for i,summ in enumerate(summary):
      text = summ
      final_summary += text

    final_summary = final_summary.replace(". ",".\n")
    open("{}/Bert2Bert/{}".format(sum_path,file),"w",encoding="utf-8").write(final_summary)
