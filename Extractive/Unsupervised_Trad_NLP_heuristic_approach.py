# Unsupervised_Trad_NLP_heuristic_approach.py
import os
import nltk
from nltk.tag import pos_tag
nltk.download("stopwords")
nltk.download("punkt")
nltk.download('averaged_perceptron_tagger')
import math
import sys
from tqdm import tqdm
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

sum_path = sys.argv[0] # path to submission folder
path_to_transcript_folders = sys.argv[1] # path to transcript folder

def pred(text):
  flag = 0 
  # to form sentences from the raw text. on every new line adding the sentence to list.
  sent_tokens = []
  sentence = ""
  for i in range(len(text)):
    if text[i] == ">":
      flag = 0
      continue
    
    if flag == 0:
      if text[i] == '\n':
        sent_tokens.append(sentence)
        sentence = ""
      elif text[i] == "<":
        flag = 1
      else:
        sentence += text[i]


  word_tokens = nltk.word_tokenize(text)
  word_tokens_lower=[word.lower() for word in word_tokens]
  stopWords = list(set(stopwords.words("english")))
  word_tokens_refined=[x for x in word_tokens_lower if x not in stopWords]
  # print(len(word_tokens_refined))

  stem = []
  ps = PorterStemmer()
  for w in word_tokens_refined:
      stem.append(ps.stem(w))
  word_tokens_refined=stem

  # ......................part 1 (cue phrases).................
  QPhrases=["incidentally", "example", "anyway", "furthermore","according"
              "first", "second", "then", "now", "thus", "moreover", "therefore", "hence", "lastly", "finally", "summary"]

  cue_phrases={}
  for sentence in sent_tokens:
      cue_phrases[sentence] = 0
      word_tokens = nltk.word_tokenize(sentence)
      for word in word_tokens:
          if word.lower() in QPhrases:
              cue_phrases[sentence] += 1
  maximum_frequency = max(cue_phrases.values())
  for k in cue_phrases.keys():
      try:
          cue_phrases[k] = cue_phrases[k] / maximum_frequency
          cue_phrases[k]=round(cue_phrases[k],3)
      except ZeroDivisionError:
          x=0
  # print(cue_phrases.values())

  # .......................part2 (numerical data)...................
  numeric_data={}
  for sentence in sent_tokens:
      numeric_data[sentence] = 0
      word_tokens = nltk.word_tokenize(sentence)
      for k in word_tokens:
          if k.isdigit():
              numeric_data[sentence] += 1
  maximum_frequency = max(numeric_data.values())
  for k in numeric_data.keys():
      try:
          numeric_data[k] = (numeric_data[k]/maximum_frequency)
          numeric_data[k] = round(numeric_data[k], 3)
      except ZeroDivisionError:
          x=0
  # print(numeric_data.values())

  #....................part3(sentence length)........................
  sent_len_score={}
  for sentence in sent_tokens:
      sent_len_score[sentence] = 0
      word_tokens = nltk.word_tokenize(sentence)
      if len(word_tokens) in range(0,10):
          sent_len_score[sentence]=1-0.05*(10-len(word_tokens))
      elif len(word_tokens) in range(7,20):
          sent_len_score[sentence]=1
      else:
          sent_len_score[sentence]=1-(0.05)*(len(word_tokens)-40)
  for k in sent_len_score.keys():
      sent_len_score[k]=round(sent_len_score[k],4)
  # print(sent_len_score.values())


  #Create a frequency table to compute the frequency of each word.
  freqTable = {}
  for word in word_tokens_refined:    
      if word in freqTable:         
          freqTable[word] += 1    
      else:         
          freqTable[word] = 1
  for k in freqTable.keys():
      freqTable[k]= math.log10(1+freqTable[k])

  #Compute word frequnecy score of each sentence
  word_frequency={}
  for sentence in sent_tokens:
      word_frequency[sentence]=0
      e=nltk.word_tokenize(sentence)
      f=[]
      for word in e:
          f.append(ps.stem(word))
      for word,freq in freqTable.items():
          if word in f:
              word_frequency[sentence]+=freq
  maximum=max(word_frequency.values())
  for key in word_frequency.keys():
      try:
          word_frequency[key]=word_frequency[key]/maximum
          word_frequency[key]=round(word_frequency[key],3)
      except ZeroDivisionError:
          x=0
  # print(word_frequency.values())

  #......................... part7 (number of proper noun)...................
  proper_noun={}
  for sentence in sent_tokens:
      tagged_sent = pos_tag(sentence.split())
      propernouns = [word for word, pos in tagged_sent if pos == 'NNP']
      proper_noun[sentence]=len(propernouns)
  maximum_frequency = max(proper_noun.values())
  for k in proper_noun.keys():
      try:
          proper_noun[k] = (proper_noun[k]/maximum_frequency)
          proper_noun[k] = round(proper_noun[k], 3)
      except ZeroDivisionError:
          x=0
  # print(proper_noun.values())

  total_score={}
  for k in cue_phrases.keys():
      total_score[k]=cue_phrases[k]+numeric_data[k]+sent_len_score[k]+word_frequency[k]+proper_noun[k]
  # print(total_score.values())

  sumValues = 0
  for sentence in total_score: 
      sumValues += total_score[sentence] 
  average = sumValues / len(total_score)
  # print(average)
  # Storing sentences into our summary. 
  summary = '' 
  for sentence in sent_tokens:
      if (sentence in total_score) and (total_score[sentence] > (1.8*average)): 
          summary += " " + sentence 
  # print("summary: ",summary)
  return summary

model_name = "Unsupervised"
if "Unsupervised" not in os.listdir(sum_path):
  os.mkdir("{}/Unsupervised".format(sum_path))

for trans_path in tqdm(path_to_transcript_folders, total=len(path_to_transcript_folders)):
    transcript_files = os.listdir(trans_path)
    folder_name = trans_path.split("/")[-1].split('_')[0]
    path_to_model_folder = os.path.join(sum_path, model_name)
    os.mkdir(os.path.join(path_to_model_folder, folder_name))
    path_to_data_folder = os.path.join(path_to_model_folder, folder_name)
    for index, transcript_file in tqdm(enumerate(transcript_files), total=len(transcript_files)):	
        path_to_transcript = os.path.join(trans_path, transcript_file)
        for id, file in enumerate(os.listdir(path_to_transcript)):
            os.mkdir(os.path.join(path_to_data_folder, transcript_file))
            if ".txt" in file:
                transcript = open(os.path.join(path_to_transcript, file),"r",encoding="utf-8").read()
                # print("{} files".format(id+1))

                final_summary = pred(transcript).replace(". ",".\n")
                open("{}/{}/{}/{}/{}".format(sum_path,model_name,folder_name,transcript_file,file),"w",encoding="utf-8").write(final_summary)