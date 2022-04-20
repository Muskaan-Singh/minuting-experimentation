def segment_text(input,limit):
  count = 0
  segments = []
  text = ""
  sents = input.split("\n")
  print(sents[0])
  for sent in sents:
    words = sent.split(" ")
    if len(words) + count <= limit:
      text += sent + "\n"
      count += len(words)
    else:
      count = 0
      segments.append(text)
      text = ""
  
  if len(text) != 0 and segments[-1] != text:
    segments.append(text)
  
  return segments
