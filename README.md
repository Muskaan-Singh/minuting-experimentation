## Installation

```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Abstractive Summarization
* List of abstractive summarization modules 
    * BART.py
    * Bert2Bert.py
    * Pegasus.py
    * Roberta2Roberta.py
    * T5.py
    * LED.py

```
python Abstractive/module.py submissions/ transcript_folder/
e.g. python Abstractive/model.py submissions/ Dev_Transcripts/
```
* The outputs file should be committed directly into the current directory

## Extractive Summarization
* List of extractive summarization modules
    * LexRank.py
    * LSA.py
    * Luhn.py
    * TextRank.py
    * Unsupervised_Trad_NLP_heuristic_approach.py

```
python Extractive/module.py submissions/ transcript_folder/
e.g. python Extractive/LSA.py submissions/ Dev_Transcripts/
```

## Evaluation
```
python3 automatic-evaluation/eval.py ../submissions/
```
* This directory contains the scripts for automatic scoring of all the submissions in ../submissions/.
* The outputs of the scoring should be also committed directly in this repo.

ROUGE would be the primary metric for automatic evaluation. However, given the abstractive nature of some minutes, we would also use two recent semantic-level metrics: BERTScore and Sentence Mover Score.

