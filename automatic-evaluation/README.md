This directory contains the scripts for automatic scoring of all the submissions in ../submissions/.

The outputs of the scoring should be also committed directly in this repo.

ROUGE would be the primary metric for automatic evaluation. However, given the abstractive nature of some minutes, we would also use two recent semantic-level metrics: BERTScore and Sentence Mover Score.

## Installation

```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en
```

## Evaluation

For every submission ``modelname``, run this:

```
source venv/bin/activate
python eval.py modelname
```

