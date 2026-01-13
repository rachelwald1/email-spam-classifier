# Email Spam Classifier

A small supervised machine-learning project that learns from labelled email text to estimate the probability that a new, unseen message is spam.

## Overview
This project treats spam detection as a modelling problem rather than a rules-based filter. Raw text emails are converted into numerical features, used to train a probabilistic classifier, and evaluated on unseen data.

## Features
- Supervised training on labelled spam / not-spam data
- Text preprocessing and numerical feature construction
- Probabilistic spam predictions for new emails
- Evaluation using confusion matrices and precision/recall metrics

## Tech stack
- Python
- scikit-learn
- pandas

## Quick start

```bash
pip install -e .
python -m spam_classifier.cli --csv data/sample.csv

Email text> free prize claim now
→ SPAM (P(spam)=0.91)

Email text> meeting tomorrow at 10
→ NOT SPAM (P(spam)=0.08)

src/spam_classifier/
  data.py    # Load and validate labelled datasets
  model.py   # TF-IDF + logistic regression model
  cli.py     # Command-line interface
tests/
  test_pipeline.py
data/
  sample.csv