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

## Project status
In progress