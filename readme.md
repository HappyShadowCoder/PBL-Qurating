# PBL-QuRating Reproduction

This project reproduces the QuRater preference learning algorithm from the QuRating (2024/2025) paper using the official QuRating-GPT3.5-Judgments dataset.

## What is implemented
- Bradley–Terry pairwise preference learning objective
- DistilBERT-based QuRater model
- Training on official QuRating judgments (educational_value criterion)
- Evaluation on official domain-wise test sets
- Downstream application: scoring unseen text corpus

## Setup

```bash
pip install -r requirements.txt