
# Few-Shot Learning vs Fine-Tuning for Biomedical QA

## Key Findings
<img width="4770" height="3529" alt="comparative_analysis" src="https://github.com/user-attachments/assets/5f18ed9d-b2ea-4fe3-b607-f33cbbd37d00" />

**Few-shot learning outperformed fine-tuning by 32.5% on biomedical question answering, using 99% fewer examples and 99.9% less training time.**

| Method | Accuracy | Examples | Time | Transfer Domain |
|--------|----------|----------|------|-----------------|
| Few-Shot (5-shot) | **53.0%** | 5 | <1 min | **73.3%** |
| Fine-Tuning | 40.0% | 500 | 12.5 hrs | 46.7% |

## Overview

This project compares few-shot prompting vs fine-tuning on GPT-2 using PubMedQA dataset. Includes cross-domain transfer evaluation on COVID-19 questions.

### Research Questions
- How does few-shot compare to fine-tuning on biomedical QA?
- Which generalizes better to new domains?

## Quick Start
``````bash
# Setup
git clone https://github.com/KUNALSINGH9373/few-shot-learning-biomedical-qa.git
cd few-shot-learning-biomedical-qa
python -m venv few_shot_env
source few_shot_env/bin/activate  # Windows: few_shot_env\Scripts\activate
pip install -r requirements.txt

# Run experiments
python scripts/run_few_shot_experiments.py
python scripts/run_fine_tuning.py
python scripts/comparative_analysis.py
python scripts/cross_domain_transfer.py

