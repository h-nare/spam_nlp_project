# Spam Detection using NLP and Machine Learning

## Overview
This project is a machine learning-based spam detection system that classifies SMS messages as **spam** or **not spam** using Natural Language Processing (NLP) techniques.

---

## Features
- Text preprocessing (cleaning, tokenization)
- TF-IDF feature extraction
- Logistic Regression model
- Model evaluation using accuracy, precision, recall, F1-score
- Cross-validation for reliable performance estimation

---

## Technologies Used
- Python
- Pandas
- Scikit-learn
- NLP (TF-IDF)

---

## Model Performance
- Accuracy: ~95%
- Cross-validation score: ~95%

---

## Dataset
The dataset used is the SMS Spam Collection dataset (`spam.csv`) containing labeled SMS messages.

Labels:
- `ham` → not spam  
- `spam` → spam message  

---

## How to Run

```bash
# Install dependencies
pip install pandas scikit-learn

# Run project
python3 main.py