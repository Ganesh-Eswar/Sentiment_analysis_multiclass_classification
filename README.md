# Movie Review Sentiment Analysis

Kaggle Competetion link: https://www.kaggle.com/competitions/mlp-term-3-2025-kaggle-assignment-3/overview

A Natural Language Processing (NLP) and Machine Learning project focused on multi-class sentiment classification of movie reviews.

The objective of this project is to predict whether a movie review is:

* **Negative (0)**
* **Neutral (1)**
* **Positive (2)**

using text preprocessing, TF-IDF vectorization, feature engineering, and ensemble machine learning models.

---

# Problem Statement

A movie review platform wants to automatically analyze user reviews and determine their sentiment category.

The dataset contains:

* Raw movie review text (`phrase`)
* Three engineered numerical features
* Sentiment labels for training data

This project builds a complete NLP classification pipeline for sentiment prediction.

---

# Dataset Description

## Files

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

---

## Features

| Column    | Description                                         |
| --------- | --------------------------------------------------- |
| id        | Unique identifier                                   |
| phrase    | Movie review text                                   |
| feature_1 | Numerical feature                                   |
| feature_2 | Numerical feature                                   |
| feature_3 | Numerical feature                                   |
| sentiment | Target label (0: Negative, 1: Neutral, 2: Positive) |

---

# Project Workflow

## 1. Data Exploration

Performed:

* Dataset inspection
* Missing value analysis
* Duplicate checking
* Descriptive statistics
* Class distribution analysis

---

## 2. Exploratory Data Analysis (EDA)

Visualizations created using Matplotlib and Seaborn:

* Sentiment distribution
* Phrase length distribution
* Feature distributions
* Outlier analysis
* Model comparison plots

---

# Text Preprocessing

Implemented multiple text-cleaning strategies.

## Light Cleaning

* Lowercasing
* Basic character cleanup
* Preserving sentiment punctuation

## Moderate Cleaning

* Stopword removal
* Lemmatization

## Aggressive Cleaning

* Stopword removal
* Stemming
* Strict token filtering

Libraries used:

```python
NLTK
Regex
PorterStemmer
WordNetLemmatizer
```

---

# Feature Engineering

Additional handcrafted text features were created:

| Feature           | Description                   |
| ----------------- | ----------------------------- |
| phrase_len        | Review length                 |
| word_count        | Number of words               |
| avg_word_len      | Average word length           |
| exclamation_count | Number of exclamation marks   |
| question_count    | Number of question marks      |
| caps_count        | Uppercase character count     |
| caps_ratio        | Ratio of uppercase characters |

---

# Feature Transformation

## TF-IDF Vectorization

Used:

```python
TfidfVectorizer
```

### Best Configuration

```python
ngram_range=(1, 2)
max_features=18000
```

This combines:

* Unigrams
* Bigrams

for improved contextual understanding.

---

## Numerical Feature Processing

Applied:

* Median Imputation
* Standard Scaling

using:

```python
SimpleImputer
StandardScaler
```

---

# Machine Learning Models

The following models were implemented and evaluated:

| Model               | Purpose                     |
| ------------------- | --------------------------- |
| Logistic Regression | Baseline text classifier    |
| Linear SVC          | Margin-based classification |
| Random Forest       | Ensemble tree classifier    |
| Voting Classifier   | Final ensemble model        |

---

# Final Ensemble Model

The final model combines:

* Logistic Regression (C=1.5)
* Logistic Regression (C=1.0)
* Linear SVC
* Random Forest

using:

```python
VotingClassifier(voting='hard')
```

---

# Hyperparameter Tuning

Hyperparameter optimization was performed using:

```python
GridSearchCV
```

Tuned models:

* Logistic Regression
* Linear SVC
* Random Forest

Evaluation metric:

* Accuracy Score

---

# Project Structure

```text
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── notebooks/
│   └── sentiment_analysis.ipynb
│
├── outputs/
│   ├── model_comparision.png
|   └── submission.csv
│
├── README.md
└── requirements.txt
```

---

# Key Learning Outcomes

This project demonstrates:

* NLP preprocessing techniques
* TF-IDF vectorization
* Text feature engineering
* Multi-class classification
* Ensemble learning
* Hyperparameter tuning
* Sparse matrix handling
* Model evaluation and comparison

---

# Future Improvements

Possible future enhancements:

* Deep Learning models (LSTM/BiLSTM)
* Transformer-based models (BERT)
* Word embeddings
* Advanced stacking ensembles
* Cross-validation optimization

---

# Author

**Ganesh Eswar**

Machine Learning & AI Enthusiast
Interested in NLP, Deep Learning, and Applied Machine Learning Projects.
