# 🔬 Personalized Cancer Diagnosis Using NLP and ML

## 📌 Project Overview

A cancer tumor can have thousands of genetic mutations, but not all of them contribute to tumor growth. Distinguishing **driver mutations** (those that promote cancer) from **passenger mutations** (those that do not) is essential for personalized treatment. Traditionally, this task is performed manually by clinical pathologists who interpret each mutation by analyzing clinical literature—an extremely time-consuming and expertise-intensive process.

This project leverages **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques to automatically classify genetic mutations based on textual evidence from clinical literature.

---

## 🎯 Problem Statement

> **Objective:** Build a machine learning model that classifies genetic mutations as relevant or not using features derived from:
- **Gene**
- **Variation**
- **Text (scientific literature)**

This model should automate the interpretation process, providing pathologists with faster and scalable assistance in identifying significant mutations.

---

## 📁 Dataset

The dataset consists of three key components:

1. `Gene` – Contains the gene involved in the mutation.
2. `Variation` – Describes the variation in the gene.
3. `Text` – Contains detailed clinical literature excerpts describing the mutations.

Each data point is labeled with a class indicating the relevance of the mutation.

---

## 🛠️ Features & Techniques Used

- **Text Preprocessing:** Tokenization, stopwords removal, TF-IDF, n-gram extraction
- **Structured Feature Engineering:** Gene and Variation encoding
- **Exploratory Data Analysis (EDA):** Class imbalance analysis, text length distributions
- **Dimensionality Reduction:** TruncatedSVD
- **Model Evaluation:** Confusion matrix, precision, recall, log loss

---

## 🧠 Models Implemented

| Model                   | Notes                                                  |
|------------------------|--------------------------------------------------------|
| Naive Bayes            | Good baseline for text classification                  |
| Logistic Regression    | Interpretable and effective on linear data             |
| K-Nearest Neighbors    | Non-parametric, used for experimentation               |
| Linear SVM             | Effective for high-dimensional space                   |
| Random Forest (Bagging)| Handles non-linearities and reduces overfitting        |
| Stacking Classifier    | **Best performer** – Reduced log loss to **0.53**      |

> ✅ **Best Performing Model:**  
> **Stacking Classifier** with **Logistic Regression as meta-model**  
> ⬇️ Log Loss reduced from **2.50** to **0.53**

---

## 📊 Results

| Metric         | Value (Best Model)      |
|----------------|--------------------------|
| Accuracy       | ~84% (varies by class)   |
| Precision      | High for majority classes|
| Recall         | Balanced using ensemble models |
| **Log Loss**   | **0.53**                 |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `nltk`, `xgboost`, etc.

### Installation

```bash
git clone https://github.com/yourusername/personalized-cancer-diagnosis-nlp.git
cd personalized-cancer-diagnosis-nlp
pip install -r requirements.txt
