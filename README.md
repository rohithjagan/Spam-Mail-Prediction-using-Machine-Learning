# Spam Mail Prediction using Machine Learning
# Overview
This project focuses on building a machine learning model to detect whether an email message is spam or ham (not spam).
Using Natural Language Processing (NLP) techniques and machine learning algorithms, the model analyzes email text and predicts its category.

# Objective
To design and train a model that can accurately classify emails as spam or ham based on their content.

# Features
Data cleaning and preprocessing of email text
Text vectorization using TF-IDF
Training multiple classifiers:
    Naive Bayes
    Logistic Regression
    Support Vector Machine (SVM)
Model evaluation using performance metrics
High accuracy and reliability

# Dataset
Source: UCI Machine Learning Repository or Kaggle
Description: Contains labeled SMS/email messages as spam or ham

# Technologies Used
Language: Python
Libraries:
    NumPy
    Pandas
    Scikit-learn
    NLTK
    Matplotlib / Seaborn

# Project Workflow
1. Import Libraries
Load all required libraries for data processing, visualization, and model training.

2. Load Dataset
Read and explore the dataset using Pandas.

3. Data Cleaning & Preprocessing
Remove punctuation and special symbols
Convert text to lowercase
Remove stopwords
Apply stemming or lemmatization

4. Feature Extraction
Convert text data to numerical form using TF-IDF Vectorizer.

5. Model Training
Train models like Naive Bayes and Logistic Regression on the transformed dataset.

6. Model Evaluation
Evaluate models using:
    Accuracy
    Confusion Matrix
    Precision
    Recall
    F1 Score

7. Prediction
Test with new messages to check if they are spam or ham.


Results

Achieved 97–99% accuracy (depending on the algorithm and dataset).

Naive Bayes performed best for text classification tasks.
