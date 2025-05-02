import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load datasets
fake_df = pd.read_csv('fake.csv')
true_df = pd.read_csv('true.csv')

# Add labels
fake_df['label'] = 0
true_df['label'] = 1

# Combine datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Combine title and text
df['content'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
df.dropna(subset=['content'], inplace=True)

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['cleaned_content'] = df['content'].apply(clean_text)

# Feature extraction
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_content'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Evaluation
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n{model_name} Performance:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(f"Precision: {precision_score(y_true, y_pred) * 100:.2f}%")
    print(f"Recall:    {recall_score(y_true, y_pred) * 100:.2f}%")
    print(f"F1 Score:  {f1_score(y_true, y_pred) * 100:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

evaluate_model(y_test, lr_pred, "Logistic Regression")
evaluate_model(y_test, rf_pred, "Random Forest")

# Prediction function
def predict_news(title, text):
    combined = title + ' ' + text
    cleaned = clean_text(combined)
    vectorized = tfidf.transform([cleaned])
    
    lr_result = lr_model.predict(vectorized)[0]
    rf_result = rf_model.predict(vectorized)[0]

    print("\nPrediction Results:")
    print(f"Logistic Regression: {'Real' if lr_result == 1 else 'Fake'}")
    print(f"Random Forest:       {'Real' if rf_result == 1 else 'Fake'}")

# User input
print("\nFake News Detector")
title_input = input("Enter News Title: ")
text_input = input("Enter News Text: ")
predict_news(title_input, text_input)
