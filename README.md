**Fake News Detection System**
A Machine Learning-based web application that detects whether a given news article is Real or Fake using Natural Language Processing (NLP) and a trained classification model.

**Features**
Detects fake vs real news instantly
Uses TF-IDF Vectorization + Logistic Regression
Simple and interactive Flask web interface
Displays prediction with confidence score
Text preprocessing for better accuracy

**🛠️ Tech Stack**
Frontend: HTML, CSS
Backend: Flask (Python)
Machine Learning: Scikit-learn
Data Processing: Pandas, NumPy
Model Storage: Pickle

**How It Works**
Data Loading
  Loads fake and real news datasets
Labels them as:
  0 → Fake
  1 → Real
Text Preprocessing
  Lowercasing, Removing URLs, punctuation, HTML tags, Cleaning unwanted characters
Feature Extraction
  Uses TF-IDF Vectorizer to convert text into numerical form
Model Training
  Uses Logistic Regression classifier
  Splits data into training and testing sets
Prediction
  User enters news text
  Model predicts:
    Real or Fake
    Confidence score

