# Fake News Detection System

A Machine Learning based web application that detects whether a given news article is **Real or Fake** using Natural Language Processing (NLP) and a trained classification model (Logistic Regression).

---

## Features

- Detects fake vs real news instantly  
- Uses TF-IDF Vectorization + Logistic Regression  
- Simple and interactive Flask web interface  
- Displays prediction with confidence score  
- Text preprocessing for better accuracy  

---

## Tech Stack

- **Frontend:** HTML, CSS  
- **Backend:** Flask (Python)  
- **Machine Learning:** Scikit-learn  
- **Data Processing:** Pandas, NumPy  
- **Model Storage:** Pickle  

---

## How It Works

1. **Data Loading**
   - Loads fake and real news datasets  
   - Labels them as:
     - 0 → Fake
     - 1 → Real

2. **Text Preprocessing**
   - Lowercasing  
   - Removing URLs, punctuation, HTML tags  
   - Cleaning unwanted characters  

3. **Feature Extraction**
   - Uses TF-IDF Vectorizer to convert text into numerical form  

4. **Model Training**
   - Uses Logistic Regression classifier  
   - Splits data into training and testing sets  

5. **Prediction**
   - User enters news text  
   - Model predicts:
     - Real or Fake  
     - Confidence score  

