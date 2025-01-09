# News Classification for Indian Languages Based on Headlines

## Overview
This project focuses on classifying news headlines in three Indian languages: English, Hindi, and Gujarati. Utilizing machine learning and deep learning approaches, the aim is to classify news headlines into predefined categories: Business, Entertainment, Politics, Sports, and Technology. The dataset includes multilingual news headlines and demonstrates the effectiveness of various models in multilingual text classification.

---

## Dataset
The dataset consists of three subsets:
- **English Dataset**: 37,395 headlines.
- **Gujarati Dataset**: 37,500 headlines.
- **Hindi Dataset**: 37,779 headlines.

Each headline is categorized into one of the following classes:
- Business
- Entertainment
- Politics
- Sports
- Technology

Datasets are stored in the repository as:
- `english-train`
- `gujarati-train`
- `hindi-train`

---

## Machine Learning Models
### Models Implemented
1. **Support Vector Machines (SVM)**
2. **Naïve Bayes (NB)**
3. **Random Forest (RF)**

### Results Summary
| Dataset  | Model  | Precision | Recall | F1-Score | Accuracy |
|----------|--------|-----------|--------|----------|----------|
| English  | SVM    | 94.55%    | 94.23% | 94.39%   | 95.04%   |
|          | NB     | 88.12%    | 87.56% | 87.84%   | 89.33%   |
|          | RF     | 92.32%    | 92.04% | 92.18%   | 93.21%   |
| Gujarati | SVM    | 87.24%    | 86.92% | 87.08%   | 88.91%   |
|          | NB     | 81.57%    | 81.23% | 81.40%   | 82.34%   |
|          | RF     | 84.61%    | 84.33% | 84.47%   | 85.78%   |
| Hindi    | SVM    | 85.78%    | 85.32% | 85.55%   | 87.21%   |
|          | NB     | 79.34%    | 79.11% | 79.22%   | 80.45%   |
|          | RF     | 83.12%    | 82.75% | 82.93%   | 84.67%   |

---

## Deep Learning Models
### Models Implemented
1. **Convolutional Neural Network (CNN)**
2. **Long Short-Term Memory (LSTM)**
3. **Multi-Layer Perceptron (MLP)**

### Results Summary
| Dataset  | Model  | Precision | Recall | F1-Score | Accuracy |
|----------|--------|-----------|--------|----------|----------|
| English  | CNN    | 90.12%    | 89.83% | 89.97%   | 91.45%   |
|          | LSTM   | 93.84%    | 93.52% | 93.68%   | 93.63%   |
| Gujarati | LSTM   | 89.73%    | 89.41% | 89.57%   | 89.62%   |
| Hindi    | CNN    | 87.89%    | 87.33% | 87.60%   | 87.66%   |

---

## Methodology
1. **Data Preprocessing**:
   - Tokenization
   - Stopword Removal
   - Lemmatization (for English)
   - Case Normalization (for English)
   - Noise Removal
2. **Feature Extraction**:
   - TF-IDF
   - Word Embeddings (for deep learning models)
3. **Model Training**:
   - Each model was trained on the respective dataset and optimized for classification accuracy.

---

## Conclusion
This project demonstrates the potential of machine learning and deep learning models for multilingual text classification. Future work can explore incorporating more Indian languages and fine-tuning state-of-the-art transformer models for improved accuracy.
