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
| Dataset  | Model | Accuracy |
|----------|-------|----------|
| English  | SVM   | 95%      |
|          | NB    | 89%      |
|          | RF    | 93%      |
| Gujarati | SVM   | 89%      |
|          | NB    | 82%      |
|          | RF    | 86%      |
| Hindi    | SVM   | 87%      |
|          | NB    | 80%      |
|          | RF    | 85%      |

---

## Deep Learning Models
### Models Implemented
1. **Convolutional Neural Network (CNN)**
2. **Long Short-Term Memory (LSTM)**
3. **Multi-Layer Perceptron (MLP)**

### Results Summary
| Dataset  | Model | Accuracy |
|----------|-------|----------|
| English  | CNN   | 94%      |
|          | LSTM  | 94%      |
|          | BERT  | 89%      |
| Gujarati | CNN   | 89%      |
|          | LSTM  | 90%      |
|          | BERT  | 89%      |
| Hindi    | CNN   | 88%      |
|          | LSTM  | 88%      |
|          | BERT  | 86%      |

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
