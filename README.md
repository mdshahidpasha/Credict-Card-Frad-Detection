# Credit Card Fraud Detection using Deep Learning

## Project Overview

This project builds a **Credit Card Fraud Detection system** using **Artificial Neural Networks (ANN)**.

The model analyzes transaction features and predicts whether the transaction is **Fraudulent or Legitimate**.

The project also includes a **Streamlit web application** for real-time fraud prediction.

---

## Dataset

Dataset used: **Credit Card Fraud Detection Dataset**

The dataset contains anonymized transaction features:

- V1 to V28 (PCA transformed features)
- Time
- Amount
- Class (Target)

Target variable:
- 0 → Normal Transaction
- 1 → Fraud Transaction

---

## Project Workflow

1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Feature Scaling
4. Handling Imbalanced Data using SMOTE
5. Model Building using Artificial Neural Network
6. Model Evaluation
7. Deployment using Streamlit

---

## Model Architecture

Artificial Neural Network

Input Layer → 29 features

Hidden Layer 1 → 16 neurons (ReLU)

Hidden Layer 2 → 8 neurons (ReLU)

Output Layer → 1 neuron (Sigmoid)

Loss Function:
Binary Crossentropy

Optimizer:
Adam

---

## Model Evaluation

Metrics used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

---

## Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit

---

## Project Structure
