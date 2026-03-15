# 💳 Credit Card Fraud Detection using Deep Learning (ANN)

![Python](https://img.shields.io/badge/Python-3.8-blue?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange?style=flat&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

A deep learning project that detects fraudulent credit card transactions using an **Artificial Neural Network (ANN)**. The model is trained on a real-world dataset with SMOTE balancing and deployed as an interactive **Streamlit web application**.

🔗 **Live App:** [Click here to try the app](https://credict-card-frad-detection-nauscgpmzcfv4bjlonox4h.streamlit.app)

---

## 📌 Project Overview

Credit card fraud is a major financial threat. This project builds a binary classification model that predicts whether a given transaction is **Fraudulent (1)** or **Legitimate (0)** based on anonymised transaction features.

---

## 📂 Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Feature | Description |
|---------|-------------|
| V1 – V28 | PCA-transformed anonymised features |
| Amount | Transaction amount |
| Class | Target variable (0 = Legitimate, 1 = Fraud) |

> **Note:** The `Time` column was dropped during preprocessing as it does not contribute to fraud prediction.

**Class Distribution:**
- Legitimate transactions: 284,315 (99.83%)
- Fraudulent transactions: 492 (0.17%) — highly imbalanced

---

## 🔁 Project Workflow

```
Data Loading → EDA → Preprocessing → Feature Scaling
→ SMOTE Balancing → ANN Model → Evaluation → Deployment
```

1. **Exploratory Data Analysis (EDA)** — distribution plots, correlation heatmap, class imbalance analysis
2. **Data Preprocessing** — dropped `Time` column, handled missing values
3. **Feature Scaling** — `StandardScaler` applied on `Amount`
4. **Handling Imbalanced Data** — SMOTE (Synthetic Minority Over-sampling Technique)
5. **Model Building** — Artificial Neural Network with Dropout regularisation
6. **Model Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
7. **Deployment** — Streamlit app deployed on Streamlit Cloud

---

## 🧠 Model Architecture

```
Input Layer     →  29 features (V1–V28 + Amount)
Hidden Layer 1  →  16 neurons, ReLU activation, Dropout (0.3)
Hidden Layer 2  →   8 neurons, ReLU activation, Dropout (0.3)
Output Layer    →   1 neuron,  Sigmoid activation
```

| Parameter | Value |
|-----------|-------|
| Loss Function | Binary Crossentropy |
| Optimizer | Adam |
| Epochs | 20 |
| Batch Size | 64 |
| Validation Split | 20% |
| Fraud Threshold | 0.3 |

---

## 📊 Model Evaluation

| Metric | Value |
|--------|-------|
| Accuracy | ✅ High |
| Precision | ✅ High |
| Recall | ✅ High |
| F1 Score | ✅ High |
| ROC-AUC Score | ✅ High |

> Confusion matrix and full classification report available in the notebook.

---

## 🗂️ Project Structure

```
credict-card-frad-detection/
├── app.py                          ← Streamlit web application
├── fraud_model.keras               ← Trained ANN model
├── scaler.pkl                      ← Fitted StandardScaler
├── Credict_Card_Fraud_Detection.ipynb  ← Full project notebook
├── requirements.txt                ← Python dependencies
└── README.md                       ← Project documentation
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8 |
| Deep Learning | TensorFlow, Keras |
| ML & Preprocessing | Scikit-learn, imbalanced-learn |
| Data Analysis | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Deployment | Streamlit, Streamlit Cloud |
| Version Control | Git, GitHub |

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/mdshahidpasha/Credict-Card-Frad-Detection
cd Credict-Card-Frad-Detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit app**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## 💡 Key Learnings

- Handling severely imbalanced datasets using SMOTE
- Building and tuning an ANN with Dropout regularisation
- Importance of Recall and F1 over Accuracy for fraud detection
- End-to-end ML project deployment using Streamlit

---

## 👨‍💻 Developer

**Md Shahid Pasha**

[![GitHub](https://img.shields.io/badge/GitHub-mdshahidpasha-black?style=flat&logo=github)](https://github.com/mdshahidpasha)

---

## 📄 License

This project is licensed under the MIT License.