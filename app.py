import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
 
# Load model and scaler
model = tf.keras.models.load_model("fraud_model.keras", compile=False)
scaler = joblib.load("scaler.pkl")
 
# Page config
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="centered")
 
# Title
st.title("💳 Credit Card Fraud Detection")
st.markdown("Detects whether a credit card transaction is **Fraudulent or Legitimate** using a Deep Learning (ANN) model.")
 
# Sidebar
st.sidebar.header("About Project")
st.sidebar.markdown("""
**Model:** Artificial Neural Network (ANN)
 
**Dataset:** Credit Card Fraud Detection
 
**Techniques Used:**
- SMOTE (for class balancing)
- StandardScaler (for Amount)
- Dropout Regularisation
 
**Tech Stack:**
- Python
- TensorFlow / Keras
- Scikit-Learn
- Streamlit
 
**Developer:** Md Shahid Pasha
""")
 
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Note:** V1–V28 features are PCA-transformed 
hidden features from the original dataset 
(not available to end users for privacy).
""")
 
# ── Mode selector ──────────────────────────────────────────
mode = st.radio("Choose input mode:", 
    ["Enter Transaction Amount", "Use Sample Transaction"],
    horizontal=True)
 
st.markdown("---")
 
# ── Sample transactions (real typical PCA patterns) ────────
# Legitimate: all PCA features near 0 (normal behaviour)
# Fraudulent: PCA features with known fraud-like pattern
LEGIT_FEATURES  = [0.0] * 28   # neutral PCA — typical legit
FRAUD_FEATURES  = [                 # typical fraud PCA pattern
    -2.3, 1.95, -1.61, 3.98, -0.52,
    -1.43, -2.55, -0.17,  0.46, -0.82,
    -0.78,  0.50,  0.25,  0.04,  0.40,
    -0.10,  0.18, -0.24, -0.97,  0.22,
    -0.13,  0.18,  0.07, -0.07, -0.40,
     0.15, -0.06,  0.02
]
 
if mode == "Enter Transaction Amount":
    st.subheader("Enter Transaction Details")
    amount = st.number_input("Transaction Amount (₹ / $)", min_value=0.0, step=0.01, value=100.0)
    pca_features = np.zeros((1, 28))  # neutral PCA features
 
    if st.button("Predict", type="primary"):
        amount_scaled = scaler.transform([[amount]])[0][0]
        input_data = np.hstack((pca_features, [[amount_scaled]]))
        prediction = model.predict(input_data, verbose=0)[0][0]
 
        st.markdown("---")
        col1, col2 = st.columns(2)
        col1.metric("Fraud Probability", f"{round(prediction * 100, 2)}%")
        col2.metric("Transaction Amount", f"{amount}")
 
        if prediction > 0.3:
            st.error("⚠️ Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")
 
        st.info("Note: Since V1–V28 are private PCA features, neutral values (0) are used. Use 'Sample Transaction' mode for a full demo.")
 
else:
    st.subheader("Select a Sample Transaction")
    sample_type = st.selectbox("Transaction Type", ["Legitimate Transaction", "Fraudulent Transaction"])
 
    amount = st.number_input("Transaction Amount (₹ / $)", min_value=0.0, step=0.01,
        value=10.0 if sample_type == "Fraudulent Transaction" else 150.0)
 
    if sample_type == "Legitimate Transaction":
        pca_features = np.array([LEGIT_FEATURES])
        st.success("Sample: A typical normal transaction with neutral PCA features.")
    else:
        pca_features = np.array([FRAUD_FEATURES])
        st.warning("Sample: A transaction with fraud-like PCA feature patterns.")
 
    if st.button("Predict", type="primary"):
        amount_scaled = scaler.transform([[amount]])[0][0]
        input_data = np.hstack((pca_features, [[amount_scaled]]))
        prediction = model.predict(input_data, verbose=0)[0][0]
 
        st.markdown("---")
        col1, col2 = st.columns(2)
        col1.metric("Fraud Probability", f"{round(prediction * 100, 2)}%")
        col2.metric("Transaction Amount", f"{amount}")
 
        if prediction > 0.3:
            st.error("⚠️ Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")
 
st.markdown("---")
st.caption("Developed by Md Shahid Pasha | ANN Model | Credit Card Fraud Detection")