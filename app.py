import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model("fraud_model.keras", compile=False)
scaler = joblib.load("scaler.pkl")

# Page configuration
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

# Title
st.title("💳 Credit Card Fraud Detection App")

st.markdown("""
This app predicts whether a credit card transaction is **Fraudulent or Legitimate** using a **Deep Learning (ANN) model**.
""")

# Sidebar
st.sidebar.header("About Project")
st.sidebar.write("""
Model: Artificial Neural Network (ANN)

Dataset: Credit Card Fraud Detection Dataset

Tech Stack:
- Python
- TensorFlow
- Streamlit
- Scikit-Learn
""")

# Input
st.subheader("Enter Transaction Details")

amount = st.number_input("Transaction Amount", min_value=0.0, step=1.0)

# Prediction
if st.button("Predict Fraud"):

    # scale amount
    amount_scaled = scaler.transform([[amount]])

    # generate random PCA features
    random_features = np.random.normal(size=(1,28))

    # combine features
    input_data = np.hstack((random_features, amount_scaled))

    # model prediction
    prediction = model.predict(input_data)[0][0]

    # show probability
    st.write("Fraud Probability:", round(prediction,4))

    # decision
    if prediction > 0.3:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

st.markdown("---")
st.markdown("Developed by **Md Shahid Pasha**")