import streamlit as st
import numpy as np
import joblib

model = joblib.load(r"D:\Shubham\Python ML\Credit Card Fraud Detection\best_model.pkl")

# Title
st.title("Credit Card Fraud Detection")

# Subtitle
st.write("Enter transaction details below:")

# Input
amount = st.number_input("Transaction Amount", min_value=0, step=100)
time = st.number_input("Transaction Time (seconds)", min_value=0)

v_features = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0)
    v_features.append(val)

# Change 1D vector to 2D vector 
features = np.array([time, amount] + v_features).reshape(1, -1)

# Prediction
if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")
