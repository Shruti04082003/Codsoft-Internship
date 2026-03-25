import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("sales_model.pkl", "rb"))

st.title("📊 Sales Prediction App")

st.write("Enter advertising budget to predict sales")

# Inputs
tv = st.number_input("TV Advertising Budget", min_value=0.0)
radio = st.number_input("Radio Advertising Budget", min_value=0.0)
newspaper = st.number_input("Newspaper Advertising Budget", min_value=0.0)

# Predict button
if st.button("Predict Sales"):
    input_data = np.array([[tv, radio, newspaper]])
    prediction = model.predict(input_data)

    st.success(f"Predicted Sales: {prediction[0]:.2f}")