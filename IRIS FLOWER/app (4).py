import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Iris Flower Prediction", layout="wide")

st.title(" Machine Learning Prediction")
st.write("Iris Flower Classification")

tabs = st.tabs(["Iris Flower Prediction"])

# ---------------- IRIS PREDICTION ---------------- #

with tabs[0]:

    st.header("Iris Flower Species Prediction")

    df = pd.read_csv("IRIS.csv")

    X = df.drop("species", axis=1)
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    st.subheader("Enter Flower Measurements")

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
        sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)

    with col2:
        petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
        petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(input_data)

    if st.button("Predict Species"):
        st.success(f"Predicted Species: {prediction[0]}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Feature Correlation")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

