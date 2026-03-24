import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Title
# -----------------------------
st.title("🎬 IMDb Movie Rating Prediction (India)")

st.write("Predict movie rating based on genre, director, actors, and duration")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("IMDb Movies India.csv", encoding='latin1')
    return df

df = load_data()

# -----------------------------
# Data Cleaning
# -----------------------------
df = df.dropna(subset=['Rating', 'Genre', 'Director', 'Actor 1'])

# Convert Duration (remove 'min')
df['Duration'] = df['Duration'].str.replace(' min', '')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

df = df.dropna(subset=['Duration'])

# -----------------------------
# Encoding
# -----------------------------
le_genre = LabelEncoder()
le_director = LabelEncoder()
le_actor1 = LabelEncoder()

df['Genre'] = le_genre.fit_transform(df['Genre'])
df['Director'] = le_director.fit_transform(df['Director'])
df['Actor 1'] = le_actor1.fit_transform(df['Actor 1'])

# -----------------------------
# Features & Target
# -----------------------------
X = df[['Genre', 'Director', 'Actor 1', 'Duration']]
y = df['Rating']

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -----------------------------
# Model
# -----------------------------
model = RandomForestRegressor()
model.fit(X_train, y_train)

# -----------------------------
# User Input
# -----------------------------
st.header("Enter Movie Details")

genre = st.selectbox("Select Genre", le_genre.classes_)
director = st.selectbox("Select Director", le_director.classes_)
actor = st.selectbox("Select Lead Actor", le_actor1.classes_)
duration = st.slider("Duration (minutes)", 60, 200)

# Encode input
genre_enc = le_genre.transform([genre])[0]
director_enc = le_director.transform([director])[0]
actor_enc = le_actor1.transform([actor])[0]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Rating"):
    input_data = np.array([[genre_enc, director_enc, actor_enc, duration]])
    prediction = model.predict(input_data)

    st.success(f"⭐ Predicted Rating: {prediction[0]:.2f}")