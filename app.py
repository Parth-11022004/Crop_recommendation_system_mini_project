import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def train_model():
    df = pd.read_csv('Crop.csv')
    x = df.drop(columns='label')
    y = df['label']
    model = RandomForestClassifier()
    model.fit(x, y)
    return model

model = train_model()
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Streamlit app UI
st.title("🌾 Crop Recommendation System")
st.write("Enter the environmental parameters to get a recommended crop:")

# Create input fields for each feature
user_input = []
for feature in features:
    val = st.number_input(f"{feature}", min_value=0.0, step=0.1)
    user_input.append(val)

# Predict and show result
if st.button("Recommend Crop"):
    prediction = model.predict([user_input])
    st.success(f"Recommended Crop: **{prediction[0]}**")
