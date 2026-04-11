import streamlit as st
import pandas as pd
import joblib
import requests
from streamlit_lottie import st_lottie

# Function to load the animation from a URL
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load a cool tech/AI animation
lottie_url = "https://lottie.host/8b4566c1-a2ef-42d3-9599-28c0cc8203f5/5s9E1L1v2r.json"
lottie_anim = load_lottie_url(lottie_url)

# Display the animation on the page
if lottie_anim:
    st_lottie(lottie_anim, height=200, key="coding")

# 1. Load the model and the expected feature columns

model = joblib.load("l1_model.pkl") 
model_features = joblib.load("model_features.pkl")

st.title("Employee Attrition Prediction")

# 2. Collect user inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
department = st.selectbox("Department", ["HR", "Sales", "IT"])

if st.button("Predict Attrition"):
    # 3. Put inputs into a DataFrame
    user_input = pd.DataFrame({
        "Age": [age],
        "Department": [department]
    })

    # 4. Format data (One-Hot Encoding + Reindexing)
    user_data_encoded = pd.get_dummies(user_input).reindex(columns=model_features, fill_value=0)

    # 5. Predict and display
    prediction = model.predict(user_data_encoded)

    if prediction[0] == 1:
        st.error("Warning: This employee is at high risk of leaving")
    else: 
        st.success("This Employee is likely to stay with the company")

