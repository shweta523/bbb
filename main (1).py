import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load sample diabetes dataset
# -----------------------------
data = {
    'Glucose': [148, 85, 183, 89, 137, 116, 78, 115],
    'BloodPressure': [72, 66, 64, 66, 40, 74, 50, 70],
    'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3],
    'Age': [50, 31, 32, 21, 33, 30, 26, 29],
    'Outcome': [1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[['Glucose', 'BloodPressure', 'BMI', 'Age']]
y = df['Outcome']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# Streamlit App
# -----------------------------
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict diabetes")

# User Inputs
glucose = st.number_input("Glucose Level", 0, 300, 120)
bp = st.number_input("Blood Pressure", 0, 200, 70)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
age = st.number_input("Age", 1, 120, 30)

# Prediction
if st.button("Predict"):
    input_data = np.array([[glucose, bp, bmi, age]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ The person is likely to have diabetes")
    else:
        st.success("✅ The person is not likely to have diabetes")
