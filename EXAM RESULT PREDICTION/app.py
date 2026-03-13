import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

st.title("🎓 Exam Result Prediction (Machine Learning Model)")

st.write("""
This app predicts **PASS / FAIL** using a Machine Learning model.
You can upload your dataset or use demo data.
""")

# --------------------------
# 1. Load or Create Dataset
# --------------------------

uploaded_file = st.file_uploader("Upload CSV (columns: study_hours, attendance, exam_score)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using Demo Dataset")
    df = pd.DataFrame({
        "study_hours": [2, 5, 1, 7, 6, 8, 3, 4, 9, 10],
        "attendance":  [60, 80, 55, 90, 85, 95, 70, 75, 98, 99],
        "exam_score":  [40, 60, 30, 75, 70, 85, 55, 65, 90, 95],
    })

    # Create target column
    df["result"] = df["exam_score"].apply(lambda x: 1 if x >= 50 else 0)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# --------------------------
# 2. Train the Model
# --------------------------

if st.button("Train Model"):

    X = df[["study_hours", "attendance"]]
    y = df["result"]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Predictions
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)

    st.success(f"✅ Model Trained Successfully! Accuracy: {acc * 100:.2f}%")

    st.write("📌 Confusion Matrix")
    st.write(confusion_matrix(y_test, preds))

    # Save model
    with open("exam_model.pkl", "wb") as f:
        pickle.dump((model, scaler), f)
    st.success("📁 Model Saved: exam_model.pkl")

# --------------------------
# 3. Prediction Section
# --------------------------

st.subheader("🎯 Predict Student Result")

study_hours = st.slider("Study Hours per Day", 0, 12, 5)
attendance = st.slider("Attendance (%)", 0, 100, 75)

if st.button("Predict Result"):

    try:
        with open("exam_model.pkl", "rb") as f:
            model, scaler = pickle.load(f)

        input_data = np.array([[study_hours, attendance]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.success("🎉 The student is likely to **PASS**")
        else:
            st.error("⚠️ The student is likely to **FAIL**")

    except FileNotFoundError:
        st.error("Please click **Train Model** first to create the model.")

