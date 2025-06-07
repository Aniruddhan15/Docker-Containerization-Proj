# app.py
import streamlit as st
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and test data
with open("model.pkl", "rb") as f:
    model, X_test, y_test = pickle.load(f)

# App title
st.title("Iris Flower Predictor 🌸")
st.write("Enter the flower dimensions to predict the Iris species.")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
classes = ["Setosa", "Versicolor", "Virginica"]

if st.button("Predict"):
    st.success(f"**Prediction:** {classes[prediction]}")

    # Accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.metric(label="Model Accuracy", value=f"{acc * 100:.2f}%")

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=classes)
    st.text(report)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
