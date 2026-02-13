import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Credit Card Fraud - Model Comparison", layout="wide")

CSS = """
<style>
.main { background: #0b1020; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
html, body, [class*="css"] { color: #e8eefc; }
.card {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.04);
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.title("Credit Card Fraud Detection (6-Model Comparison)")
st.write("Select a pre-trained model, upload unseen test data (CSV), and view evaluation metrics.")

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

@st.cache_resource
def load_assets():
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    models = {}
    for name, fname in MODEL_FILES.items():
        with open(os.path.join(MODEL_DIR, fname), "rb") as f:
            models[name] = pickle.load(f)

    return scaler, models

@st.cache_data
def file_to_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

scaler, models = load_assets()

with st.sidebar:
    st.header("Controls")

    st.subheader("Sample test file")
    if os.path.exists("test_data.csv"):
        st.download_button(
            label="Download test_data.csv",
            data=file_to_bytes("test_data.csv"),
            file_name="test_data.csv",
            mime="text/csv",
        )
        st.caption("Download this sample, then upload it below to test the app.")
    else:
        st.warning("test_data.csv not found. Run train_models.py to generate it.")

    model_name = st.selectbox("Model", list(models.keys()))
    uploaded = st.file_uploader("Upload CSV (must include 'Class')", type=["csv"])
    run_btn = st.button("Evaluate", type="primary")

st.markdown(f"<div class='card'><b>Selected model:</b> {model_name}</div>", unsafe_allow_html=True)

if uploaded is None:
    st.info("Upload a CSV (or download the sample from the sidebar) and click Evaluate.")
    st.stop()

df = pd.read_csv(uploaded)

if "Class" not in df.columns:
    st.error("Uploaded CSV must contain 'Class' column (0=legit, 1=fraud) for evaluation.")
    st.stop()

X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

if run_btn:
    X_s = scaler.transform(X)
    model = models[model_name]

    y_pred = model.predict(X_s)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_s)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    if len(np.unique(y)) < 2:
        auc = None
    else:
        auc = roc_auc_score(y, y_proba)

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }

    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    c1.metric("AUC", "N/A" if metrics["AUC"] is None else f"{metrics['AUC']:.4f}")
    c2.metric("Precision", f"{metrics['Precision']:.4f}")
    c2.metric("Recall", f"{metrics['Recall']:.4f}")
    c3.metric("F1", f"{metrics['F1']:.4f}")
    c3.metric("MCC", f"{metrics['MCC']:.4f}")

    left, right = st.columns(2)

    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(["Legit (0)", "Fraud (1)"])
        ax.set_yticklabels(["Legit (0)", "Fraud (1)"])
        st.pyplot(fig)

    with right:
        st.subheader("Classification report")
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        rep_df = pd.DataFrame(report).transpose()
        st.dataframe(rep_df.style.format("{:.4f}"), use_container_width=True)

else:
    st.info("Click Evaluate to compute metrics.")
