import os
import time
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
TEST_CSV_PATH = os.path.join(BASE_DIR, "test_data.csv")

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
  margin-bottom: 12px;
}

.tag {
  display: inline-block;
  padding: 5px 10px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.85rem;
  border: 1px solid rgba(255,255,255,0.15);
}

.skeleton {
  border-radius: 14px;
  background: linear-gradient(90deg, rgba(255,255,255,0.05) 25%, rgba(255,255,255,0.10) 37%, rgba(255,255,255,0.05) 63%);
  background-size: 400% 100%;
  animation: shimmer 1.2s ease-in-out infinite;
  height: 120px;
  border: 1px solid rgba(255,255,255,0.08);
}
@keyframes shimmer {
  0% { background-position: 100% 0; }
  100% { background-position: 0 0; }
}

div[data-testid="stMetric"] {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  padding: 10px 12px;
  border-radius: 12px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.title("Credit Card Fraud Detection (6-Model Comparison)")
st.write("Upload unseen test data (CSV) and evaluate using pre-trained models.")

MODEL_THEME = {
    "Logistic Regression": {"color": "#7aa2ff"},
    "Decision Tree": {"color": "#ffb86c"},
    "KNN": {"color": "#50fa7b"},
    "Naive Bayes": {"color": "#ff79c6"},
    "Random Forest": {"color": "#f1fa8c"},
    "XGBoost": {"color": "#bd93f9"},
}

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
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing: {scaler_path}")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    models = {}
    for name, fname in MODEL_FILES.items():
        p = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")
        with open(p, "rb") as f:
            models[name] = pickle.load(f)

    return scaler, models

@st.cache_data
def file_to_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

scaler, models = load_assets()

with st.sidebar:
