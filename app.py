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

@st.cache_data
def load_csv_from_disk(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def compute_metrics(y_true, y_pred, y_proba):
    auc = None if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_proba)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

def evaluate_dataframe(df: pd.DataFrame, scaler, model):
    if "Class" not in df.columns:
        raise ValueError("CSV must contain 'Class' column for evaluation.")

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int).to_numpy()

    X_s = scaler.transform(X)

    y_pred = model.predict(X_s)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_s)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    metrics = compute_metrics(y, y_pred, y_proba)
    return metrics, y, y_pred

scaler, models = load_assets()

if "active_df" not in st.session_state:
    st.session_state.active_df = None
if "active_source" not in st.session_state:
    st.session_state.active_source = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

with st.sidebar:
    st.header("Controls")

    st.subheader("Model")
    model_name = st.selectbox("Select model", list(models.keys()))
    tag_color = MODEL_THEME[model_name]["color"]

    st.markdown(
        f"<div class='card'><span class='tag' style='background:{tag_color}22;color:{tag_color};'>"
        f"{model_name}</span></div>",
        unsafe_allow_html=True,
    )

    st.subheader("Sample test file")
    if os.path.exists(TEST_CSV_PATH):
        st.download_button(
            label="Download test_data.csv",
            data=file_to_bytes(TEST_CSV_PATH),
            file_name="test_data.csv",
            mime="text/csv",
        )
        use_sample = st.button("Use sample (one click)", type="primary")
    else:
        use_sample = False
        st.warning("test_data.csv not found in repo. Run train_models.py to generate it.")

    st.subheader("Upload unseen test CSV")
    uploaded = st.file_uploader("Upload CSV (must include 'Class')", type=["csv"])
    eval_upload = st.button("Evaluate uploaded file", disabled=(uploaded is None), type="primary")

    st.subheader("Options")
    show_preview = st.checkbox("Show data preview", value=True)
    show_animation = st.checkbox("Show loading animation", value=True)

    if st.button("Clear results"):
        st.session_state.active_df = None
        st.session_state.active_source = None
        st.session_state.last_result = None
        st.rerun()

if use_sample:
    st.session_state.active_df = load_csv_from_disk(TEST_CSV_PATH)
    st.session_state.active_source = "Sample (one click)"

if eval_upload and uploaded is not None:
    st.session_state.active_df = pd.read_csv(uploaded)
    st.session_state.active_source = "Uploaded CSV"

df = st.session_state.active_df
source = st.session_state.active_source

if df is None:
    st.info("Choose: Use sample (one click) OR upload a CSV and click Evaluate uploaded file.")
    st.stop()

if "Class" not in df.columns:
    st.error("Your CSV must contain 'Class' column (0=legit, 1=fraud).")
    st.stop()

if show_preview:
    st.subheader(f"Data preview ({source})")
    st.dataframe(df.head(25), use_container_width=True)

status_area = st.empty()

if show_animation:
    status_area.markdown("<div class='card'><b>Status:</b> Preparing evaluation...</div>", unsafe_allow_html=True)
    prog = st.progress(0, text="Loading model and data...")
    time.sleep(0.10)
    prog.progress(35, text="Scaling features...")
    time.sleep(0.10)
    prog.progress(70, text="Running predictions...")
    time.sleep(0.10)

with st.spinner("Evaluating..."):
    try:
        metrics, y_true, y_pred = evaluate_dataframe(df, scaler, models[model_name])
        st.session_state.last_result = (metrics, y_true, y_pred)
    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        st.stop()

if show_animation:
    prog.progress(100, text="Done")
    time.sleep(0.10)
    prog.empty()
    status_area.markdown("<div class='card'><b>Status:</b> Evaluation complete.</div>", unsafe_allow_html=True)

metrics, y_true, y_pred = st.session_state.last_result

st.subheader("Results")
st.markdown(
    f"<div class='card'><b>Source:</b> {source} &nbsp; | &nbsp; "
    f"<b>Rows:</b> {len(df):,} &nbsp; | &nbsp; "
    f"<b>Fraud %:</b> {df['Class'].mean()*100:.3f}%</div>",
    unsafe_allow_html=True,
)

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
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["Legit (0)", "Fraud (1)"])
    ax.set_yticklabels(["Legit (0)", "Fraud (1)"])
    st.pyplot(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(report).transpose()
    st.dataframe(rep_df.style.format("{:.4f}"), use_container_width=True)
