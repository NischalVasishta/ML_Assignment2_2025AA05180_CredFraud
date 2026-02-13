import time
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Fraud Detection - Model Demo", layout="wide")

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

.badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.9rem;
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
st.write("Download the sample CSV or upload your own, choose a model, and run evaluation.")


MODEL_THEME = {
    "Logistic Regression": {"color": "#7aa2ff", "desc": "Linear baseline; good with scaling; class_weight helps imbalance."},
    "Decision Tree": {"color": "#ffb86c", "desc": "Interpretable; can overfit; captures non-linear splits."},
    "KNN": {"color": "#50fa7b", "desc": "Distance-based; sensitive to scaling; slower with more data."},
    "Naive Bayes": {"color": "#ff79c6", "desc": "Fast probabilistic baseline; independence assumption."},
    "Random Forest": {"color": "#f1fa8c", "desc": "Bagging ensemble; robust; strong general performance."},
    "XGBoost": {"color": "#bd93f9", "desc": "Boosting ensemble; strong performance; supports imbalance weighting."},
}


@st.cache_data
def file_to_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


@st.cache_data
def load_sample_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def get_model(model_name: str, scale_pos_weight: float):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=10)
    if model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=5)
    if model_name == "Naive Bayes":
        return GaussianNB()
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced",
            max_depth=14, n_jobs=-1
        )
    return XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        max_depth=6,
        n_estimators=350,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    )


def safe_auc(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_proba)


def run_pipeline(df: pd.DataFrame, model_name: str, test_size_pct: int):
    if "Class" not in df.columns:
        raise ValueError("Missing target column 'Class'.")

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(test_size_pct / 100.0), random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / max(pos, 1))

    model = get_model(model_name, scale_pos_weight)

    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_s)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": safe_auc(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    return metrics, y_test.to_numpy(), y_pred


if "results" not in st.session_state:
    st.session_state.results = None


with st.sidebar:
    st.header("Controls")

    st.subheader("Sample file")
    try:
        sample_bytes = file_to_bytes("test_data.csv")
        st.download_button(
            label="Download test_data.csv",
            data=sample_bytes,
            file_name="test_data.csv",
            mime="text/csv",
            key="download-test-csv",
        )
    except FileNotFoundError:
        st.warning("test_data.csv missing in repo (download disabled).")

    st.subheader("Run")

    mode = st.radio("Data source", ["Sample (one click)", "Upload CSV"], horizontal=True)
    uploaded = st.file_uploader("Upload CSV (must include 'Class')", type=["csv"], disabled=(mode != "Upload CSV"))

    with st.form("run_form", clear_on_submit=False):
        model_name = st.selectbox("Model", list(MODEL_THEME.keys()))
        test_size = st.slider("Test split (%)", 10, 40, 20)
        show_preview = st.checkbox("Show data preview", value=True)
        animate = st.checkbox("Show animations", value=True)
        submitted = st.form_submit_button("Train and evaluate", type="primary")  # colored [web:149]

    if st.button("Clear results"):
        st.session_state.results = None
        st.rerun()


theme = MODEL_THEME[model_name]
st.markdown(
    f"""
<div class="card">
<span class="badge" style="background: {theme['color']}22; color: {theme['color']};">
{model_name}
</span>
&nbsp;&nbsp;{theme['desc']}
</div>
    """,
    unsafe_allow_html=True,
)

result_placeholder = st.container()

if submitted:
    try:
        if mode == "Sample (one click)":
            df = load_sample_df("test_data.csv")
            source = "Sample"
        else:
            if uploaded is None:
                st.error("Upload a CSV first or switch to Sample.")
                st.stop()
            df = pd.read_csv(uploaded)
            source = "Uploaded"

        if "Class" not in df.columns:
            st.error("CSV must contain 'Class' column.")
            st.stop()

        if show_preview:
            st.subheader(f"{source} data preview")
            st.dataframe(df.head(25), use_container_width=True)

        if animate:
            prog = st.progress(0, text="Preparing...")
            time.sleep(0.2)
            prog.progress(20, text="Splitting data...")
            time.sleep(0.2)
            prog.progress(50, text="Training model...")
            time.sleep(0.2)
            prog.progress(80, text="Evaluating metrics...")
            time.sleep(0.2)

        with st.spinner("Running training + evaluation..."):
            metrics, y_test, y_pred = run_pipeline(df, model_name, test_size)

        if animate:
            prog.progress(100, text="Done")
            time.sleep(0.3)
            prog.empty()
            st.balloons()

        st.session_state.results = {
            "source": source,
            "rows": int(len(df)),
            "fraud_pct": float(df["Class"].mean() * 100),
            "model_name": model_name,
            "test_size": test_size,
            "metrics": metrics,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    except Exception as e:
        st.error(f"Run failed: {e}")

res = st.session_state.results

with result_placeholder:
    st.subheader("Results")
    if res is None:
        st.info("Run the model to see results here.")
    else:
        st.markdown(
            f"""
<div class="card">
<b>Source:</b> {res['source']} &nbsp; | &nbsp;
<b>Rows:</b> {res['rows']:,} &nbsp; | &nbsp;
<b>Fraud %:</b> {res['fraud_pct']:.3f}% &nbsp; | &nbsp;
<b>Test split:</b> {res['test_size']}%
</div>
            """,
            unsafe_allow_html=True,
        )

        m = res["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m['Accuracy']:.4f}")
        c1.metric("AUC", "N/A" if m["AUC"] is None else f"{m['AUC']:.4f}")
        c2.metric("Precision", f"{m['Precision']:.4f}")
        c2.metric("Recall", f"{m['Recall']:.4f}")
        c3.metric("F1", f"{m['F1']:.4f}")
        c3.metric("MCC", f"{m['MCC']:.4f}")

        y_test = res["y_test"]
        y_pred = res["y_pred"]

        left, right = st.columns(2)

        with left:
            st.subheader("Confusion matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4.8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_xticklabels(["Legit (0)", "Fraud (1)"])
            ax.set_yticklabels(["Legit (0)", "Fraud (1)"])
            st.pyplot(fig)

        with right:
            st.subheader("Classification report")
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            rep_df = pd.DataFrame(report).transpose()
            st.dataframe(rep_df.style.format("{:.4f}"), use_container_width=True)
