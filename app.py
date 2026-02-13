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


# ----------------------------
# Page + Styling
# ----------------------------
st.set_page_config(page_title="Credit Card Fraud - Model Playground", layout="wide")

CUSTOM_CSS = """
<style>
/* Subtle background + nicer spacing */
.main { background: #0b1020; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* Make most text readable on dark bg */
html, body, [class*="css"]  { color: #e8eefc; }

/* Card-like containers */
.card {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.04);
}

/* Model badge */
.badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.9rem;
  border: 1px solid rgba(255,255,255,0.15);
}

/* Improve dataframe header contrast slightly */
div[data-testid="stDataFrame"] { background: rgba(255,255,255,0.02); }

/* Metric box spacing */
div[data-testid="stMetric"] {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  padding: 10px 12px;
  border-radius: 12px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("Credit Card Fraud Detection (6-Model Comparison)")
st.write(
    "Use the sample dataset in one click, or upload your own test CSV. "
    "The app trains the selected model and evaluates it on a hold-out test split."
)

MODEL_THEME = {
    "Logistic Regression": {"color": "#7aa2ff", "desc": "Linear baseline; good with scaling; can handle imbalance via class_weight."},
    "Decision Tree": {"color": "#ffb86c", "desc": "Non-linear, interpretable; can overfit without constraints."},
    "KNN": {"color": "#50fa7b", "desc": "Distance-based; sensitive to scaling; slower as data grows."},
    "Naive Bayes": {"color": "#ff79c6", "desc": "Fast probabilistic baseline; independence assumption."},
    "Random Forest": {"color": "#f1fa8c", "desc": "Bagging ensemble; robust; strong general-purpose performance."},
    "XGBoost": {"color": "#bd93f9", "desc": "Boosting ensemble; often top performer; supports imbalance weighting."},
}


# ----------------------------
# Utilities
# ----------------------------
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
    # AUC requires both classes present in y_true
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_proba)

def compute_metrics(y_true, y_pred, y_proba):
    auc = safe_auc(y_true, y_proba)
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return out


# ----------------------------
# Sidebar controls (always visible)
# ----------------------------
st.sidebar.header("Quick actions")

# 1) Download sample
st.sidebar.subheader("1) Download sample input")
try:
    sample_bytes = file_to_bytes("test_data.csv")
    st.sidebar.download_button(
        label="Download test_data.csv",
        data=sample_bytes,
        file_name="test_data.csv",
        mime="text/csv",
        help="Download a ready-to-upload sample file (includes 'Class').",
        key="download-test-csv",
    )
except FileNotFoundError:
    st.sidebar.warning("test_data.csv not found in repo. Add it to enable downloads.")

# 2) One-click load sample
st.sidebar.subheader("2) One-click run sample")
use_sample = st.sidebar.button("Load sample and run", type="primary")

# 3) Manual upload
st.sidebar.subheader("3) Or upload your CSV")
uploaded = st.sidebar.file_uploader("Upload CSV (must include 'Class')", type=["csv"])

# 4) Settings
st.sidebar.subheader("Model + split")
model_name = st.sidebar.selectbox(
    "Choose model",
    list(MODEL_THEME.keys())
)
test_size = st.sidebar.slider("Test split (%)", 10, 40, 20)
show_data_preview = st.sidebar.checkbox("Show data preview", value=True)


# ----------------------------
# Data selection logic
# ----------------------------
df = None
data_source = None

if use_sample:
    try:
        df = load_sample_df("test_data.csv")
        data_source = "Sample (one-click)"
        st.sidebar.success("Loaded sample dataset.")
    except FileNotFoundError:
        st.sidebar.error("test_data.csv is missing in the repo.")
        df = None

elif uploaded is not None:
    df = pd.read_csv(uploaded)
    data_source = "Uploaded file"

# Main tabs layout
tab1, tab2, tab3 = st.tabs(["Run model", "Metrics dashboard", "About"])  # Tabs [web:112]

with tab3:
    st.markdown(
        """
<div class="card">
<b>How to use</b><br>
1) Click <i>Load sample and run</i> OR upload your CSV (must include column <code>Class</code>).<br>
2) Pick a model and test split.<br>
3) View metrics + confusion matrix + classification report.
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="card">
<b>Note on imbalanced data</b><br>
Fraud data is highly imbalanced. Accuracy alone can look very high even when the model misses frauds.
Focus on AUC, Recall, F1, and MCC for meaningful comparisons.
</div>
        """,
        unsafe_allow_html=True,
    )

with tab1:
    if df is None:
        st.info("Use the sidebar: either click **Load sample and run** or upload a CSV to start.")
    else:
        if "Class" not in df.columns:
            st.error("Your CSV must contain a target column named 'Class' (0=legit, 1=fraud).")
        else:
            theme = MODEL_THEME[model_name]
            st.markdown(
                f"""
<div class="card">
<span class="badge" style="background: {theme['color']}22; color: {theme['color']};">
{model_name}
</span>
&nbsp;&nbsp;{theme['desc']}
<br><br>
<b>Data source:</b> {data_source} &nbsp; | &nbsp;
<b>Rows:</b> {len(df):,} &nbsp; | &nbsp;
<b>Fraud %:</b> {df['Class'].mean()*100:.3f}%
</div>
                """,
                unsafe_allow_html=True,
            )

            if show_data_preview:
                st.subheader("Data preview")
                st.dataframe(df.head(20), use_container_width=True)

            # Basic checks
            if df.isna().any().any():
                st.warning("Your dataset contains missing values. Consider cleaning them before upload.")

            X = df.drop(columns=["Class"])
            y = df["Class"].astype(int)

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=(test_size / 100.0), random_state=42, stratify=y
            )

            # Scaling
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            pos = int(y_train.sum())
            neg = int((y_train == 0).sum())
            scale_pos_weight = (neg / max(pos, 1))

            model = get_model(model_name, scale_pos_weight)

            run = st.button("Train and evaluate", type="primary")
            if run:
                with st.spinner("Training model and evaluating..."):
                    model.fit(X_train_s, y_train)
                    y_pred = model.predict(X_test_s)

                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X_test_s)[:, 1]
                    else:
                        y_proba = y_pred.astype(float)

                    metrics = compute_metrics(y_test, y_pred, y_proba)

                st.session_state["last_run"] = {
                    "model_name": model_name,
                    "metrics": metrics,
                    "y_test": y_test.to_numpy(),
                    "y_pred": y_pred,
                }
                st.success("Done. Open the Metrics dashboard tab.")

with tab2:
    last = st.session_state.get("last_run", None)
    if last is None:
        st.info("Run a model first in the **Run model** tab.")
    else:
        model_name = last["model_name"]
        metrics = last["metrics"]
        y_test = last["y_test"]
        y_pred = last["y_pred"]
        theme = MODEL_THEME[model_name]

        st.markdown(
            f"""
<div class="card">
<span class="badge" style="background: {theme['color']}22; color: {theme['color']};">
Metrics dashboard: {model_name}
</span>
</div>
            """,
            unsafe_allow_html=True,
        )

        # Metrics row
        c1, c2, c3 = st.columns(3)

        c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        if metrics["AUC"] is None:
            c1.metric("AUC", "N/A")
        else:
            c1.metric("AUC", f"{metrics['AUC']:.4f}")

        c2.metric("Precision", f"{metrics['Precision']:.4f}")
        c2.metric("Recall", f"{metrics['Recall']:.4f}")

        c3.metric("F1", f"{metrics['F1']:.4f}")
        c3.metric("MCC", f"{metrics['MCC']:.4f}")

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
