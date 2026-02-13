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


st.set_page_config(page_title="Credit Card Fraud - Model Demo", layout="wide")
st.title("Credit Card Fraud Detection (6-Model Comparison)")

# -------------------------
# Sidebar: download first
# -------------------------
st.sidebar.header("Upload + Settings")

@st.cache_data
def file_to_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

st.sidebar.subheader("1) Download sample CSV (input format)")
try:
    sample_bytes = file_to_bytes("test_data.csv")
    st.sidebar.download_button(
        label="Download test_data.csv",
        data=sample_bytes,
        file_name="test_data.csv",
        mime="text/csv",
        key="download-test-csv",
        help="Download a sample file in the correct format (includes 'Class'), then upload it below.",
    )
    st.sidebar.caption("New users: download this, then upload it.")
except FileNotFoundError:
    st.sidebar.warning("test_data.csv not found in repo. Add it to enable downloads.")

# -------------------------
# Sidebar: inputs
# -------------------------
st.sidebar.subheader("2) Upload CSV (input)")
uploaded = st.sidebar.file_uploader("Upload CSV (must include 'Class')", type=["csv"])
test_size = st.sidebar.slider("Test split (%)", 10, 40, 20)
model_name = st.sidebar.selectbox(
    "Choose model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

st.write(
    "Upload a small CSV (recommended: a subset of the dataset). "
    "The app will train the selected model and evaluate on a hold-out test split."
)

if uploaded is None:
    st.info("Step 1: Download the sample CSV (optional). Step 2: Upload a CSV to continue.")
    st.stop()  # Anything after this won't run if no upload [web:99]

df = pd.read_csv(uploaded)

if "Class" not in df.columns:
    st.error("CSV must contain target column named 'Class' (0=legit, 1=fraud).")
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head(), use_container_width=True)

X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=(test_size / 100.0), random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

pos = int(y_train.sum())
neg = int((y_train == 0).sum())
scale_pos_weight = (neg / max(pos, 1))

if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=10)
elif model_name == "KNN":
    model = KNeighborsClassifier(n_neighbors=5)
elif model_name == "Naive Bayes":
    model = GaussianNB()
elif model_name == "Random Forest":
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced", max_depth=12, n_jobs=-1
    )
else:
    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        max_depth=6,
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    )

model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_test_s)[:, 1]
else:
    y_proba = y_pred.astype(float)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

st.subheader(f"Metrics ({model_name})")
c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{acc:.4f}")
c1.metric("AUC", f"{auc:.4f}")
c2.metric("Precision", f"{prec:.4f}")
c2.metric("Recall", f"{rec:.4f}")
c3.metric("F1", f"{f1:.4f}")
c3.metric("MCC", f"{mcc:.4f}")

left, right = st.columns(2)

with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
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
