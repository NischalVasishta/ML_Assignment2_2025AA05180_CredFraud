import pandas as pd
import numpy as np
import pickle
import warnings

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
    f1_score, matthews_corrcoef
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

df = pd.read_csv("creditcard.csv")

if "Class" not in df.columns:
    raise ValueError("Expected target column 'Class' in creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

pos = int(y_train.sum())
neg = int((y_train == 0).sum())
scale_pos_weight = (neg / max(pos, 1))

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE, class_weight="balanced", max_depth=10
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced", max_depth=12, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        max_depth=6,
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    ),
}

def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

results = {}

for name, model in models.items():
    print(f"Training: {name}")
    model.fit(X_train_s, y_train)

    fname = name.replace(" ", "_").lower() + ".pkl"
    with open(fname, "wb") as f:
        pickle.dump(model, f)

    y_pred = model.predict(X_test_s)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_s)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    results[name] = compute_metrics(y_test, y_pred, y_proba)

results_df = pd.DataFrame(results).T
results_df = results_df[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
results_df.to_csv("model_results.csv", index=True)

print("\nSaved model_results.csv")
print(results_df.round(4))

test_n = 5000
test_df = pd.DataFrame(X_test_s[:test_n], columns=X.columns)
test_df["Class"] = y_test.iloc[:test_n].to_numpy()
test_df.to_csv("test_data.csv", index=False)

print("\nSaved test_data.csv (for Streamlit upload)")
print("Done.")
