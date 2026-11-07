import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

st.set_page_config(page_title="AIML Classifier Evaluation", page_icon="🧪", layout="wide")
st.title("Machine Learning Classifier Evaluation")
st.write(
    "Upload a labelled CSV test file containing the 30 input features and a "
    "`target` column. The app loads saved models; it does not retrain them."
)
st.caption("Dataset: Wisconsin Diagnostic Breast Cancer (WDBC) | Target: 0 = malignant, 1 = benign")

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
if uploaded is None:
    st.info("Upload the included test_data.csv file to view evaluation results.")
    st.stop()

data = pd.read_csv(uploaded)
model_choice = st.selectbox("Select classifier", list(MODEL_FILES.keys()))

if "target" not in data.columns:
    st.error("The uploaded CSV must contain the target column.")
    st.stop()

try:
    model = joblib.load(MODEL_DIR / MODEL_FILES[model_choice])
    with open(MODEL_DIR / "metrics.json", "r") as f:
        saved_metrics = json.load(f)
except Exception as exc:
    st.error(f"Could not load saved model files: {exc}")
    st.stop()

expected_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
X = data.drop(columns=["target"])
y = data["target"]

if expected_features is not None:
    missing = sorted(set(expected_features) - set(X.columns))
    extra = sorted(set(X.columns) - set(expected_features))
    if missing:
        st.error(f"Missing required feature columns: {missing}")
        st.stop()
    if extra:
        st.warning(f"Extra columns will be ignored: {extra}")
    X = X[expected_features]

try:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    scores = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score(y, probabilities),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1 Score": f1_score(y, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y, predictions)
    }
except Exception as exc:
    st.error(f"Could not evaluate this file. Check its columns and target values. Details: {exc}")
    st.stop()

st.subheader(f"Results: {model_choice}")
cols = st.columns(3)
for col, (label, value) in zip(cols * 2, scores.items()):
    col.metric(label, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, predictions, labels=[0, 1])
    st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"],
                              columns=["Predicted 0", "Predicted 1"]))
with right:
    st.subheader("Classification Report")
    report = classification_report(
        y, predictions, labels=[0, 1],
        target_names=["Malignant (0)", "Benign (1)"],
        output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(4))

with st.expander("Show uploaded test data"):
    st.dataframe(data)
