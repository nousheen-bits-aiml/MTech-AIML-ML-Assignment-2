"""Streamlit dashboard for the Wisconsin Diagnostic Breast Cancer classifiers."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_curve,
)

st.set_page_config(
    page_title="WDBC | Classifier Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
  background: radial-gradient(circle at 8% 2%, rgba(99,102,241,.12), transparent 27%),
              radial-gradient(circle at 95% 8%, rgba(14,165,233,.10), transparent 25%), #f6f8fc;
}
[data-testid="stHeader"] { background: rgba(246,248,252,.85); }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#111827 0%,#172554 100%); }
[data-testid="stSidebar"] * { color: #f8fafc !important; }
.hero {
  padding: 1.55rem 1.8rem; border-radius: 22px; color: white;
  background: linear-gradient(115deg,#172554 0%,#4338ca 52%,#0284c7 100%);
  box-shadow: 0 14px 35px rgba(49,46,129,.18); margin: .3rem 0 1.25rem;
}
.hero-kicker { text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 700; opacity: .82; }
.hero h1 { color: white; font-size: 2.05rem; line-height: 1.2; margin: .35rem 0 .45rem; font-weight: 800; }
.hero p { color: #e0e7ff; margin: 0; font-size: .98rem; }
.section-title { color: #172554; font-size: 1.22rem; font-weight: 800; margin: .8rem 0 .2rem; }
.section-subtitle { color: #64748b; font-size: .88rem; margin-bottom: .8rem; }
.metric-card {
  border: 1px solid rgba(148,163,184,.18); border-radius: 16px; padding: 15px 16px 13px;
  min-height: 112px; background: white; box-shadow: 0 5px 18px rgba(15,23,42,.045);
}
.metric-label { color: #64748b; font-size: .78rem; font-weight: 700; letter-spacing: .035em; text-transform: uppercase; }
.metric-value { color: #172554; font-size: 1.8rem; font-weight: 800; margin-top: 5px; }
.metric-accent { height: 4px; border-radius: 99px; margin-top: 9px; }
.info-strip { background: #eef2ff; border: 1px solid #dbeafe; color: #3730a3; border-radius: 13px; padding: 12px 15px; font-size: .9rem; }
.stButton > button { border-radius: 10px; border: 0; color: white; font-weight: 700; background: linear-gradient(100deg,#4f46e5,#0284c7); }
div[data-testid="stFileUploader"] { background: white; border: 1px dashed #a5b4fc; border-radius: 14px; padding: 8px; }
div[data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors (KNN)": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine (SVM)": "svm.joblib",
}
FEATURE_COLUMNS = list(load_breast_cancer().feature_names)
TARGET_COLUMN = "target"

@st.cache_resource
def load_model(model_path: str):
    """Load an already-trained model. No training occurs in this app."""
    return joblib.load(model_path)

def metric_card(label: str, value, color: str) -> str:
    shown = "N/A" if pd.isna(value) else f"{value:.4f}"
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{shown}</div>
      <div class="metric-accent" style="background:{color};"></div>
    </div>"""

def positive_class_scores(model, features):
    """Get scores for class 1 (benign), for ROC-AUC."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)
        classes = list(getattr(model, "classes_", []))
        return probs[:, classes.index(1)] if 1 in classes else probs[:, -1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        classes = list(getattr(model, "classes_", []))
        if len(classes) == 2 and classes[1] == 0:
            scores = -scores
        return scores
    return None
