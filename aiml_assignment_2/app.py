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
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#111827 0%,#172554 100%);
  min-width: 270px;
}
[data-testid="stSidebar"] * { color: #f8fafc !important; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] label {
  color: #f8fafc !important;
  font-weight: 700 !important;
}
.model-picker {
  background: #eaf0ff;
  border: 1px solid #c7d2fe;
  border-radius: 14px;
  padding: 12px 16px 4px;
  margin: 0 0 16px;
}
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
div[data-testid="stFileUploader"] {
  background: #ffffff !important; border: 1px dashed #a5b4fc;
  border-radius: 14px; padding: 8px;
}
/* Keep all uploader instructions and controls clearly readable. */
div[data-testid="stFileUploader"] *,
div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] p {
  color: #111827 !important;
  opacity: 1 !important;
}
div[data-testid="stFileUploader"] section {
  background: #ffffff !important;
  border-color: #cbd5e1 !important;
}
div[data-testid="stFileUploader"] button {
  color: #111827 !important;
  background: #f1f5f9 !important;
  border: 1px solid #cbd5e1 !important;
}
div[data-testid="stFileUploader"] button:hover {
  background: #e2e8f0 !important;
  border-color: #94a3b8 !important;
}
[data-testid="stMarkdownContainer"] h4 {
  color: #172554 !important;
  font-weight: 800 !important;
}
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

def sync_model_from_sidebar():
    st.session_state["main_model_picker"] = st.session_state["sidebar_model_picker"]

def sync_model_from_main():
    st.session_state["sidebar_model_picker"] = st.session_state["main_model_picker"]

if "sidebar_model_picker" not in st.session_state:
    st.session_state["sidebar_model_picker"] = list(MODEL_FILES.keys())[0]
if "main_model_picker" not in st.session_state:
    st.session_state["main_model_picker"] = st.session_state["sidebar_model_picker"]

with st.sidebar:
    st.markdown("## 🧬 Classifier Studio")
    st.caption("M.Tech AIML • Machine Learning Assignment 2")
    st.markdown("---")
    st.markdown("### Experiment setup")
    st.selectbox(
        "Choose a trained classifier",
        list(MODEL_FILES.keys()),
        key="sidebar_model_picker",
        on_change=sync_model_from_sidebar,
    )
    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown("**Wisconsin Diagnostic Breast Cancer**")
    st.markdown("569 observations  \n30 numerical features  \nBinary classification")
    st.markdown("---")
    st.caption("Saved models are loaded from `model/`. Uploaded data is used only for evaluation.")

st.markdown("""
<div class="hero">
  <div class="hero-kicker">Machine learning • Model evaluation</div>
  <h1>Breast Cancer Classifier Dashboard</h1>
  <p>Explore five supervised learning models on the Wisconsin Diagnostic Breast Cancer dataset.</p>
</div>
""", unsafe_allow_html=True)

# A second, visible selector keeps model switching possible even if the
# Streamlit sidebar is collapsed in the virtual-machine browser.
st.markdown('<div class="model-picker">', unsafe_allow_html=True)
st.selectbox(
    "Choose classifier (also available in the left sidebar)",
    list(MODEL_FILES.keys()),
    key="main_model_picker",
    on_change=sync_model_from_main,
)
st.markdown('</div>', unsafe_allow_html=True)
selected_model_name = st.session_state["main_model_picker"]

st.markdown('<div class="section-title">01 · Upload evaluation data</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Upload a labelled CSV containing all 30 input features and a <code>target</code> column.</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose a test CSV file", type=["csv"],
    help="The CSV must contain the 30 WDBC features and target (0 = malignant, 1 = benign).",
)

if uploaded_file is None:
    st.markdown('<div class="info-strip">⬆️ Upload the included <b>test_data.csv</b> file to view evaluation results.</div>', unsafe_allow_html=True)
    st.stop()

try:
    test_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the uploaded CSV: {exc}")
    st.stop()

missing_features = [c for c in FEATURE_COLUMNS if c not in test_df.columns]
if missing_features:
    st.error(
        f"Missing {len(missing_features)} required WDBC feature column(s): "
        + ", ".join(missing_features[:8]) + (" ..." if len(missing_features) > 8 else "")
    )
    st.stop()
if TARGET_COLUMN not in test_df.columns:
    st.error("The uploaded CSV must include a `target` column for labelled evaluation.")
    st.stop()
if test_df.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()

try:
    X = test_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(test_df[TARGET_COLUMN], errors="raise").astype(int)
except Exception as exc:
    st.error(f"Please ensure all feature and target values are numeric. Details: {exc}")
    st.stop()

if not set(y.unique()).issubset({0, 1}):
    st.error("The target column must contain only 0 (malignant) and 1 (benign).")
    st.stop()
if y.nunique() < 2:
    st.error("The uploaded test data must contain both target classes (0 and 1) to calculate AUC.")
    st.stop()
if X.isnull().any().any() or y.isnull().any():
    st.error("The uploaded file contains missing values. Please clean the CSV and upload it again.")
    st.stop()

st.markdown(
    f'<div class="info-strip">✓ Loaded <b>{len(test_df):,}</b> labelled records and <b>{len(FEATURE_COLUMNS)}</b> input features.</div>',
    unsafe_allow_html=True,
)
with st.expander("Preview uploaded data"):
    st.dataframe(test_df.head(8), use_container_width=True)

model_path = MODEL_DIR / MODEL_FILES[selected_model_name]
if not model_path.exists():
    st.error(f"Model file not found: `model/{MODEL_FILES[selected_model_name]}`. Upload the `model/` folder with app.py.")
    st.stop()

try:
    model = load_model(str(model_path))
    predictions = model.predict(X)
except Exception as exc:
    st.error(f"Could not load or run the selected model: {exc}")
    st.stop()

try:
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, pos_label=1, zero_division=0)
    recall = recall_score(y, predictions, pos_label=1, zero_division=0)
    f1 = f1_score(y, predictions, pos_label=1, zero_division=0)
    mcc = matthews_corrcoef(y, predictions)
    scores = positive_class_scores(model, X)
    if scores is not None:
        fpr, tpr, _ = roc_curve(y, scores, pos_label=1)
        auc_value = auc(fpr, tpr)
    else:
        auc_value = float("nan")
except Exception as exc:
    st.error(f"Evaluation failed: {exc}")
    st.stop()

st.markdown(f'<div class="section-title">02 · Results — {selected_model_name}</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Metrics calculated on the uploaded labelled test set.</div>', unsafe_allow_html=True)

metric_items = [
    ("Accuracy", accuracy, "#4f46e5"),
    ("AUC", auc_value, "#0ea5e9"),
    ("Precision", precision, "#8b5cf6"),
    ("Recall", recall, "#06b6d4"),
    ("F1 Score", f1, "#7c3aed"),
    ("MCC", mcc, "#0f766e"),
]
for start in (0, 3):
    cols = st.columns(3)
    for col, item in zip(cols, metric_items[start:start + 3]):
        with col:
            st.markdown(metric_card(*item), unsafe_allow_html=True)

st.markdown('<div class="section-title">03 · Detailed evaluation</div>', unsafe_allow_html=True)
cm = confusion_matrix(y, predictions, labels=[0, 1])

left, right = st.columns(2, gap="large")

with left:
    st.markdown("#### Confusion matrix")
    fig, ax = plt.subplots(figsize=(5.0, 4.0), facecolor="white")
    ax.set_facecolor("white")
    image = ax.imshow(cm, cmap="Blues", interpolation="nearest", vmin=0)
    ax.set_title("Actual vs predicted", fontsize=13, fontweight="bold",
                 color="#172554", pad=12)
    ax.set_xlabel("Predicted label", fontsize=10, fontweight="bold",
                  color="#172554", labelpad=8)
    ax.set_ylabel("Actual label", fontsize=10, fontweight="bold",
                  color="#172554", labelpad=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Malignant (0)", "Benign (1)"],
                       fontsize=8, color="#111827")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Malignant (0)", "Benign (1)"],
                       fontsize=8, color="#111827")
    ax.tick_params(axis="both", colors="#111827", length=0, pad=7)
    threshold = cm.max() / 2 if cm.size else 0
    for row in range(2):
        for col in range(2):
            value = int(cm[row, col])
            ax.text(
                col, row, f"{value:,}",
                ha="center", va="center",
                color="white" if value > threshold else "#111827",
                fontsize=16, fontweight="bold",
            )
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    colorbar = fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    colorbar.ax.tick_params(colors="#334155", labelsize=8)
    fig.tight_layout(pad=1.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with right:
    st.markdown("#### Classification report")
    report = classification_report(
        y, predictions, labels=[0, 1],
        target_names=["Malignant (0)", "Benign (1)"],
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_style = (
        report_df.style
        .format({
            "precision": "{:.4f}", "recall": "{:.4f}",
            "f1-score": "{:.4f}", "support": "{:.0f}",
        })
        .set_properties(**{
            "color": "#111827",
            "background-color": "#ffffff",
            "font-weight": "600",
            "font-size": "12px",
            "border": "1px solid #d1d5db",
            "padding": "8px 6px",
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#e2e8f0"),
                ("color", "#172554"),
                ("font-weight", "800"),
                ("font-size", "11px"),
                ("border", "1px solid #cbd5e1"),
                ("padding", "8px 6px"),
                ("white-space", "normal"),
            ]},
            {"selector": "td", "props": [
                ("color", "#111827"),
                ("border", "1px solid #d1d5db"),
            ]},
            {"selector": "tbody tr:nth-child(even) td", "props": [
                ("background-color", "#f1f5f9"),
            ]},
            {"selector": "tbody th", "props": [
                ("background-color", "#e2e8f0"),
                ("color", "#172554"),
                ("font-weight", "700"),
                ("border", "1px solid #cbd5e1"),
                ("white-space", "normal"),
            ]},
        ])
    )
    st.table(report_style)

if scores is not None:
    st.markdown('<div class="section-title">04 · ROC curve</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Trade-off between true positive rate and false positive rate.</div>', unsafe_allow_html=True)
    # Keep the ROC chart compact and centered instead of stretching it
    # across the full dashboard width.
    _, roc_center, _ = st.columns([1.5, 2, 1.5])
    with roc_center:
        fig_roc, ax_roc = plt.subplots(figsize=(4.2, 2.8), facecolor="white")
        ax_roc.set_facecolor("white")
        ax_roc.plot(
            fpr, tpr, linewidth=2.2, color="#4f46e5",
            label=f"AUC = {auc_value:.4f}",
        )
        ax_roc.plot(
            [0, 1], [0, 1], linestyle="--", linewidth=1.1,
            color="#94a3b8", label="Random classifier",
        )
        ax_roc.set(
            xlim=(0, 1), ylim=(0, 1.03),
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
        )
        ax_roc.set_title(
            "Receiver Operating Characteristic",
            fontsize=11, fontweight="bold", color="#172554", pad=9,
        )
        ax_roc.tick_params(axis="both", labelsize=8, colors="#334155")
        ax_roc.grid(alpha=.18)
        ax_roc.legend(loc="lower right", frameon=False, fontsize=8)
        fig_roc.tight_layout(pad=1.0)
        st.pyplot(fig_roc, use_container_width=True)
        plt.close(fig_roc)

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#64748b;font-size:.82rem;padding:.4rem 0 1rem;">
  <b style="color:#334155;">Classifier Studio</b> · M.Tech Artificial Intelligence & Machine Learning<br>
  Wisconsin Diagnostic Breast Cancer dataset · Saved trained models; no retraining in this app
</div>
""", unsafe_allow_html=True)
