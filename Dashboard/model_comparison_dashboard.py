import os
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "Model_DIR")
MODEL_PATHS = {
    "Logistic Regression (Tuned)": os.path.join(MODEL_DIR, "best_logistic_regression_rs_model.joblib"),
    "Neural Network (Tuned)": os.path.join(MODEL_DIR, "best_nn_model.joblib"),
    "XGBoost (Tuned)": os.path.join(MODEL_DIR, "best_xgboost_model.joblib"),
}
SCALER_PATH = os.path.join(ROOT_DIR, "scaler_for_api.joblib")


st.set_page_config(page_title="Fraud Model Comparison Dashboard", layout="wide")


def get_default_feature_order():
    return ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


@st.cache_resource
def load_artifacts():
    missing = [name for name, path in MODEL_PATHS.items() if not os.path.exists(path)]
    if missing:
        st.error(f"Missing model files: {', '.join(missing)} in Model_DIR.")
        st.stop()

    if not os.path.exists(SCALER_PATH):
        st.error("Missing scaler_for_api.joblib in project root.")
        st.stop()

    models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}
    scaler = joblib.load(SCALER_PATH)
    return models, scaler


def get_expected_columns(models: Dict[str, object]):
    for model in models.values():
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    return get_default_feature_order()


def preprocess_features(raw_df: pd.DataFrame, expected_columns, scaler):
    model_df = raw_df.copy()

    for col in expected_columns:
        if col not in model_df.columns:
            model_df[col] = 0.0

    model_df = model_df[expected_columns]
    model_df = model_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if "Time" in model_df.columns and "Amount" in model_df.columns:
        model_df[["Time", "Amount"]] = scaler.transform(model_df[["Time", "Amount"]])

    return model_df


def get_model_probabilities(model, model_input: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(model_input)
        if isinstance(probs, np.ndarray) and probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
        return np.asarray(probs).reshape(-1)

    raw_preds = model.predict(model_input)
    probs = np.asarray(raw_preds).reshape(-1)
    return np.clip(probs, 0.0, 1.0)


def evaluate_model(y_true, y_prob, threshold, fp_cost, fn_cost) -> Tuple[dict, np.ndarray, dict]:
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1 = report["1"]["f1-score"]

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")
        fpr_curve, tpr_curve = np.array([0.0, 1.0]), np.array([0.0, 1.0])

    total_fp_cost = fp * fp_cost
    total_fn_cost = fn * fn_cost

    metrics = {
        "Precision (Fraud)": precision,
        "Recall (Fraud)": recall,
        "F1-Score (Fraud)": f1,
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc,
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "Total FP Cost ($)": float(total_fp_cost),
        "Total FN Cost ($)": float(total_fn_cost),
        "Total Misclassification Cost ($)": float(total_fp_cost + total_fn_cost),
    }

    curves = {
        "pr_recall": recall_curve,
        "pr_precision": precision_curve,
        "roc_fpr": fpr_curve,
        "roc_tpr": tpr_curve,
    }

    return metrics, cm, curves


st.title("Fraud Detection Model Comparison Dashboard")
st.caption(
    "Reference: comparison workflow from Credit_card_fraud_detection_project.ipynb "
    "(Logistic Regression vs Neural Network vs XGBoost)."
)

models, scaler = load_artifacts()
expected_columns = get_expected_columns(models)

st.sidebar.header("Input Data")
default_dataset_path = os.path.join(ROOT_DIR, "creditcard.csv")
use_default_dataset = st.sidebar.checkbox("Use local creditcard.csv (if available)", value=True)
uploaded_file = st.sidebar.file_uploader("Upload labeled CSV (must include Class)", type=["csv"])

st.sidebar.header("Evaluation Settings")
threshold = st.sidebar.slider("Decision Threshold", min_value=0.10, max_value=0.99, value=0.50, step=0.01)
fp_cost = st.sidebar.number_input("Cost per False Positive ($)", min_value=0, value=100, step=10)
fn_cost = st.sidebar.number_input("Cost per False Negative ($)", min_value=0, value=1000, step=50)

source_label = None
if uploaded_file is not None:
    source_label = "uploaded file"
    df = pd.read_csv(uploaded_file)
elif use_default_dataset and os.path.exists(default_dataset_path):
    source_label = default_dataset_path
    df = pd.read_csv(default_dataset_path)
else:
    df = None

if df is None:
    st.info(
        "Upload a labeled CSV with columns Time, V1..V28, Amount, Class to generate model comparison charts. "
        "If creditcard.csv exists in project root, enable the sidebar checkbox to use it."
    )
    st.stop()

if "Class" not in df.columns:
    st.error("The dataset must include a 'Class' column (0/1) for comparison metrics.")
    st.stop()

missing_features = [col for col in expected_columns if col not in df.columns]
if missing_features:
    st.error(f"Dataset is missing required feature columns: {missing_features}")
    st.stop()

if df["Class"].nunique() < 2:
    st.error("The Class column needs both classes (0 and 1) to compute comparison metrics.")
    st.stop()

st.success(f"Loaded {len(df):,} rows from {source_label}.")

feature_df = df[expected_columns]
y_true = pd.to_numeric(df["Class"], errors="coerce").fillna(0).astype(int).to_numpy()
model_input = preprocess_features(feature_df, expected_columns, scaler)

results = []
confusion_matrices = {}
curves_by_model = {}

for model_name, model in models.items():
    y_prob = get_model_probabilities(model, model_input)
    metrics, cm, curves = evaluate_model(y_true, y_prob, threshold, fp_cost, fn_cost)
    results.append({"Model": model_name, **metrics})
    confusion_matrices[model_name] = cm
    curves_by_model[model_name] = curves

comparison_df = pd.DataFrame(results)

st.subheader("Comparison Table")
st.dataframe(
    comparison_df.sort_values("F1-Score (Fraud)", ascending=False).round(4),
    use_container_width=True,
)

st.subheader("Metric Charts")
metric_cols = ["Precision (Fraud)", "Recall (Fraud)", "F1-Score (Fraud)", "PR-AUC", "ROC-AUC"]
metric_plot_df = comparison_df[["Model"] + metric_cols].melt(id_vars="Model", var_name="Metric", value_name="Score")

fig_metrics, ax_metrics = plt.subplots(figsize=(12, 6))
sns.barplot(data=metric_plot_df, x="Metric", y="Score", hue="Model", ax=ax_metrics)
ax_metrics.set_ylim(0, 1.05)
ax_metrics.set_title("Core Classification Metrics by Model")
ax_metrics.set_xlabel("")
ax_metrics.set_ylabel("Score")
plt.xticks(rotation=15)
plt.tight_layout()
st.pyplot(fig_metrics)

cost_plot_df = comparison_df[["Model", "Total FP Cost ($)", "Total FN Cost ($)", "Total Misclassification Cost ($)"]]
cost_plot_df = cost_plot_df.melt(id_vars="Model", var_name="Cost Type", value_name="Cost")

fig_cost, ax_cost = plt.subplots(figsize=(12, 6))
sns.barplot(data=cost_plot_df, x="Cost Type", y="Cost", hue="Model", ax=ax_cost)
ax_cost.set_title("Cost Comparison by Model")
ax_cost.set_xlabel("")
ax_cost.set_ylabel("Cost ($)")
plt.xticks(rotation=15)
plt.tight_layout()
st.pyplot(fig_cost)

curve_col_1, curve_col_2 = st.columns(2)

with curve_col_1:
    st.subheader("Precision-Recall Curves")
    fig_pr, ax_pr = plt.subplots(figsize=(7, 5))
    for model_name, curve_data in curves_by_model.items():
        ax_pr.plot(curve_data["pr_recall"], curve_data["pr_precision"], label=model_name)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("PR Curve")
    ax_pr.legend()
    ax_pr.grid(True, alpha=0.3)
    st.pyplot(fig_pr)

with curve_col_2:
    st.subheader("ROC Curves")
    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
    for model_name, curve_data in curves_by_model.items():
        ax_roc.plot(curve_data["roc_fpr"], curve_data["roc_tpr"], label=model_name)
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()
    ax_roc.grid(True, alpha=0.3)
    st.pyplot(fig_roc)

st.subheader("Confusion Matrices")
cm_cols = st.columns(3)
for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
    with cm_cols[idx % 3]:
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
            ax=ax_cm,
        )
        ax_cm.set_title(model_name)
        ax_cm.set_xlabel("")
        ax_cm.set_ylabel("")
        st.pyplot(fig_cm)

st.caption(
    "Tip: tune the decision threshold and FP/FN costs in the sidebar to compare business trade-offs between the three models."
)
