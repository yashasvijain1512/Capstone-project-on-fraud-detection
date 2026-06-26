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
    f1_score,
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

    models = {}
    load_errors = {}

    for name, path in MODEL_PATHS.items():
        try:
            models[name] = joblib.load(path)
        except ModuleNotFoundError as exc:
            load_errors[name] = f"Missing dependency: {exc.name}"
        except Exception as exc:
            load_errors[name] = str(exc)

    if not models:
        details = "; ".join(f"{name}: {error}" for name, error in load_errors.items())
        st.error(f"Unable to load any comparison models. {details}")
        st.stop()

    scaler = joblib.load(SCALER_PATH)
    return models, scaler, load_errors


def get_expected_columns(models: Dict[str, object]):
    for model in models.values():
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    return get_default_feature_order()


@st.cache_data
def load_dataset_from_path(path: str):
    return pd.read_csv(path)


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
    _, fp, fn, _ = cm.ravel()

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

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
        "Precision (Fraud)": report["1"]["precision"],
        "Recall (Fraud)": report["1"]["recall"],
        "F1-Score (Fraud)": report["1"]["f1-score"],
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


def compute_threshold_curve(y_true, y_prob):
    rows = []
    for threshold in np.arange(0.1, 1.0, 0.05):
        y_pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "Threshold": round(float(threshold), 2),
                "Precision": float(classification_report(y_true, y_pred, output_dict=True, zero_division=0)["1"]["precision"]),
                "Recall": float(classification_report(y_true, y_pred, output_dict=True, zero_division=0)["1"]["recall"]),
                "F1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def render_model_comparison_page(show_title=True):
    if show_title:
        st.header("Model Comparison")
        st.caption(
            "Reference: Credit_card_fraud_detection_project.ipynb comparison workflow "
            "for Logistic Regression, Neural Network, and XGBoost."
        )

    models, scaler, load_errors = load_artifacts()
    expected_columns = get_expected_columns(models)

    if load_errors:
        skipped_models = ", ".join(f"{name} ({error})" for name, error in load_errors.items())
        st.warning(f"Some models were skipped: {skipped_models}")

    st.sidebar.header("Model Comparison Data")
    default_dataset_candidates = [
        os.path.join(ROOT_DIR, "credit_card.csv"),
        os.path.join(ROOT_DIR, "creditcard.csv"),
    ]
    default_dataset_path = next((path for path in default_dataset_candidates if os.path.exists(path)), None)

    use_default_dataset = st.sidebar.checkbox("Use local credit_card.csv", value=True, key="cmp_use_default")
    uploaded_file = st.sidebar.file_uploader(
        "Upload labeled CSV (must include Class)",
        type=["csv"],
        key="cmp_upload",
    )

    st.sidebar.header("Model Comparison Settings")
    threshold = st.sidebar.slider("Decision Threshold", 0.10, 0.99, 0.50, 0.01, key="cmp_threshold")
    fp_cost = st.sidebar.number_input("Cost per False Positive ($)", min_value=0, value=100, step=10, key="cmp_fp")
    fn_cost = st.sidebar.number_input("Cost per False Negative ($)", min_value=0, value=1000, step=50, key="cmp_fn")
    use_sample = st.sidebar.checkbox("Use sampled rows for faster comparison", value=True, key="cmp_sample")
    sample_size = st.sidebar.slider("Sample size", min_value=5000, max_value=120000, value=60000, step=5000, key="cmp_sample_size")

    source_label = None
    if uploaded_file is not None:
        source_label = "uploaded file"
        df = pd.read_csv(uploaded_file)
    elif use_default_dataset and default_dataset_path and os.path.exists(default_dataset_path):
        source_label = default_dataset_path
        df = load_dataset_from_path(default_dataset_path)
    else:
        df = None

    if df is None:
        st.info(
            "Upload a labeled CSV with columns Time, V1..V28, Amount, Class, or keep credit_card.csv in project root."
        )
        return

    if "Class" not in df.columns:
        st.error("Dataset must include a Class column (0/1).")
        return

    missing_features = [col for col in expected_columns if col not in df.columns]
    if missing_features:
        st.error(f"Dataset is missing required feature columns: {missing_features}")
        return

    if df["Class"].nunique() < 2:
        st.error("Class must contain both 0 and 1 values for comparison metrics.")
        return

    if use_sample and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42, replace=False)
        st.warning(f"Using sampled dataset ({len(df):,} rows) for faster rendering.")

    st.success(f"Loaded {len(df):,} rows from {source_label}.")

    y_true = pd.to_numeric(df["Class"], errors="coerce").fillna(0).astype(int).to_numpy()
    model_input = preprocess_features(df[expected_columns], expected_columns, scaler)

    results = []
    confusion_matrices = {}
    curves_by_model = {}
    threshold_frames = []

    for model_name, model in models.items():
        y_prob = get_model_probabilities(model, model_input)
        metrics, cm, curves = evaluate_model(y_true, y_prob, threshold, fp_cost, fn_cost)
        results.append({"Model": model_name, **metrics})
        confusion_matrices[model_name] = cm
        curves_by_model[model_name] = curves

        threshold_df = compute_threshold_curve(y_true, y_prob)
        threshold_df["Model"] = model_name
        threshold_frames.append(threshold_df)

    comparison_df = pd.DataFrame(results)
    threshold_curve_df = pd.concat(threshold_frames, ignore_index=True)

    better_high = {
        "Precision (Fraud)": True,
        "Recall (Fraud)": True,
        "F1-Score (Fraud)": True,
        "PR-AUC": True,
        "ROC-AUC": True,
        "Total Misclassification Cost ($)": False,
    }
    rank_metric = st.selectbox(
        "Primary ranking metric",
        list(better_high.keys()),
        index=2,
        key="cmp_rank_metric",
    )

    winner_row = comparison_df.sort_values(rank_metric, ascending=not better_high[rank_metric]).iloc[0]

    top_col_1, top_col_2, top_col_3 = st.columns(3)
    top_col_1.metric("Best Model", winner_row["Model"])
    top_col_2.metric(rank_metric, f"{winner_row[rank_metric]:.4f}")
    top_col_3.metric("Rows Evaluated", f"{len(df):,}")

    tab_summary, tab_curves, tab_confusion, tab_data = st.tabs(
        ["Summary", "Curves", "Confusion Matrices", "Dataset & Export"]
    )

    with tab_summary:
        summary_col_1, summary_col_2 = st.columns([2, 1])
        with summary_col_1:
            st.subheader("Model Comparison Table")
            st.dataframe(
                comparison_df.sort_values(rank_metric, ascending=not better_high[rank_metric]).round(4),
                use_container_width=True,
            )

        with summary_col_2:
            st.subheader("Class Distribution")
            class_df = pd.Series(y_true).value_counts().sort_index().rename(index={0: "Non-Fraud", 1: "Fraud"})
            class_df = class_df.reset_index()
            class_df.columns = ["Class", "Count"]
            fig_dist, ax_dist = plt.subplots(figsize=(4.5, 3.5))
            sns.barplot(data=class_df, x="Class", y="Count", ax=ax_dist)
            ax_dist.set_xlabel("")
            ax_dist.set_ylabel("Count")
            st.pyplot(fig_dist)

        st.subheader("Metric and Cost Charts")
        metric_cols = ["Precision (Fraud)", "Recall (Fraud)", "F1-Score (Fraud)", "PR-AUC", "ROC-AUC"]
        metric_plot_df = comparison_df[["Model"] + metric_cols].melt(
            id_vars="Model", var_name="Metric", value_name="Score"
        )

        fig_metrics, ax_metrics = plt.subplots(figsize=(12, 5))
        sns.barplot(data=metric_plot_df, x="Metric", y="Score", hue="Model", ax=ax_metrics)
        ax_metrics.set_ylim(0, 1.05)
        ax_metrics.set_xlabel("")
        ax_metrics.set_ylabel("Score")
        plt.xticks(rotation=12)
        plt.tight_layout()
        st.pyplot(fig_metrics)

        cost_plot_df = comparison_df[["Model", "Total FP Cost ($)", "Total FN Cost ($)", "Total Misclassification Cost ($)"]]
        cost_plot_df = cost_plot_df.melt(id_vars="Model", var_name="Cost Type", value_name="Cost")

        fig_cost, ax_cost = plt.subplots(figsize=(12, 5))
        sns.barplot(data=cost_plot_df, x="Cost Type", y="Cost", hue="Model", ax=ax_cost)
        ax_cost.set_xlabel("")
        ax_cost.set_ylabel("Cost ($)")
        plt.xticks(rotation=12)
        plt.tight_layout()
        st.pyplot(fig_cost)

    with tab_curves:
        curve_col_1, curve_col_2 = st.columns(2)

        with curve_col_1:
            st.subheader("Precision-Recall Curves")
            fig_pr, ax_pr = plt.subplots(figsize=(6.5, 4.5))
            for model_name, curve_data in curves_by_model.items():
                ax_pr.plot(curve_data["pr_recall"], curve_data["pr_precision"], label=model_name)
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.grid(True, alpha=0.25)
            ax_pr.legend()
            st.pyplot(fig_pr)

        with curve_col_2:
            st.subheader("ROC Curves")
            fig_roc, ax_roc = plt.subplots(figsize=(6.5, 4.5))
            for model_name, curve_data in curves_by_model.items():
                ax_roc.plot(curve_data["roc_fpr"], curve_data["roc_tpr"], label=model_name)
            ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6)
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.grid(True, alpha=0.25)
            ax_roc.legend()
            st.pyplot(fig_roc)

        st.subheader("Threshold vs F1 / Precision / Recall")
        model_for_threshold = st.selectbox("Select model", comparison_df["Model"].tolist(), key="cmp_threshold_model")
        selected = threshold_curve_df[threshold_curve_df["Model"] == model_for_threshold]

        fig_thr, ax_thr = plt.subplots(figsize=(10, 4.5))
        ax_thr.plot(selected["Threshold"], selected["Precision"], label="Precision")
        ax_thr.plot(selected["Threshold"], selected["Recall"], label="Recall")
        ax_thr.plot(selected["Threshold"], selected["F1"], label="F1")
        ax_thr.axvline(threshold, color="red", linestyle="--", alpha=0.7, label=f"Active threshold {threshold:.2f}")
        ax_thr.set_xlabel("Threshold")
        ax_thr.set_ylabel("Score")
        ax_thr.set_ylim(0, 1.02)
        ax_thr.grid(True, alpha=0.25)
        ax_thr.legend()
        st.pyplot(fig_thr)

    with tab_confusion:
        st.subheader("Confusion Matrices")
        cm_cols = st.columns(3)
        for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
            with cm_cols[idx % 3]:
                fig_cm, ax_cm = plt.subplots(figsize=(4.6, 4.1))
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

    with tab_data:
        st.subheader("Export Comparison")
        export_df = comparison_df.sort_values(rank_metric, ascending=not better_high[rank_metric]).round(6)
        st.dataframe(export_df, use_container_width=True)
        st.download_button(
            label="Download comparison as CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison_summary.csv",
            mime="text/csv",
            key="cmp_download",
        )
        st.caption("Tip: increase sample size or disable sampling for final, full-dataset model ranking.")
