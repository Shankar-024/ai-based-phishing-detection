#!/usr/bin/env python3
"""
generate_report_charts.py
Generates report-quality charts from the model evaluation pipeline.

Charts saved to report_charts/:
  1. dataset_distribution.png   — class balance across both datasets
  2. confusion_matrices.png     — 3 × 1 grid of confusion matrices
  3. roc_curves.png             — overlaid ROC curves for all 3 models
  4. metrics_comparison.png     — grouped bar chart (Precision/Recall/F1/AUC)
  5. shap_importance.png        — horizontal bar chart of mean |SHAP| values
  6. precision_recall_curves.png— Precision-Recall curves for all 3 models

Usage:
  python generate_report_charts.py
  python generate_report_charts.py \
      --analysis analysis.csv --dataset "Phishing Email Dataset" \
      --analysis2 analysis_human_llm.csv \
      --dataset2 "Human-LLM generated phishing-legitimate emails Dataset"
"""

import argparse
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Raise CSV field limit
try:
    csv.field_size_limit(10_000_000)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

FEATURE_COLS = [
    "has_ip_url", "has_shortener", "suspect_typosquat",
    "any_brand_in_subdomain", "any_suspicious_path", "any_high_entropy",
    "max_risk_score", "url_count", "min_levenshtein",
    "max_url_length", "max_special_char_count", "max_digit_count",
    "max_subdomain_depth", "max_path_depth", "max_entropy_score",
    "any_double_ext", "any_susp_port", "any_hex_domain",
]

OUT_DIR = "report_charts"

PALETTE = {
    "Logistic Regression": "#4C72B0",
    "Random Forest":        "#55A868",
    "XGBoost":              "#C44E52",
}


# ---------------------------------------------------------------------------
# Data helpers (same as classify.py)
# ---------------------------------------------------------------------------

def load_analysis(path):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    for col in ["has_ip_url", "suspect_typosquat", "any_brand_in_subdomain",
                "any_suspicious_path", "any_high_entropy", "max_risk_score",
                "min_levenshtein", "max_url_length", "max_special_char_count",
                "max_digit_count", "max_subdomain_depth", "max_path_depth",
                "max_entropy_score", "any_double_ext", "any_susp_port", "any_hex_domain"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0
    df["has_shortener"] = df["shortener_domains"].fillna("").str.strip().ne("").astype(int)
    df["url_count"] = (df["found_urls"].fillna("")
                       .apply(lambda x: len([u for u in x.split(";") if u.strip()])))
    # Normalise path separators (regex=False avoids regex interpretation of backslash)
    df["source_file"] = (df["source_file"]
        .str.replace("\\\\", "/", regex=False)
        .str.replace("\\", "/", regex=False))
    df["row_index"] = pd.to_numeric(df["row_index"], errors="coerce").fillna(-1).astype(int)
    return df


def strip_dataset_prefix(df, dataset_dir):
    """Normalise source_file to a path relative to dataset_dir (forward slashes)."""
    dd = dataset_dir.replace("\\", "/").rstrip("/") + "/"
    def _strip(path):
        p = path.replace("\\", "/")
        idx = p.find(dd)
        if idx >= 0:
            return p[idx + len(dd):]
        dir_name = dd.rstrip("/").split("/")[-1] + "/"
        idx2 = p.rfind(dir_name)
        if idx2 >= 0:
            return p[idx2 + len(dir_name):]
        return os.path.basename(p)
    df = df.copy()
    df["source_file"] = df["source_file"].apply(_strip)
    return df


def load_labels(dataset_dir):
    """source_file key is a path relative to dataset_dir (forward slashes)."""
    records = []
    for fpath in glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True):
        rel = os.path.relpath(fpath, dataset_dir).replace("\\", "/")
        with open(fpath, encoding="utf-8", errors="ignore", newline="") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                label = row.get("label", "").strip()
                if label in ("0", "1"):
                    records.append({"source_file": rel, "row_index": i, "label": int(label)})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 1. Dataset distribution
# ---------------------------------------------------------------------------

def chart_dataset_distribution(merged):
    counts = merged["label"].value_counts().sort_index()
    labels = ["Legitimate (0)", "Phishing (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    colors = ["#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Dataset Class Distribution", fontsize=14, fontweight="bold")

    # Bar chart
    bars = axes[0].bar(labels, values, color=colors, width=0.5, edgecolor="white")
    axes[0].set_ylabel("Number of samples")
    axes[0].set_title("Sample Counts")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     f"{val:,}", ha="center", va="bottom", fontsize=11)

    # Pie chart
    axes[1].pie(values, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, wedgeprops={"edgecolor": "white"})
    axes[1].set_title("Class Split")

    fig.tight_layout()
    save(fig, "1_dataset_distribution.png")


# ---------------------------------------------------------------------------
# 2. Confusion matrices
# ---------------------------------------------------------------------------

def chart_confusion_matrices(results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Confusion Matrices (5-Fold CV)", fontsize=14, fontweight="bold")

    for ax, (name, data) in zip(axes, results.items()):
        cm = data["cm"]
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Legit", "Phishing"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Legit", "Phishing"])

        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=13, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    save(fig, "2_confusion_matrices.png")


# ---------------------------------------------------------------------------
# 3. ROC curves
# ---------------------------------------------------------------------------

def chart_roc_curves(results):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")

    for name, data in results.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        auc_score = data["auc"]
        ax.plot(fpr, tpr, lw=2, color=PALETTE[name],
                label=f"{name}  (AUC = {auc_score:.4f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — 5-Fold Cross-Validation", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "3_roc_curves.png")


# ---------------------------------------------------------------------------
# 4. Metrics comparison bar chart
# ---------------------------------------------------------------------------

def chart_metrics_comparison(results):
    metric_keys  = ["precision_phish", "recall_phish", "f1_phish",
                    "precision_legit",  "recall_legit",  "f1_legit", "auc"]
    metric_labels = ["Precision\n(Phishing)", "Recall\n(Phishing)", "F1\n(Phishing)",
                     "Precision\n(Legit)",    "Recall\n(Legit)",    "F1\n(Legit)", "ROC-AUC"]

    model_names = list(results.keys())
    x = np.arange(len(metric_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, name in enumerate(model_names):
        vals = [results[name][k] for k in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=PALETTE[name], edgecolor="white", alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Metrics Comparison (5-Fold CV)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "4_metrics_comparison.png")


# ---------------------------------------------------------------------------
# 5. SHAP feature importances
# ---------------------------------------------------------------------------

def chart_shap_importance(X, y, spw):
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    xgb_model.fit(X, y)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)
    feats  = [FEATURE_COLS[i] for i in order]
    values = [mean_abs[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 7))
    bars = ax.barh(feats, values, color="#4C72B0", edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("XGBoost — SHAP Feature Importances\n(trained on full dataset)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save(fig, "5_shap_importance.png")


# ---------------------------------------------------------------------------
# 6. Precision-Recall curves
# ---------------------------------------------------------------------------

def chart_pr_curves(results):
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, data in results.items():
        prec, rec, _ = precision_recall_curve(data["y_true"], data["y_prob"])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, lw=2, color=PALETTE[name],
                label=f"{name}  (AP = {pr_auc:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — 5-Fold Cross-Validation",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "6_precision_recall_curves.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis",  default="analysis.csv")
    parser.add_argument("--dataset",   default="Phishing Email Dataset")
    parser.add_argument("--analysis2", default=None)
    parser.add_argument("--dataset2",  default=None)
    args = parser.parse_args()

    # Load & merge
    print("Loading data...")
    analysis = strip_dataset_prefix(load_analysis(args.analysis), args.dataset)
    labels   = load_labels(args.dataset)
    if args.analysis2 and args.dataset2:
        analysis = pd.concat(
            [analysis, strip_dataset_prefix(load_analysis(args.analysis2), args.dataset2)],
            ignore_index=True)
        labels = pd.concat([labels, load_labels(args.dataset2)], ignore_index=True)

    merged = analysis.merge(labels, on=["source_file", "row_index"], how="inner")
    print(f"Merged dataset: {len(merged):,} rows  "
          f"(legit={( merged['label']==0).sum():,}, phishing={(merged['label']==1).sum():,})")

    X = np.asarray(merged[FEATURE_COLS], dtype=float)
    y = np.asarray(merged["label"], dtype=int)
    n_legit = int((y == 0).sum())
    n_phish = int((y == 1).sum())
    spw = n_legit / max(n_phish, 1)

    classifiers = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=None,
                class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
        "XGBoost": Pipeline([
            ("clf", xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=5,
                scale_pos_weight=spw, eval_metric="logloss",
                random_state=42, n_jobs=-1)),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Running cross-validation (this may take a minute)...")
    results = {}
    for name, pipeline in classifiers.items():
        print(f"  {name}...")
        y_pred = cross_val_predict(pipeline, X, y, cv=cv)
        y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
        rep    = classification_report(y, y_pred, output_dict=True)
        cm     = confusion_matrix(y, y_pred)
        results[name] = {
            "y_true":         y,
            "y_pred":         y_pred,
            "y_prob":         y_prob,
            "cm":             cm,
            "auc":            roc_auc_score(y, y_prob),
            "precision_phish": rep["1"]["precision"],
            "recall_phish":    rep["1"]["recall"],
            "f1_phish":        rep["1"]["f1-score"],
            "precision_legit": rep["0"]["precision"],
            "recall_legit":    rep["0"]["recall"],
            "f1_legit":        rep["0"]["f1-score"],
        }

    print("\nGenerating charts...")
    chart_dataset_distribution(merged)
    chart_confusion_matrices(results)
    chart_roc_curves(results)
    chart_metrics_comparison(results)
    chart_shap_importance(X, y, spw)
    chart_pr_curves(results)

    print(f"\nAll charts saved to: {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
