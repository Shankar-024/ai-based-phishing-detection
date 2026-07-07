#!/usr/bin/env python3
"""
classify.py — Phishing email classifier using URL-based features.

Pipeline
--------
1. Load analysis.csv  (URL signals extracted by url.py)
2. Load original source CSVs to pull the `label` column (1=phishing, 0=legit)
3. Merge on (source_file, row_index) to create a labelled feature matrix
4. Evaluate three classifiers with 5-fold cross-validation:
     - Logistic Regression  (linear baseline)
     - Random Forest        (ensemble, handles non-linearity)
     - XGBoost              (gradient-boosted trees, primary model)
5. Print a full classification report + confusion matrix for each model
6. Show SHAP feature importances from XGBoost (mean |SHAP| values)
7. Save trained XGBoost model to model.pkl

Usage
-----
  python classify.py
  python classify.py --analysis analysis.csv --dataset "Phishing Email Dataset"
"""

import argparse
import csv
import glob
import os
import sys

import pickle

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Raise CSV field limit — some email bodies are very large
try:
    csv.field_size_limit(10_000_000)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

# ---------------------------------------------------------------------------
# Feature columns we pull from analysis.csv.
# All are numeric (0/1 flags or integer counts/scores).
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "has_ip_url",              # 1 if any URL uses a raw IP address
    "has_shortener",           # 1 if any URL is a known shortener (derived below)
    "suspect_typosquat",       # 1 if any URL is one edit away from a trusted domain
    "any_brand_in_subdomain",  # 1 if a trusted brand appears as a subdomain
    "any_suspicious_path",     # 1 if URL path contains phishing keywords
    "any_high_entropy",        # 1 if any domain looks DGA-generated
    "max_risk_score",          # 0-100 composite risk score (highest URL in email)
    "url_count",               # number of unique URLs found in the email
    "min_levenshtein",         # edit distance to the closest trusted domain
    # Raw numeric features
    "max_url_length",          # length of the longest URL
    "max_special_char_count",  # special chars (@-_=%;/) in the longest URL
    "max_digit_count",         # digit characters in the most digit-heavy URL
    "max_subdomain_depth",     # subdomain label depth (labels before SLD)
    "max_path_depth",          # path segment depth of the deepest URL
    "max_entropy_score",       # Shannon entropy of the highest-entropy domain
    # New signals
    "any_double_ext",          # 1 if any URL has a double file extension (e.g. .pdf.exe)
    "any_susp_port",           # 1 if any URL uses a non-standard port
    "any_hex_domain",          # 1 if any URL has percent-encoded chars in hostname
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_analysis(path: str) -> pd.DataFrame:
    """Read analysis.csv produced by url.py and engineer derived columns."""
    df = pd.read_csv(path, dtype=str, low_memory=False)

    # Convert numeric columns; fill missing with 0
    for col in ["has_ip_url", "suspect_typosquat", "any_brand_in_subdomain",
                "any_suspicious_path", "any_high_entropy", "max_risk_score",
                "min_levenshtein",
                "max_url_length", "max_special_char_count", "max_digit_count",
                "max_subdomain_depth", "max_path_depth", "max_entropy_score",
                "any_double_ext", "any_susp_port", "any_hex_domain"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    # `shortener_domains` is a comma-separated string of domains (or empty)
    df["has_shortener"] = (
        df["shortener_domains"].fillna("").str.strip().ne("").astype(int)
    )

    # `found_urls` is a semicolon-separated list of URLs produced by url.py
    df["url_count"] = (
        df["found_urls"]
        .fillna("")
        .apply(lambda x: len([u for u in x.split(";") if u.strip()]))
    )

    # Normalise path separators (regex=False avoids regex interpretation of backslash)
    df["source_file"] = (df["source_file"]
        .str.replace("\\\\", "/", regex=False)
        .str.replace("\\", "/", regex=False))
    df["row_index"] = pd.to_numeric(df["row_index"], errors="coerce").fillna(-1).astype(int)

    return df


def load_labels(dataset_dir: str) -> pd.DataFrame:
    """
    Read every CSV under dataset_dir and collect (source_file, row_index, label).
    source_file is stored as a path relative to dataset_dir (forward slashes).
    row_index is 0-based (first data row = 0, matching url.py's output).
    """
    records = []
    for fpath in glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True):
        rel = os.path.relpath(fpath, dataset_dir).replace("\\", "/")
        with open(fpath, encoding="utf-8", errors="ignore", newline="") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                label = row.get("label", "").strip()
                if label in ("0", "1"):
                    records.append({
                        "source_file": rel,
                        "row_index": i,
                        "label": int(label),
                    })
    return pd.DataFrame(records)


def strip_dataset_prefix(df: pd.DataFrame, dataset_dir: str) -> pd.DataFrame:
    """
    Normalise the source_file column in an analysis DataFrame so it is a
    relative path from dataset_dir — matching what load_labels() produces.
    e.g. 'Phishing Email Dataset/CEAS_08.csv' -> 'CEAS_08.csv'
         'Human-LLM.../human-generated/legit.csv' -> 'human-generated/legit.csv'
    """
    dd = dataset_dir.replace("\\", "/").rstrip("/") + "/"
    def _strip(path: str) -> str:
        p = path.replace("\\", "/")
        idx = p.find(dd)
        if idx >= 0:
            return p[idx + len(dd):]
        # Fallback: strip up to the last occurrence of dataset dir name
        dir_name = dd.rstrip("/").split("/")[-1] + "/"
        idx2 = p.rfind(dir_name)
        if idx2 >= 0:
            return p[idx2 + len(dir_name):]
        return os.path.basename(p)
    df = df.copy()
    df["source_file"] = df["source_file"].apply(_strip)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train phishing classifiers on URL features.")
    parser.add_argument("--analysis", default="analysis.csv",
                        help="Path to analysis.csv from url.py (default: analysis.csv)")
    parser.add_argument("--dataset", default="Phishing Email Dataset",
                        help="Directory containing the original labelled CSVs")
    parser.add_argument("--analysis2", default=None,
                        help="Optional second analysis CSV (e.g. analysis_human_llm.csv)")
    parser.add_argument("--dataset2", default=None,
                        help="Optional second labelled dataset directory")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    print(f"Loading analysis  : {args.analysis}")
    analysis = load_analysis(args.analysis)
    analysis = strip_dataset_prefix(analysis, args.dataset)
    print(f"  {len(analysis):,} rows")

    print(f"Loading labels    : {args.dataset}/")
    labels = load_labels(args.dataset)
    print(f"  {len(labels):,} labelled rows  "
          f"(phishing={labels['label'].sum():,}, "
          f"legit={(labels['label']==0).sum():,})")

    # Optionally load a second analysis + label dataset and concatenate
    if args.analysis2 and args.dataset2:
        print(f"Loading analysis2 : {args.analysis2}")
        analysis2 = load_analysis(args.analysis2)
        analysis2 = strip_dataset_prefix(analysis2, args.dataset2)
        print(f"  {len(analysis2):,} rows")

        print(f"Loading labels2   : {args.dataset2}/")
        labels2 = load_labels(args.dataset2)
        print(f"  {len(labels2):,} labelled rows  "
              f"(phishing={labels2['label'].sum():,}, "
              f"legit={(labels2['label']==0).sum():,})")

        analysis = pd.concat([analysis, analysis2], ignore_index=True)
        labels   = pd.concat([labels,   labels2],   ignore_index=True)
        print(f"Combined analysis : {len(analysis):,} rows")
        print(f"Combined labels   : {len(labels):,} rows")

    # -----------------------------------------------------------------------
    # 2. Merge features with labels
    # -----------------------------------------------------------------------
    merged = analysis.merge(labels, on=["source_file", "row_index"], how="inner")
    print(f"\nMerged dataset    : {len(merged):,} rows")

    if merged.empty:
        print("ERROR: merge produced 0 rows — check that source_file / row_index match.")
        sys.exit(1)

    label_counts = merged["label"].value_counts().sort_index()
    print(f"  Label 0 (legit)   : {label_counts.get(0, 0):,}")
    print(f"  Label 1 (phishing): {label_counts.get(1, 0):,}")

    X = np.asarray(merged[FEATURE_COLS], dtype=float)
    y = np.asarray(merged["label"], dtype=int)

    # -----------------------------------------------------------------------
    # 3. Define classifiers
    # -----------------------------------------------------------------------
    # Compute scale_pos_weight for XGBoost (handles class imbalance)
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
                class_weight="balanced", random_state=42, n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("clf", xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=5,
                scale_pos_weight=spw, eval_metric="logloss",
                random_state=42, n_jobs=-1,
            )),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # -----------------------------------------------------------------------
    # 4. Evaluate each classifier
    # -----------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 64)

    for name, pipeline in classifiers.items():
        print(f"\n{'-' * 64}")
        print(f"  {name}")
        print(f"{'-' * 64}")

        # cross_val_predict gives one prediction per sample (out-of-fold)
        y_pred = cross_val_predict(pipeline, X, y, cv=cv)
        y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

        print(classification_report(y, y_pred, target_names=["Legit (0)", "Phishing (1)"],
                                    digits=4))

        auc = roc_auc_score(y, y_prob)
        print(f"  ROC-AUC : {auc:.4f}")

        # Confusion matrix (counts)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"\n  Confusion matrix:")
        print(f"              Predicted Legit  Predicted Phishing")
        print(f"  Actual Legit       {tn:>7,}           {fp:>7,}")
        print(f"  Actual Phishing    {fn:>7,}           {tp:>7,}")

    # -----------------------------------------------------------------------
    # 5. XGBoost — fit on full dataset, compute SHAP values, save model
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 64}")
    print("SHAP FEATURE IMPORTANCES (XGBoost, trained on full dataset)")
    print(f"{'=' * 64}")

    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    xgb_model.fit(X, y)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
    # shap_values shape: (n_samples, n_features) for binary classification
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_pairs = sorted(zip(FEATURE_COLS, mean_abs_shap), key=lambda t: t[1], reverse=True)

    max_width = max(len(c) for c in FEATURE_COLS)
    for feat, imp in shap_pairs:
        bar = "#" * int(imp * 30)
        print(f"  {feat:<{max_width}}  {imp:.4f}  {bar}")

    # -----------------------------------------------------------------------
    # 6. Save trained XGBoost model for use by app.py
    # -----------------------------------------------------------------------
    model_path = "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": xgb_model, "features": FEATURE_COLS}, f)
    print(f"\nModel saved to {model_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
