"""Reproducible training script.

Pipeline: TF-IDF (word 1-2 grams) -> classifier.
Candidates (compared by 5-fold cross-validation on the TRAINING split only):
  * Logistic Regression
  * Linear SVM wrapped in CalibratedClassifierCV (so probabilities are valid)
The best candidate by macro-F1 is refit on the training split and evaluated ONCE
on a held-out test split. Nothing from the test split is used for selection
or fitting (no leakage). The TF-IDF vectoriser lives inside the pipeline, so it
is re-fitted inside every CV fold.

Outputs: models/resume_model.joblib, models/metrics.json
Usage:   python train_model.py [--data dataset/resumes.csv] [--test-size 0.2]
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils.text_preprocessing import clean_text

BASE = Path(__file__).parent
SEED = 42


def expected_calibration_error(y_true, proba, classes, n_bins=10):
    """Top-label ECE: gap between stated confidence and actual accuracy."""
    conf = proba.max(axis=1)
    pred = np.array(classes)[proba.argmax(axis=1)]
    correct = (pred == np.array(y_true)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def build_candidates():
    def tfidf():
        return TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), sublinear_tf=True,
                               min_df=2, max_df=0.9)
    lr = Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(max_iter=2000))])
    svm = Pipeline([("tfidf", tfidf()),
                    ("clf", CalibratedClassifierCV(LinearSVC(), cv=3, method="sigmoid"))])
    return {
        "TF-IDF + Logistic Regression": (lr, {"clf__C": [1, 5, 20, 50]}),
        "TF-IDF + Linear SVM (calibrated)": (svm, {"clf__estimator__C": [0.1, 0.5, 2]}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(BASE / "dataset" / "resumes.csv"))
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    n_raw = len(df)
    df = df.dropna(subset=["resume_text", "category"])
    df["resume_text"] = df["resume_text"].astype(str)
    df = df[df["resume_text"].str.strip().str.len() > 30]
    # Drop exact duplicates BEFORE splitting so a record can't appear in both splits.
    df["_key"] = df["resume_text"].map(clean_text)
    df = df.drop_duplicates("_key").drop(columns="_key").reset_index(drop=True)
    print(f"Records: {n_raw} raw -> {len(df)} after cleaning/deduplication")

    X_train, X_test, y_train, y_test = train_test_split(
        df["resume_text"], df["category"], test_size=args.test_size,
        stratify=df["category"], random_state=SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, best = [], None
    for name, (pipe, grid) in build_candidates().items():
        t0 = time.time()
        gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=1, refit=True)
        gs.fit(X_train, y_train)
        row = {"model": name, "cv_macro_f1": round(float(gs.best_score_), 4),
               "best_params": {k: v for k, v in gs.best_params_.items()},
               "fit_seconds": round(time.time() - t0, 1)}
        results.append(row)
        print(f"{name}: CV macro-F1={row['cv_macro_f1']} params={row['best_params']}")
        if best is None or gs.best_score_ > best[0]:
            best = (gs.best_score_, name, gs.best_estimator_)

    _, best_name, model = best
    print(f"Selected: {best_name}")

    # ---- single evaluation on the untouched test split ----
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    p, r, f, s = precision_recall_fscore_support(y_test, y_pred, labels=classes, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    info_path = Path(args.data).parent / "dataset_info.json"
    ds_info = json.loads(info_path.read_text()) if info_path.exists() else {
        "type": "user-provided", "note": "No dataset_info.json found; origin not documented."}

    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "selected_model": best_name,
        "candidates": results,
        "dataset": {**ds_info, "records_after_cleaning": int(len(df)), "records_raw": int(n_raw),
                    "class_distribution": df["category"].value_counts().to_dict()},
        "split": {"method": "stratified hold-out", "test_size": args.test_size, "random_state": SEED,
                  "train_records": int(len(X_train)), "test_records": int(len(X_test)),
                  "selection": "5-fold stratified CV on the training split only; test split used once"},
        "test_metrics": {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "macro": {"precision": round(float(pm), 4), "recall": round(float(rm), 4), "f1": round(float(fm), 4)},
            "weighted": {"precision": round(float(pw), 4), "recall": round(float(rw), 4), "f1": round(float(fw), 4)},
            "expected_calibration_error": round(expected_calibration_error(y_test, proba, classes), 4),
        },
        "per_class": [{"category": c, "precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                       "f1": round(float(f[i]), 4), "support": int(s[i])} for i, c in enumerate(classes)],
        "confusion_matrix": {"labels": classes, "matrix": cm.tolist()},
        "classes": classes,
    }

    out_dir = BASE / "models"
    out_dir.mkdir(exist_ok=True)
    joblib.dump(model, out_dir / "resume_model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["test_metrics"], indent=2))
    print(f"Saved model and metrics to {out_dir}")


if __name__ == "__main__":
    main()
