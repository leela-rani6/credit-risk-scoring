"""
Credit Risk Scoring - Training Pipeline
Dataset: Kaggle "Credit Risk Dataset" (loan_status = 1 means the borrower defaulted)
Put the CSV at data/cs-training.csv, then run:  python train.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import TARGET, clean_training_data, engineer

warnings.filterwarnings("ignore")
DATA_PATH = Path("cs-training.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
SEED = 42


def ks_statistic(y_true, y_prob) -> float:
    """Kolmogorov-Smirnov: max gap between cumulative defaulter and non-defaulter distributions."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def cost_threshold(y_true, y_prob, fn_cost=5, fp_cost=1):
    """A missed default costs fn_cost times more than rejecting a good customer."""
    ts = np.linspace(0.05, 0.95, 91)
    costs = [fn_cost * ((y_prob < t) & (y_true == 1)).sum()
             + fp_cost * ((y_prob >= t) & (y_true == 0)).sum() for t in ts]
    return float(ts[int(np.argmin(costs))])


def evaluate(name, y_true, y_prob, threshold) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    res = {
        "model": name,
        "AUC": round(roc_auc_score(y_true, y_prob), 4),
        "PR_AUC": round(average_precision_score(y_true, y_prob), 4),
        "KS": round(ks_statistic(y_true, y_prob), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "threshold": round(threshold, 2),
    }
    print(f"\n=== {name} ===")
    print(json.dumps(res, indent=2))
    return res


def assign_band(p: float, low: float, high: float) -> str:
    return "Low" if p < low else "Medium" if p < high else "High"


def main():
    raw = pd.read_csv(DATA_PATH)
    print(f"Raw rows: {len(raw):,}")
    df, stats = clean_training_data(raw)
    print(f"Rows after cleaning: {len(df):,} | default rate: {df[TARGET].mean():.2%}")

    X, y = engineer(df), df[TARGET]
    X = X.drop(columns=["grade_num", "loan_int_rate", "IntRateMissing"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    # Baseline: Logistic Regression
    logreg = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=2000, class_weight="balanced"))
    logreg.fit(X_train, y_train)
    lr_prob = logreg.predict_proba(X_test)[:, 1]
    lr_res = evaluate("Logistic Regression (baseline)", y_test, lr_prob,
                      cost_threshold(y_test, lr_prob))

    # Main model: XGBoost. The default rate (~22%) is only mildly imbalanced, so we do NOT
    # reweight classes - that keeps predicted probabilities close to real default rates.
    # Imbalance is handled by the F1-optimised threshold and PR-AUC instead of accuracy.
    xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="auc", random_state=SEED, n_jobs=-1)
    xgb.fit(X_train, y_train)
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    thr = cost_threshold(y_test, xgb_prob)
    xgb_res = evaluate("XGBoost", y_test, xgb_prob, thr)
    print(classification_report(y_test, (xgb_prob >= thr).astype(int)))

    # Risk bands from predicted probability of default
    low, high = float(np.quantile(xgb_prob, 0.50)), float(np.quantile(xgb_prob, 0.80))
    bands = pd.Series([assign_band(p, low, high) for p in xgb_prob], index=y_test.index)
    band_table = (pd.DataFrame({"band": bands, "default": y_test})
                  .groupby("band")["default"].agg(["count", "mean"])
                  .rename(columns={"mean": "actual_default_rate"})
                  .reindex(["Low", "Medium", "High"]))
    print("\nRisk bands (actual default rate should rise Low -> High):")
    print(band_table)

    # SHAP explainability
    sample = X_test.sample(min(2000, len(X_test)), random_state=SEED)
    sv = shap.TreeExplainer(xgb).shap_values(sample)
    shap.summary_plot(sv, sample, show=False)
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=150)
    plt.close()

    # ROC curves
    plt.figure(figsize=(5, 4))
    for name, prob in [("LogReg", lr_prob), ("XGBoost", xgb_prob)]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.legend(); plt.title("ROC curve"); plt.tight_layout()
    plt.savefig(OUT / "roc_curve.png", dpi=150)
    plt.close()

    # Save artifacts for the app
    joblib.dump({"model": xgb, "features": list(X.columns), "stats": stats,
                 "band_low": low, "band_high": high, "threshold": thr}, OUT / "model.joblib")
    (OUT / "metrics.json").write_text(json.dumps([lr_res, xgb_res], indent=2))
    band_table.to_csv(OUT / "risk_bands.csv")
    print("\nSaved model, metrics and plots to ./outputs")


if __name__ == "__main__":
    main()
