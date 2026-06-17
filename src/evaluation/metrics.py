"""
src/evaluation/metrics.py
--------------------------
Computes all evaluation metrics for phishing detection models.

Metrics
-------
1.  Accuracy
2.  Precision         (weighted)
3.  Recall            (weighted)
4.  F1 Score          (weighted)
5.  ROC AUC
6.  PR AUC            (average precision)
7.  Matthews Correlation Coefficient (MCC)
8.  Balanced Accuracy
9.  Brier Score

Error metrics:
  - FPR  (False Positive Rate  = FP / (FP + TN))
  - FNR  (False Negative Rate  = FN / (FN + TP))
  - Specificity (TN / (TN + FP))
  - Sensitivity (TP / (TP + FN))  = Recall for positive class

Public API
----------
    compute_all_metrics(y_true, y_pred, y_proba)   -> dict
    compute_error_metrics(y_true, y_pred)           -> dict
    compute_full_metrics(y_true, y_pred, y_proba)   -> dict  (combined)
"""

import sys
from pathlib import Path
from typing  import Any

import numpy  as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score,
    brier_score_loss, confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Primary metrics ───────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true : np.ndarray | pd.Series,
    y_pred : np.ndarray | pd.Series,
    y_proba: np.ndarray,
) -> dict:
    """
    Compute the full set of 9 classification metrics.

    Parameters
    ----------
    y_true  : true binary labels (0=phishing, 1=legitimate)
    y_pred  : predicted binary labels
    y_proba : predicted probabilities for class 1 (legitimate)

    Returns
    -------
    dict  with keys matching the 9 metric names
    """
    y_true  = np.asarray(y_true)
    y_pred  = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    metrics = {
        "accuracy"         : round(float(accuracy_score(y_true, y_pred)),            6),
        "precision"        : round(float(precision_score(y_true, y_pred,
                                          average="weighted", zero_division=0)),      6),
        "recall"           : round(float(recall_score(y_true, y_pred,
                                          average="weighted", zero_division=0)),      6),
        "f1"               : round(float(f1_score(y_true, y_pred,
                                          average="weighted", zero_division=0)),      6),
        "roc_auc"          : round(float(roc_auc_score(y_true, y_proba)),             6),
        "pr_auc"           : round(float(average_precision_score(y_true, y_proba)),  6),
        "mcc"              : round(float(matthews_corrcoef(y_true, y_pred)),          6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)),   6),
        "brier_score"      : round(float(brier_score_loss(y_true, y_proba)),         6),
    }
    return metrics


# ── Error metrics from confusion matrix ──────────────────────────────────────

def compute_error_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict:
    """
    Derive FPR, FNR, Specificity, Sensitivity from the confusion matrix.

    For binary classification where:
      label = 0 → Phishing  (negative class in standard convention)
      label = 1 → Legitimate (positive class)

    Returns
    -------
    dict  keys: tn, fp, fn, tp, fpr, fnr, specificity, sensitivity
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm      = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr         = fp / max(fp + tn, 1)   # legitimate sites flagged as phishing
    fnr         = fn / max(fn + tp, 1)   # phishing sites missed as legitimate
    specificity = tn / max(tn + fp, 1)   # true negative rate
    sensitivity = tp / max(tp + fn, 1)   # true positive rate (= recall for class 1)

    return {
        "tn"         : int(tn),
        "fp"         : int(fp),
        "fn"         : int(fn),
        "tp"         : int(tp),
        "fpr"        : round(float(fpr),         6),
        "fnr"        : round(float(fnr),         6),
        "specificity": round(float(specificity), 6),
        "sensitivity": round(float(sensitivity), 6),
    }


# ── Combined ──────────────────────────────────────────────────────────────────

def compute_full_metrics(
    y_true       : np.ndarray | pd.Series,
    y_pred       : np.ndarray | pd.Series,
    y_proba      : np.ndarray,
    model_name   : str = "",
    track        : str = "",
    training_time: float = 0.0,
) -> dict:
    """
    Compute all 9 primary metrics plus 4 error metrics in one call.

    Parameters
    ----------
    y_true, y_pred, y_proba : arrays as in compute_all_metrics()
    model_name   : for logging
    track        : "A" or "B"
    training_time: seconds (from M5 training)

    Returns
    -------
    dict  with 13 metrics + metadata fields
    """
    m1 = compute_all_metrics(y_true, y_pred, y_proba)
    m2 = compute_error_metrics(y_true, y_pred)

    full = {
        "model"        : model_name,
        "track"        : track.upper() if track else "",
        "training_time": round(training_time, 3),
        **m1,
        **m2,
    }

    if model_name:
        logger.info(
            f"[Track {track.upper()}] {model_name:<26} "
            f"AUC={m1['roc_auc']:.4f}  MCC={m1['mcc']:.4f}  "
            f"Brier={m1['brier_score']:.4f}  "
            f"FPR={m2['fpr']:.4f}  FNR={m2['fnr']:.4f}"
        )
    return full


# ── Build metrics DataFrame from list of results ──────────────────────────────

def build_metrics_dataframe(results: list[dict]) -> pd.DataFrame:
    """
    Convert a list of full_metrics dicts into a tidy DataFrame.

    Parameters
    ----------
    results : list of dicts from compute_full_metrics()

    Returns
    -------
    pd.DataFrame  — one row per model/track combination
    """
    ordered_cols = [
        "model", "track",
        "accuracy", "precision", "recall", "f1",
        "roc_auc", "pr_auc", "mcc", "balanced_accuracy", "brier_score",
        "fpr", "fnr", "specificity", "sensitivity",
        "tp", "tn", "fp", "fn",
        "training_time",
    ]
    df = pd.DataFrame(results)
    cols_present = [c for c in ordered_cols if c in df.columns]
    return df[cols_present].reset_index(drop=True)
