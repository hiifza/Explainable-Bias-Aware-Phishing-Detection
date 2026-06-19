"""src/bias/fairness_metrics.py — per-group metric computation and disparity analysis."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.utils.logger import get_logger
logger = get_logger(__name__)

DISPARITY_METRICS = ["accuracy","precision","recall","f1","roc_auc","fpr","fnr"]

def compute_group_metrics(y_true, y_pred, y_proba, group_label):
    n = len(y_true)
    if n == 0:
        return {"group": group_label, "n": 0,
                **{m: np.nan for m in ["accuracy","precision","recall","f1","roc_auc",
                                        "fpr","fnr","specificity","sensitivity",
                                        "phishing_count","legit_count","tp","tn","fp","fn"]}}
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); y_proba = np.asarray(y_proba)
    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec  = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1   = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    try:    auc = float(roc_auc_score(y_true, y_proba))
    except: auc = np.nan
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel() if cm.shape==(2,2) else (0,0,0,0)
    fpr = fp/max(fp+tn,1); fnr = fn/max(fn+tp,1)
    return {
        "group": group_label, "n": int(n),
        "phishing_count": int((y_true==0).sum()), "legit_count": int((y_true==1).sum()),
        "accuracy": round(acc,6), "precision": round(prec,6), "recall": round(rec,6),
        "f1": round(f1,6), "roc_auc": round(auc,6) if not np.isnan(auc) else np.nan,
        "fpr": round(fpr,6), "fnr": round(fnr,6),
        "specificity": round(tn/max(tn+fp,1),6), "sensitivity": round(tp/max(tp+fn,1),6),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

def compute_all_group_metrics(groups_df, y_true, y_pred, y_proba, group_col, dim_name):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred); y_proba=np.asarray(y_proba)
    rows = []
    for grp_val in sorted(groups_df[group_col].unique()):
        mask = (groups_df[group_col]==grp_val).values
        if mask.sum()<10: continue
        row = compute_group_metrics(y_true[mask], y_pred[mask], y_proba[mask], str(grp_val))
        row["dimension"] = dim_name; rows.append(row)
    df = pd.DataFrame(rows).reset_index(drop=True)
    logger.info(f"[{dim_name}] {len(df)} groups  n_total={len(y_true):,}")
    return df

def compute_disparity(metrics_df, dim_name):
    rows = []
    for metric in DISPARITY_METRICS:
        if metric not in metrics_df.columns: continue
        sub = metrics_df.dropna(subset=[metric])
        if sub.empty: continue
        if metric in ("fpr","fnr"):
            worst_idx=sub[metric].idxmax(); best_idx=sub[metric].idxmin()
        else:
            worst_idx=sub[metric].idxmin(); best_idx=sub[metric].idxmax()
        best_val=float(sub.loc[best_idx,metric]); worst_val=float(sub.loc[worst_idx,metric])
        rows.append({"metric": metric, "dimension": dim_name,
                     "best_group": str(sub.loc[best_idx,"group"]), "best_value": round(best_val,6),
                     "worst_group": str(sub.loc[worst_idx,"group"]), "worst_value": round(worst_val,6),
                     "disparity": round(abs(best_val-worst_val),6)})
    return pd.DataFrame(rows)
