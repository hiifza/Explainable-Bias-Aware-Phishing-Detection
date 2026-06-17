"""
src/training/cross_validation.py
----------------------------------
Stratified 5-fold cross-validation for model evaluation.

CV is run on the TRAINING set only using the already-preprocessed data
(train/test split was done in M3.1).  The test set remains untouched.

Public API
----------
    cross_validate_model(model, X, y, n_splits, scoring) -> dict
    format_cv_result(cv_result)                          -> str
"""

import sys
import time
from copy    import deepcopy
from pathlib import Path
from typing  import Any, Optional

import numpy  as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics         import (
    make_scorer, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Metrics computed during every CV fold
# Use response_method (sklearn 1.4+) instead of the deprecated needs_proba
try:
    from sklearn.metrics import make_scorer as _ms
    import inspect as _inspect
    _sig = _inspect.signature(_ms)
    _USE_RESPONSE_METHOD = "response_method" in _sig.parameters
except Exception:
    _USE_RESPONSE_METHOD = False


def _roc_auc_scorer():
    """Return a ROC AUC scorer compatible with sklearn 1.3 and 1.4+."""
    if _USE_RESPONSE_METHOD:
        return make_scorer(roc_auc_score, response_method="predict_proba")
    else:
        return make_scorer(roc_auc_score, needs_proba=True)


CV_SCORING: dict[str, Any] = {
    "accuracy" : make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, average="weighted",
                             zero_division=0),
    "recall"   : make_scorer(recall_score,    average="weighted",
                             zero_division=0),
    "f1"       : make_scorer(f1_score,        average="weighted",
                             zero_division=0),
    "roc_auc"  : _roc_auc_scorer(),
}


def cross_validate_model(
    model    : Any,
    X        : pd.DataFrame | np.ndarray,
    y        : pd.Series    | np.ndarray,
    n_splits : int  = 5,
    n_jobs   : int  = -1,
) -> dict:
    """
    Run stratified k-fold cross-validation on the TRAINING set.

    A deep copy of the model is used so the original remains unfitted
    and can be trained on the full training set afterwards.

    Parameters
    ----------
    model    : unfitted sklearn-compatible estimator
    X        : training feature matrix
    y        : training labels
    n_splits : number of CV folds (default 5)
    n_jobs   : parallelism for CV (default -1 = all cores)

    Returns
    -------
    dict  keys: cv_mean_{metric}, cv_std_{metric} for each metric,
               plus cv_n_splits, cv_elapsed_s
    """
    logger.info(
        f"Starting {n_splits}-fold stratified CV  "
        f"(n={len(y):,}, n_jobs={n_jobs}) …"
    )

    cv     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model_copy = deepcopy(model)

    t0 = time.perf_counter()
    cv_scores = cross_validate(
        estimator  = model_copy,
        X          = X,
        y          = y,
        cv         = cv,
        scoring    = CV_SCORING,
        n_jobs     = n_jobs,
        return_train_score = False,
        error_score = "raise",
    )
    elapsed = time.perf_counter() - t0

    result: dict = {"cv_n_splits": n_splits, "cv_elapsed_s": round(elapsed, 2)}

    for metric in CV_SCORING:
        key    = f"test_{metric}"
        scores = cv_scores[key]
        mean   = float(np.mean(scores))
        std    = float(np.std(scores))
        result[f"cv_mean_{metric}"] = round(mean, 6)
        result[f"cv_std_{metric}"]  = round(std,  6)
        logger.info(
            f"  CV {metric:<10}: {mean:.6f} ± {std:.6f}"
        )

    logger.info(
        f"CV complete in {elapsed:.1f}s  |  "
        f"ROC AUC: {result['cv_mean_roc_auc']:.4f} "
        f"± {result['cv_std_roc_auc']:.4f}"
    )
    return result


def format_cv_result(cv_result: dict) -> str:
    """Return a human-readable multi-line string of CV results."""
    lines = [f"  {cv_result['cv_n_splits']}-Fold Cross-Validation:"]
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for m in metrics:
        mean = cv_result.get(f"cv_mean_{m}", float("nan"))
        std  = cv_result.get(f"cv_std_{m}",  float("nan"))
        lines.append(f"    {m:<12}: {mean:.6f} ± {std:.6f}")
    lines.append(f"  Elapsed: {cv_result.get('cv_elapsed_s', 0):.1f}s")
    return "\n".join(lines)
