"""
src/training/trainer.py
------------------------
Single-model training, test-set evaluation, and result aggregation.

Public API
----------
    load_track_data(track, processed_dir)        -> (X_train, X_test, y_train, y_test)
    train_model(model, X_train, y_train)         -> fitted model
    evaluate_model(model, X_test, y_test)        -> dict
    train_and_evaluate(model_id, track, ...)     -> dict
    run_track_training(track, ...)               -> list[dict]
"""

import sys
import time
from pathlib import Path
from typing  import Any, Optional

import numpy  as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger              import get_logger
from src.utils.metrics_logger      import MetricsLogger
from src.training.cross_validation import cross_validate_model
from src.training.model_registry   import MODEL_DISPLAY_NAMES
from src.training.model_saver      import save_model

logger = get_logger(__name__)

TARGET = "label"
PROCESSED_DIR   = Path("data/processed")
MODELS_OUT_DIR  = Path("outputs/models")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_track_data(
    track        : str,
    processed_dir: str | Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load the preprocessed train/test split for a given track.

    Parameters
    ----------
    track         : "A" or "B"
    processed_dir : base directory containing track_A/ and track_B/

    Returns
    -------
    (X_train, X_test, y_train, y_test)
    """
    track  = track.upper()
    base   = Path(processed_dir)
    t_dir  = base / f"track_{track}"

    paths = {
        "X_train": t_dir  / "X_train.csv",
        "X_test" : t_dir  / "X_test.csv",
        "y_train": base   / "y_train.csv",
        "y_test" : base   / "y_test.csv",
    }

    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(
                f"Preprocessed data not found: {p}. "
                "Run M3.1 notebook first."
            )

    logger.info(f"Loading Track {track} preprocessed data …")
    X_train = pd.read_csv(paths["X_train"])
    X_test  = pd.read_csv(paths["X_test"])
    y_train = pd.read_csv(paths["y_train"])[TARGET]
    y_test  = pd.read_csv(paths["y_test"])[TARGET]

    logger.info(
        f"Track {track} loaded: "
        f"X_train={X_train.shape}  X_test={X_test.shape}  "
        f"y_train={len(y_train):,}  y_test={len(y_test):,}"
    )
    return X_train, X_test, y_train, y_test


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    model  : Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Any, float]:
    """
    Fit *model* on training data and record wall-clock time.

    Parameters
    ----------
    model   : unfitted estimator
    X_train : training feature matrix
    y_train : training labels

    Returns
    -------
    (fitted_model, training_time_seconds)
    """
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info(
        f"Training complete: {type(model).__name__}  "
        f"({len(y_train):,} rows, {elapsed:.2f}s)"
    )
    return model, round(elapsed, 3)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    model : Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Compute all five metrics on the held-out test set.

    Parameters
    ----------
    model  : fitted estimator
    X_test : test feature matrix
    y_test : test labels

    Returns
    -------
    dict  keys: accuracy, precision, recall, f1, roc_auc,
               n_test, confusion_matrix
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy" : round(float(accuracy_score(y_test, y_pred)),           6),
        "precision": round(float(precision_score(y_test, y_pred,
                                  average="weighted", zero_division=0)),      6),
        "recall"   : round(float(recall_score(y_test, y_pred,
                                  average="weighted", zero_division=0)),      6),
        "f1"       : round(float(f1_score(y_test, y_pred,
                                  average="weighted", zero_division=0)),      6),
        "roc_auc"  : round(float(roc_auc_score(y_test, y_proba)),            6),
        "n_test"   : int(len(y_test)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    logger.info(
        f"Test evaluation: accuracy={metrics['accuracy']:.4f}  "
        f"f1={metrics['f1']:.4f}  roc_auc={metrics['roc_auc']:.4f}"
    )
    return metrics


# ── Per-model orchestrator ────────────────────────────────────────────────────

def train_and_evaluate(
    model_id     : str,
    model        : Any,
    track        : str,
    X_train      : pd.DataFrame,
    X_test       : pd.DataFrame,
    y_train      : pd.Series,
    y_test       : pd.Series,
    run_cv       : bool = True,
    cv_splits    : int  = 5,
    save_dir     : Optional[str | Path] = None,
) -> dict:
    """
    Full pipeline for one model: train → evaluate → CV → (optionally save).

    Parameters
    ----------
    model_id  : registry identifier
    model     : unfitted estimator
    track     : "A" or "B"
    X_train, X_test, y_train, y_test : split data
    run_cv    : whether to run 5-fold CV (default True)
    cv_splits : number of CV folds
    save_dir  : if provided, save fitted model to save_dir/model_id.pkl

    Returns
    -------
    dict  with all metrics, CV results, timing, and metadata
    """
    display  = MODEL_DISPLAY_NAMES.get(model_id, model_id)
    mlogger  = MetricsLogger(display, track)
    sep      = "─" * 50

    logger.info(sep)
    logger.info(f"Training: {display}  |  Track {track.upper()}")
    logger.info(sep)

    # 1. Train
    model, train_time = train_model(model, X_train, y_train)
    mlogger.log_training_time(train_time)

    # 2. Test evaluation
    test_metrics = evaluate_model(model, X_test, y_test)
    mlogger.log_metrics(test_metrics, stage="test",
                        n_samples=test_metrics["n_test"])

    # 3. Cross-validation (on training set)
    cv_result = {}
    if run_cv:
        from copy import deepcopy
        cv_result = cross_validate_model(
            deepcopy(model), X_train, y_train, n_splits=cv_splits
        )
        mlogger.log_cv(cv_result, n_splits=cv_splits)

    # 4. Optionally save
    if save_dir:
        save_model(model, Path(save_dir) / f"{model_id}.pkl")

    # 5. Assemble full result record
    result = {
        "model"          : display,
        "model_id"       : model_id,
        "track"          : track.upper(),
        "training_time_s": train_time,
        **{k: v for k, v in test_metrics.items()
           if k not in ("confusion_matrix", "n_test")},
        **cv_result,
        "fitted_model"   : model,
        "confusion_matrix": test_metrics.get("confusion_matrix"),
    }

    mlogger.log_summary(result)
    return result


# ── Track-level orchestrator ─────────────────────────────────────────────────

def run_track_training(
    track        : str,
    models_dict  : dict[str, Any],
    processed_dir: str | Path = PROCESSED_DIR,
    models_out   : str | Path = MODELS_OUT_DIR,
    run_cv       : bool = True,
    cv_splits    : int  = 5,
) -> list[dict]:
    """
    Train and evaluate all models for one track.

    Parameters
    ----------
    track         : "A" or "B"
    models_dict   : dict[model_id -> unfitted model] from model_registry
    processed_dir : preprocessed CSV directory
    models_out    : where to save .pkl files
    run_cv        : run 5-fold CV (default True)
    cv_splits     : CV folds

    Returns
    -------
    list[dict]  — one result dict per model
    """
    sep = "=" * 55
    logger.info(sep)
    logger.info(f"TRACK {track.upper()} — TRAINING {len(models_dict)} MODELS")
    logger.info(sep)

    X_train, X_test, y_train, y_test = load_track_data(track, processed_dir)
    save_dir = Path(models_out) / f"track_{track.upper()}"

    results = []
    for model_id, model in models_dict.items():
        result = train_and_evaluate(
            model_id  = model_id,
            model     = model,
            track     = track,
            X_train   = X_train,
            X_test    = X_test,
            y_train   = y_train,
            y_test    = y_test,
            run_cv    = run_cv,
            cv_splits = cv_splits,
            save_dir  = save_dir,
        )
        results.append(result)

    logger.info(sep)
    logger.info(f"TRACK {track.upper()} TRAINING COMPLETE — {len(results)} models")
    logger.info(sep)
    return results
