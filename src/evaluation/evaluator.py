"""
src/evaluation/evaluator.py
-----------------------------
Main orchestrator for Module M6.1.

Loads all trained models, runs full evaluation on test sets,
generates all plots, saves all CSVs, and returns the complete
evaluation results dict used by model_selector and evaluation_report.

Public API
----------
    load_all_models_and_data(models_dir, processed_dir) -> dict
    run_single_evaluation(model, X_test, y_test, model_name, track, training_time) -> dict
    run_full_evaluation(models_dir, processed_dir, reports_dir, plots_dir) -> dict
"""

import sys
from pathlib import Path
from typing  import Any, Optional

import numpy  as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger                  import get_logger
from src.training.model_saver          import load_all_models
from src.training.model_registry       import MODEL_IDS, MODEL_DISPLAY_NAMES
from src.evaluation.metrics            import compute_full_metrics, build_metrics_dataframe
from src.evaluation.confusion_analysis import (
    plot_all_confusion_matrices, plot_combined_confusion_grid
)
from src.evaluation.roc_analysis       import (
    plot_combined_roc, plot_roc_all_tracks, plot_combined_pr, plot_auc_ranking
)
from src.evaluation.calibration_analysis import (
    plot_calibration_curves, plot_probability_distribution,
    plot_calibration_comparison, calibration_quality_score,
)

logger = get_logger(__name__)

TARGET        = "label"
MODELS_DIR    = Path("outputs/models")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR   = Path("outputs/reports")
PLOTS_BASE    = Path("outputs/plots/evaluation")


# ── Data / model loading ──────────────────────────────────────────────────────

def load_all_models_and_data(
    models_dir   : str | Path = MODELS_DIR,
    processed_dir: str | Path = PROCESSED_DIR,
) -> dict:
    """
    Load all 8 trained models and both track test sets.

    Returns
    -------
    dict  keys:
        models_A, models_B         : dict[model_id -> fitted estimator]
        X_test_A, X_test_B         : pd.DataFrame
        y_test                     : pd.Series (shared)
        feature_names_A, _B        : list[str]
        training_times             : dict (from benchmark CSV if available)
    """
    models_dir    = Path(models_dir)
    processed_dir = Path(processed_dir)

    logger.info("Loading trained models ...")
    models_A = load_all_models(models_dir, "A")
    models_B = load_all_models(models_dir, "B")
    logger.info(f"  Track A: {len(models_A)} models  Track B: {len(models_B)} models")

    logger.info("Loading test data ...")
    X_test_A = pd.read_csv(processed_dir / "track_A" / "X_test.csv")
    X_test_B = pd.read_csv(processed_dir / "track_B" / "X_test.csv")
    y_test   = pd.read_csv(processed_dir / "y_test.csv")[TARGET]
    logger.info(
        f"  X_test_A: {X_test_A.shape}  X_test_B: {X_test_B.shape}  "
        f"y_test: {len(y_test):,}"
    )

    # Load training times from benchmark CSV if available
    training_times: dict[str, float] = {}
    bench_csv = ROOT / "outputs" / "reports" / "model_benchmark.csv"
    if bench_csv.exists():
        bench = pd.read_csv(bench_csv)
        if "training_time_s" in bench.columns:
            for _, row in bench.iterrows():
                key = f"{row['model']}_{row['track']}"
                training_times[key] = float(row.get("training_time_s", 0))

    return {
        "models_A"       : models_A,
        "models_B"       : models_B,
        "X_test_A"       : X_test_A,
        "X_test_B"       : X_test_B,
        "y_test"         : y_test,
        "feature_names_A": list(X_test_A.columns),
        "feature_names_B": list(X_test_B.columns),
        "training_times" : training_times,
    }


# ── Single model evaluation ───────────────────────────────────────────────────

def run_single_evaluation(
    model        : Any,
    X_test       : pd.DataFrame,
    y_test       : pd.Series,
    model_name   : str,
    track        : str,
    training_time: float = 0.0,
) -> dict:
    """
    Evaluate one model and return a complete result dict.

    Returns
    -------
    dict  with all metrics + raw predictions + probabilities + metadata
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    full_metrics = compute_full_metrics(
        y_true        = y_test.values,
        y_pred        = y_pred,
        y_proba       = y_proba,
        model_name    = model_name,
        track         = track,
        training_time = training_time,
    )

    # Add calibration quality
    try:
        cal_q = calibration_quality_score(y_test.values, y_proba)
    except Exception:
        cal_q = float("nan")
    full_metrics["calibration_quality"] = round(cal_q, 6)

    return {
        **full_metrics,
        "y_true"  : y_test.values,
        "y_pred"  : y_pred,
        "y_proba" : y_proba,
        "model_id": next(
            (k for k, v in MODEL_DISPLAY_NAMES.items() if v == model_name),
            model_name.lower().replace(" ", "_"),
        ),
        "fitted_model": model,
    }


# ── Full evaluation orchestrator ──────────────────────────────────────────────

def run_full_evaluation(
    models_dir   : str | Path = MODELS_DIR,
    processed_dir: str | Path = PROCESSED_DIR,
    reports_dir  : str | Path = REPORTS_DIR,
    plots_dir    : str | Path = PLOTS_BASE,
) -> dict:
    """
    Evaluate all 8 models, generate all plots, save all CSVs.

    Parameters
    ----------
    models_dir, processed_dir, reports_dir, plots_dir : paths

    Returns
    -------
    dict  keys:
        eval_results     : list[dict]  — one per model/track
        metrics_df       : pd.DataFrame
        error_df         : pd.DataFrame
        best_A, best_B   : (model_id, result_dict) tuples
        final_deployment : result dict for Track B winner
    """
    sep = "=" * 55
    logger.info(sep)
    logger.info("M6.1 — FULL MODEL EVALUATION")
    logger.info(sep)

    reports_dir = Path(reports_dir)
    plots_dir   = Path(plots_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cm_dir   = plots_dir / "confusion_matrices"
    roc_dir  = plots_dir / "roc"
    pr_dir   = plots_dir / "pr_curves"
    cal_dir  = plots_dir / "calibration"
    for d in [cm_dir, roc_dir, pr_dir, cal_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load everything
    loaded = load_all_models_and_data(models_dir, processed_dir)

    # 2. Evaluate all models
    eval_results: list[dict] = []

    for track, models, X_test in [
        ("A", loaded["models_A"], loaded["X_test_A"]),
        ("B", loaded["models_B"], loaded["X_test_B"]),
    ]:
        for model_id, model in models.items():
            display      = MODEL_DISPLAY_NAMES.get(model_id, model_id)
            t_time_key   = f"{display}_{track}"
            training_t   = loaded["training_times"].get(t_time_key, 0.0)

            logger.info(f"Evaluating: {display} | Track {track}")
            result = run_single_evaluation(
                model         = model,
                X_test        = X_test,
                y_test        = loaded["y_test"],
                model_name    = display,
                track         = track,
                training_time = training_t,
            )
            eval_results.append(result)

    logger.info(f"Evaluated {len(eval_results)} models")

    # 3. Build metrics DataFrames
    metrics_df = build_metrics_dataframe(eval_results)
    metrics_df.to_csv(reports_dir / "evaluation_metrics.csv", index=False)
    logger.info("Saved: evaluation_metrics.csv")

    # Error analysis subset
    error_cols = ["model","track","fpr","fnr","specificity","sensitivity",
                  "tp","tn","fp","fn","calibration_quality"]
    err_cols   = [c for c in error_cols if c in metrics_df.columns]
    error_df   = metrics_df[err_cols].copy()
    error_df.to_csv(reports_dir / "error_analysis.csv", index=False)
    logger.info("Saved: error_analysis.csv")

    # 4. Confusion matrix plots
    logger.info("Generating confusion matrix plots ...")
    plot_all_confusion_matrices(eval_results, cm_dir)
    for track in ["A", "B"]:
        plot_combined_confusion_grid(eval_results, track, cm_dir)

    # 5. ROC plots
    logger.info("Generating ROC plots ...")
    for track in ["A", "B"]:
        plot_combined_roc(eval_results, track, roc_dir)
    plot_roc_all_tracks(eval_results, roc_dir)
    plot_auc_ranking(metrics_df, roc_dir)

    # 6. PR plots
    logger.info("Generating PR curve plots ...")
    for track in ["A", "B"]:
        plot_combined_pr(eval_results, track, pr_dir)

    # 7. Calibration plots
    logger.info("Generating calibration plots ...")
    for track in ["A", "B"]:
        plot_calibration_curves(eval_results, track, cal_dir)
        plot_probability_distribution(eval_results, track, cal_dir)
    plot_calibration_comparison(eval_results, cal_dir)

    logger.info(sep)
    logger.info("EVALUATION COMPLETE")
    logger.info(sep)

    return {
        "eval_results"         : eval_results,
        "metrics_df"           : metrics_df,
        "error_df"             : error_df,
        "loaded"               : loaded,
    }
