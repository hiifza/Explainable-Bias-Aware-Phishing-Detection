"""
src/evaluation/model_selector.py
----------------------------------
Implements the three-criterion final model selection logic and
exposes the FINAL_DEPLOYMENT_MODEL along with all downstream
interface objects.

Selection logic
---------------
1. Primary   : ROC AUC   — highest wins
2. Secondary : F1 Score  — breaks ties
3. Tertiary  : Calibration Quality — breaks further ties

The Track B winner is designated FINAL_DEPLOYMENT_MODEL and is
the single model consumed by SHAP, LIME, Bias Analysis, and Blind
Spot Analysis.

Public API
----------
    select_best_model(eval_results, track)             -> dict
    select_final_deployment_model(eval_results)        -> dict
    build_deployment_interface(selection, loaded_data) -> dict
    print_selection_report(selection_dict)
"""

import sys
from pathlib import Path
from typing  import Any

import numpy  as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger          import get_logger
from src.training.model_registry import MODEL_DISPLAY_NAMES

logger = get_logger(__name__)

PRIMARY_METRIC    = "roc_auc"
SECONDARY_METRIC  = "f1"
TERTIARY_METRIC   = "calibration_quality"   # higher = better calibrated


def select_best_model(
    eval_results: list[dict],
    track       : str,
) -> dict:
    """
    Select the best-performing model for a given track using three criteria.

    Parameters
    ----------
    eval_results : list of result dicts from evaluator.run_full_evaluation()
    track        : "A" or "B"

    Returns
    -------
    dict  keys: model_id, model_name, track,
               primary_score, secondary_score, tertiary_score,
               fitted_model, metrics, rationale
    """
    track_results = [r for r in eval_results
                     if r.get("track", "").upper() == track.upper()]

    if not track_results:
        raise ValueError(f"No evaluation results for Track {track}")

    # Sort by three criteria descending
    def _sort_key(r):
        p = float(r.get(PRIMARY_METRIC,   0))
        s = float(r.get(SECONDARY_METRIC, 0))
        t = float(r.get(TERTIARY_METRIC,  0))
        return (p, s, t)

    ranked = sorted(track_results, key=_sort_key, reverse=True)
    best   = ranked[0]

    model_name = best["model"]
    model_id   = best.get("model_id") or next(
        (k for k, v in MODEL_DISPLAY_NAMES.items() if v == model_name),
        model_name.lower().replace(" ", "_"),
    )

    # Build rationale string
    rationale_parts = [
        f"Highest ROC AUC on Track {track.upper()} test set: "
        f"{best.get(PRIMARY_METRIC, 0):.6f}",
    ]
    # Check if tie on primary
    runner_up_auc = float(ranked[1].get(PRIMARY_METRIC, 0)) if len(ranked) > 1 else 0
    if abs(best.get(PRIMARY_METRIC, 0) - runner_up_auc) < 1e-4:
        rationale_parts.append(
            f"Tie-broke by F1: {best.get(SECONDARY_METRIC, 0):.6f}"
        )
    rationale_parts.append(
        f"Calibration quality: {best.get(TERTIARY_METRIC, 0):.4f}"
    )

    selection = {
        "model_id"        : model_id,
        "model_name"      : model_name,
        "track"           : track.upper(),
        "primary_score"   : float(best.get(PRIMARY_METRIC,   0)),
        "secondary_score" : float(best.get(SECONDARY_METRIC, 0)),
        "tertiary_score"  : float(best.get(TERTIARY_METRIC,  0)),
        "fitted_model"    : best.get("fitted_model"),
        "metrics"         : {k: v for k, v in best.items()
                              if k not in ("y_true","y_pred","y_proba",
                                           "fitted_model","confusion_matrix")},
        "rationale"       : " | ".join(rationale_parts),
        "all_ranked"      : [r["model"] for r in ranked],
    }

    logger.info(f"Best Track {track.upper()}: {model_name}")
    logger.info(f"  ROC AUC  : {selection['primary_score']:.6f}")
    logger.info(f"  F1 Score : {selection['secondary_score']:.6f}")
    logger.info(f"  Calibrat : {selection['tertiary_score']:.6f}")
    logger.info(f"  Rationale: {selection['rationale']}")

    return selection


def select_final_deployment_model(eval_results: list[dict]) -> dict:
    """
    Select best models for both tracks and designate the Track B winner
    as FINAL_DEPLOYMENT_MODEL.

    Returns
    -------
    dict  keys:
        best_A              : selection dict for Track A
        best_B              : selection dict for Track B
        FINAL_DEPLOYMENT_MODEL : fitted estimator (Track B winner)
        deployment_model_name  : str
        deployment_model_id    : str
        leakage_impact         : float  (Track A AUC − Track B AUC)
    """
    best_A = select_best_model(eval_results, "A")
    best_B = select_best_model(eval_results, "B")

    leakage_impact = best_A["primary_score"] - best_B["primary_score"]

    result = {
        "best_A"                 : best_A,
        "best_B"                 : best_B,
        "FINAL_DEPLOYMENT_MODEL" : best_B["fitted_model"],
        "deployment_model_name"  : best_B["model_name"],
        "deployment_model_id"    : best_B["model_id"],
        "leakage_impact_auc"     : round(float(leakage_impact), 6),
    }

    sep = "=" * 55
    logger.info(sep)
    logger.info("FINAL DEPLOYMENT MODEL SELECTION")
    logger.info(sep)
    logger.info(f"  Track A winner : {best_A['model_name']}")
    logger.info(f"  Track B winner : {best_B['model_name']}  ← FINAL_DEPLOYMENT_MODEL")
    logger.info(f"  Leakage impact : {leakage_impact:+.6f} AUC (Track A − Track B)")
    logger.info(sep)

    return result


def build_deployment_interface(
    selection  : dict,
    loaded_data: dict,
) -> dict:
    """
    Assemble the exact objects that every downstream module will receive.

    Parameters
    ----------
    selection   : dict from select_final_deployment_model()
    loaded_data : dict from evaluator.load_all_models_and_data()

    Returns
    -------
    dict  — the authoritative downstream interface contract
    """
    X_test_B  = loaded_data["X_test_B"]
    X_train_B = pd.read_csv(ROOT / "data" / "processed" / "track_B" / "X_train.csv")
    y_test    = loaded_data["y_test"]

    # Load best Track B model to get predictions
    best_model_B  = selection["FINAL_DEPLOYMENT_MODEL"]
    y_pred_B      = best_model_B.predict(X_test_B)
    y_proba_B     = best_model_B.predict_proba(X_test_B)[:, 1]

    interface = {
        # ── SHAP (M7.1) ──────────────────────────────────────────────────────
        "SHAP_model"          : best_model_B,            # TreeExplainer / LinearExplainer
        "SHAP_X_train"        : X_train_B,               # background dataset
        "SHAP_X_test"         : X_test_B,                # instances to explain
        "SHAP_feature_names"  : list(X_test_B.columns),  # 56 feature names

        # ── LIME (M8.1) ──────────────────────────────────────────────────────
        "LIME_predict_fn"     : best_model_B.predict_proba,  # callable
        "LIME_X_train_np"     : X_train_B.values,            # np.ndarray background
        "LIME_X_test_np"      : X_test_B.values,             # np.ndarray to explain
        "LIME_feature_names"  : list(X_test_B.columns),

        # ── Bias Analysis (M9) ───────────────────────────────────────────────
        "BIAS_model"          : best_model_B,
        "BIAS_X_test"         : X_test_B,
        "BIAS_y_test"         : y_test,
        "BIAS_y_pred"         : y_pred_B,
        "BIAS_y_proba"        : y_proba_B,

        # ── Blind Spot Analysis (M10) ────────────────────────────────────────
        "BLINDSPOT_model"     : best_model_B,
        "BLINDSPOT_X_test"    : X_test_B,
        "BLINDSPOT_y_test"    : y_test,
        "BLINDSPOT_y_pred"    : y_pred_B,
        "BLINDSPOT_y_proba"   : y_proba_B,

        # ── Shared metadata ───────────────────────────────────────────────────
        "FINAL_DEPLOYMENT_MODEL"  : best_model_B,
        "feature_names_B"         : list(X_test_B.columns),
        "deployment_model_name"   : selection["deployment_model_name"],
        "deployment_model_id"     : selection["deployment_model_id"],
        "n_test"                  : len(y_test),
        "n_features"              : X_test_B.shape[1],
    }

    logger.info("Deployment interface assembled:")
    logger.info(f"  FINAL_DEPLOYMENT_MODEL : {selection['deployment_model_name']}")
    logger.info(f"  feature_names_B        : {X_test_B.shape[1]} features")
    logger.info(f"  SHAP_X_train shape     : {X_train_B.shape}")
    logger.info(f"  SHAP_X_test  shape     : {X_test_B.shape}")

    return interface


def print_selection_report(selection: dict) -> None:
    """Pretty-print the selection results to logger."""
    sep = "─" * 55
    logger.info(sep)
    logger.info("MODEL SELECTION REPORT")
    logger.info(sep)
    for track, sel in [("A", selection["best_A"]),
                        ("B", selection["best_B"])]:
        logger.info(f"Track {track}: {sel['model_name']}")
        logger.info(f"  Primary   ROC AUC   : {sel['primary_score']:.6f}")
        logger.info(f"  Secondary F1        : {sel['secondary_score']:.6f}")
        logger.info(f"  Tertiary  Calibrat  : {sel['tertiary_score']:.6f}")
        logger.info(f"  Full rank           : {sel['all_ranked']}")
        logger.info(f"  Rationale           : {sel['rationale']}")
    logger.info(f"Leakage impact (A-B AUC) : "
                f"{selection.get('leakage_impact_auc', 0):+.6f}")
    logger.info(f"FINAL_DEPLOYMENT_MODEL   : {selection['deployment_model_name']}")
    logger.info(sep)
