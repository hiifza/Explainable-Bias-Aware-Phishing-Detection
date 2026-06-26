"""
src/blindspots/blindspot_analyzer.py
--------------------------------------
Main M10 orchestrator — Failure Intelligence Engine.

Runs the complete blind spot analysis pipeline:
  1. Load model and predictions
  2. Extract all failure cases (4 tiers)
  3. Discover failure archetypes (auto-clustering)
  4. Compute severity scores + top-20 ranking
  5. Confidence Reliability Engine (Green/Yellow/Red zones)
  6. SHAP failure analysis
  7. LIME failure analysis
  8. SHAP-LIME reliability correlation
  9. Cluster visualisations
 10. Save all CSVs and generate HTML report

Public API
----------
    load_m10_inputs(processed_dir, models_dir, reports_dir) -> dict
    run_failure_intelligence_engine(inputs, shap_values, feature_names,
                                     agreement_df, severity_thresholds,
                                     plots_dir, reports_dir) -> dict
"""

import sys
from pathlib import Path
from typing  import Optional

import numpy  as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger                          import get_logger
from src.training.model_saver                  import load_all_models
from src.training.model_registry               import MODEL_DISPLAY_NAMES
from src.blindspots.failure_case_extractor     import extract_failure_cases
from src.blindspots.failure_archetype_discovery import discover_archetypes
from src.blindspots.severity_ranker            import compute_severity_scores, rank_blind_spots
from src.blindspots.uncertainty_analysis       import run_uncertainty_analysis
from src.blindspots.shap_failure_analysis      import run_shap_failure_analysis
from src.blindspots.lime_failure_analysis      import run_lime_failure_analysis
from src.blindspots.reliability_analysis       import run_reliability_analysis
from src.blindspots.cluster_visualization      import run_cluster_visualization

logger = get_logger(__name__)

TARGET = "label"


def load_m10_inputs(
    processed_dir: str | Path = "data/processed",
    models_dir   : str | Path = "outputs/models",
    reports_dir  : str | Path = "outputs/reports",
) -> dict:
    """
    Load the deployment model, test data, and pre-computed artefacts.

    Returns
    -------
    dict  keys: model, X_test_B, y_test, y_pred, y_proba,
               feature_names, agreement_df
    """
    processed_dir = Path(processed_dir)
    models_dir    = Path(models_dir)
    reports_dir   = Path(reports_dir)

    # Best Track B model
    bench   = pd.read_csv(reports_dir / "evaluation_metrics.csv")
    best_B  = bench[bench["track"] == "B"].sort_values("roc_auc", ascending=False).iloc[0]
    id_map  = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}
    best_id = id_map.get(best_B["model"], "logistic_regression")
    model   = load_all_models(models_dir, "B")[best_id]

    X_test_B      = pd.read_csv(processed_dir / "track_B" / "X_test.csv")
    y_test        = pd.read_csv(processed_dir / "y_test.csv")[TARGET].values
    y_pred        = model.predict(X_test_B)
    y_proba       = model.predict_proba(X_test_B)[:, 1]
    feature_names = list(X_test_B.columns)

    # Load agreement CSV from M8.1 if available
    agree_path = reports_dir / "shap_lime_agreement.csv"
    agreement_df = pd.read_csv(agree_path) if agree_path.exists() else pd.DataFrame()

    logger.info(
        f"M10 inputs: {type(model).__name__}  "
        f"X_test={X_test_B.shape}  "
        f"errors={(y_pred!=y_test).sum()}  "
        f"agreement_rows={len(agreement_df)}"
    )

    return {
        "model"       : model,
        "X_test_B"    : X_test_B,
        "y_test"      : y_test,
        "y_pred"      : y_pred,
        "y_proba"     : y_proba,
        "feature_names": feature_names,
        "agreement_df": agreement_df,
    }


def run_failure_intelligence_engine(
    inputs          : dict,
    shap_values     : np.ndarray,
    feature_names   : list[str],
    agreement_df    : Optional[pd.DataFrame] = None,
    plots_dir       : str | Path = "outputs/plots/blindspot",
    reports_dir     : str | Path = "outputs/reports",
    top_n_severity  : int = 20,
) -> dict:
    """
    Execute the complete Failure Intelligence Engine pipeline.

    Parameters
    ----------
    inputs          : from load_m10_inputs()
    shap_values     : np.ndarray (n_shap × n_features) from M7.1
    feature_names   : 56 Track B feature names
    agreement_df    : SHAP-LIME agreement DataFrame from M8.1
    plots_dir       : base plot directory
    reports_dir     : CSV/HTML output directory
    top_n_severity  : top-N blind spots to rank

    Returns
    -------
    dict  — complete M10 results for report generation
    """
    sep = "=" * 55
    logger.info(sep); logger.info("M10 — FAILURE INTELLIGENCE ENGINE"); logger.info(sep)

    plots_dir   = Path(plots_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_test    = inputs["X_test_B"]
    y_true    = inputs["y_test"]
    y_pred    = inputs["y_pred"]
    y_proba   = inputs["y_proba"]
    agree_df  = agreement_df if agreement_df is not None else inputs.get("agreement_df", pd.DataFrame())

    # ── Step 1: Extract failure cases ────────────────────────────────────────
    logger.info("\n── Step 1: Failure Case Extraction ──")
    fcs = extract_failure_cases(
        y_true, y_pred, y_proba, X_test,
        agreement_df=agree_df,
        feature_names=feature_names,
    )

    # ── Step 2: Confidence Reliability Engine ─────────────────────────────────
    logger.info("\n── Step 2: Confidence Reliability Engine ──")
    uncertainty_r = run_uncertainty_analysis(
        y_true, y_pred, y_proba,
        plots_dir=plots_dir / "confidence",
    )

    # ── Step 3: Severity scoring ──────────────────────────────────────────────
    logger.info("\n── Step 3: Severity Scoring ──")
    severity_df = compute_severity_scores(fcs, shap_values, feature_names)
    top_20_bs   = rank_blind_spots(severity_df, top_n=top_n_severity)

    # Save
    severity_df.to_csv(reports_dir / "blindspot_severity.csv", index=False)
    top_20_bs.to_csv(  reports_dir / "top20_blind_spots.csv",  index=False)
    logger.info(f"Saved: blindspot_severity.csv ({len(severity_df)} rows)")
    logger.info(f"Saved: top20_blind_spots.csv  ({len(top_20_bs)} rows)")

    # ── Step 4: Failure archetype discovery ───────────────────────────────────
    logger.info("\n── Step 4: Failure Archetype Discovery ──")
    archetype_r = discover_archetypes(fcs, X_test, feature_names, shap_values)
    archetype_r.archetype_df.to_csv(reports_dir / "failure_archetypes.csv", index=False)
    logger.info(f"Saved: failure_archetypes.csv ({len(archetype_r.archetype_df)} rows)")

    # ── Step 5: Cluster visualisations ────────────────────────────────────────
    logger.info("\n── Step 5: Cluster Visualisations ──")
    cluster_r = run_cluster_visualization(
        archetype_r, fcs, severity_df, X_test, feature_names,
        plots_dir=plots_dir / "clusters",
    )

    # ── Step 6: SHAP failure analysis ─────────────────────────────────────────
    logger.info("\n── Step 6: SHAP Failure Analysis ──")
    shap_fail_r = run_shap_failure_analysis(
        shap_values, feature_names, fcs, y_true, y_proba,
        plots_dir=plots_dir / "shap_failure",
        reports_dir=reports_dir,
    )

    # ── Step 7: LIME failure analysis ─────────────────────────────────────────
    logger.info("\n── Step 7: LIME Failure Analysis ──")
    lime_fail_r = run_lime_failure_analysis(
        agree_df, fcs,
        plots_dir=plots_dir / "failure_archetypes",
        reports_dir=reports_dir,
    )

    # ── Step 8: Reliability analysis ──────────────────────────────────────────
    logger.info("\n── Step 8: SHAP-LIME Reliability Correlation ──")
    reliability_r = run_reliability_analysis(
        agree_df, severity_df,
        zone_stats=uncertainty_r["zone_stats"],
        plots_dir=plots_dir / "reliability",
        reports_dir=reports_dir,
    )

    # ── Compile summary ───────────────────────────────────────────────────────
    n_total  = len(y_true)
    n_errors = int((y_pred != y_true).sum())
    zs       = uncertainty_r["zone_stats"]

    summary = {
        "n_test"            : n_total,
        "n_errors"          : n_errors,
        "n_fp"              : int(((y_pred==0)&(y_true==1)).sum()),
        "n_fn"              : int(((y_pred==1)&(y_true==0)).sum()),
        "error_rate"        : round(n_errors / max(n_total,1), 8),
        "n_failure_cases"   : len(fcs),
        "n_archetypes"      : archetype_r.n_clusters,
        "silhouette"        : round(archetype_r.silhouette_score, 4),
        "green_zone_n"      : zs.get("green",{}).get("n",0),
        "yellow_zone_n"     : zs.get("yellow",{}).get("n",0),
        "red_zone_n"        : zs.get("red",{}).get("n",0),
        "max_severity"      : round(float(severity_df["severity_score_norm"].max()),4)
                              if not severity_df.empty else 0.0,
        "mean_severity"     : round(float(severity_df["severity_score_norm"].mean()),4)
                              if not severity_df.empty else 0.0,
        "top_archetype"     : archetype_r.cluster_meta[0]["label"]
                              if archetype_r.cluster_meta else "—",
        "shap_top_failure_feat": shap_fail_r["comparison"]["top_failure_features"][0]
                              if shap_fail_r["comparison"].get("top_failure_features") else "—",
    }

    logger.info(sep)
    logger.info(f"FAILURE INTELLIGENCE ENGINE COMPLETE")
    logger.info(f"  Errors (FP+FN)      : {n_errors}")
    logger.info(f"  Failure cases total : {len(fcs)}")
    logger.info(f"  Archetypes discovered: {archetype_r.n_clusters}")
    logger.info(f"  Red-zone samples    : {zs.get('red',{}).get('n',0)}")
    logger.info(sep)

    return {
        "fcs"          : fcs,
        "severity_df"  : severity_df,
        "top_20_bs"    : top_20_bs,
        "archetype_r"  : archetype_r,
        "uncertainty_r": uncertainty_r,
        "cluster_r"    : cluster_r,
        "shap_fail_r"  : shap_fail_r,
        "lime_fail_r"  : lime_fail_r,
        "reliability_r": reliability_r,
        "summary"      : summary,
        "inputs"       : inputs,
    }
