"""
src/blindspots/severity_ranker.py
-----------------------------------
Computes a composite Blind Spot Severity Score for every failure case
and ranks the top-20 most dangerous blind spots.

Severity Score formula
----------------------
severity = w1 * error_weight
         + w2 * (1 - confidence)
         + w3 * (1 - agreement_score)   [0 if NaN]
         + w4 * shap_disagreement       [optional]

Weights (default):
  w1 = 0.40  (hard error is most important)
  w2 = 0.30  (low confidence → high uncertainty)
  w3 = 0.20  (SHAP-LIME disagreement → unreliable explanation)
  w4 = 0.10  (SHAP-feature dominance instability)

Score is normalised to [0, 1].

Public API
----------
    compute_severity_scores(fcs, shap_values, feature_names,
                            weights) -> pd.DataFrame
    rank_blind_spots(severity_df, top_n)  -> pd.DataFrame
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
from src.blindspots.failure_case_extractor     import FailureCaseSet

logger = get_logger(__name__)

DEFAULT_WEIGHTS = {
    "error"         : 0.40,
    "uncertainty"   : 0.30,
    "disagreement"  : 0.20,
    "shap_instab"   : 0.10,
}


def _shap_instability(
    sample_idx   : int,
    shap_values  : Optional[np.ndarray],
    n_total      : int,
) -> float:
    """
    Proxy for SHAP instability: how much a single sample's top feature
    deviates from the global top feature.

    Returns a float in [0, 1].
    """
    if shap_values is None or sample_idx >= len(shap_values):
        return 0.0
    sv       = np.abs(shap_values[sample_idx])
    sv_norm  = sv / max(sv.sum(), 1e-12)
    # Gini-like concentration: high concentration = stable (low instability)
    gini_coef = 1.0 - (sv_norm ** 2).sum()   # ranges 0 (one dominant) to ~1 (uniform)
    return float(np.clip(gini_coef, 0, 1))


def compute_severity_scores(
    fcs          : FailureCaseSet,
    shap_values  : Optional[np.ndarray] = None,
    feature_names: Optional[list[str]]  = None,
    weights      : Optional[dict]       = None,
) -> pd.DataFrame:
    """
    Compute the Blind Spot Severity Score for every failure case.

    Parameters
    ----------
    fcs           : FailureCaseSet
    shap_values   : optional SHAP array (n_shap × n_features)
    feature_names : feature names (for SHAP labelling)
    weights       : override DEFAULT_WEIGHTS

    Returns
    -------
    pd.DataFrame  columns: sample_idx, y_true, y_pred, confidence,
                           confidence_zone, is_error, agreement_score,
                           error_component, uncertainty_component,
                           disagreement_component, shap_instab_component,
                           severity_score, severity_rank, risk_level, label_str
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    n_shap = len(shap_values) if shap_values is not None else 0

    rows = []
    for c in fcs:
        # Component 1: error (binary → 1.0 if wrong, 0 otherwise)
        err_comp  = 1.0 if c.is_error else 0.0

        # Component 2: uncertainty (1 − confidence, higher = worse)
        unc_comp  = float(1.0 - c.confidence)

        # Component 3: SHAP-LIME disagreement (1 − agreement)
        ascore    = c.agreement_score
        if np.isnan(ascore):
            dis_comp = 0.0   # no agreement data → neutral
        else:
            dis_comp = float(1.0 - ascore)

        # Component 4: SHAP instability
        instab    = _shap_instability(c.sample_idx, shap_values, n_shap)

        score = (w["error"]       * err_comp
               + w["uncertainty"] * unc_comp
               + w["disagreement"]* dis_comp
               + w["shap_instab"] * instab)

        rows.append({
            "sample_idx"            : c.sample_idx,
            "y_true"                : c.y_true,
            "y_pred"                : c.y_pred,
            "confidence"            : round(c.confidence, 6),
            "confidence_zone"       : c.confidence_zone,
            "is_error"              : c.is_error,
            "is_fp"                 : c.is_fp,
            "is_fn"                 : c.is_fn,
            "agreement_score"       : round(float(ascore) if not np.isnan(ascore) else -1, 4),
            "tiers_flagged"         : "|".join(c.tiers_flagged),
            "error_component"       : round(err_comp,  4),
            "uncertainty_component" : round(unc_comp,  4),
            "disagreement_component": round(dis_comp,  4),
            "shap_instab_component" : round(instab,    4),
            "severity_score"        : round(float(score), 6),
            "risk_level"            : c.risk_level,
            "label_str"             : c.label_str,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalise severity_score to [0, 1]
    s_min, s_max = df["severity_score"].min(), df["severity_score"].max()
    df["severity_score_norm"] = (
        (df["severity_score"] - s_min) / max(s_max - s_min, 1e-9)
    ).round(6)

    # Rank: 1 = most severe
    df["severity_rank"] = df["severity_score_norm"].rank(
        ascending=False, method="min"
    ).astype(int)

    df = df.sort_values("severity_rank").reset_index(drop=True)

    logger.info(
        f"Severity scores computed: {len(df)} cases  "
        f"max_score={df['severity_score_norm'].max():.4f}  "
        f"mean_score={df['severity_score_norm'].mean():.4f}"
    )
    return df


def rank_blind_spots(
    severity_df: pd.DataFrame,
    top_n      : int = 20,
) -> pd.DataFrame:
    """
    Return the top-N most dangerous blind spots with full context.

    Parameters
    ----------
    severity_df : output of compute_severity_scores()
    top_n       : number of top cases to return

    Returns
    -------
    pd.DataFrame  — top_n rows sorted by severity_rank
    """
    if severity_df.empty:
        return severity_df

    top = severity_df.nsmallest(top_n, "severity_rank").copy()
    logger.info(
        f"Top-{top_n} blind spots:  "
        f"errors={top['is_error'].sum()}  "
        f"red_zone={int((top['confidence_zone']=='red').sum())}  "
        f"max_severity={top['severity_score_norm'].max():.4f}"
    )
    return top.reset_index(drop=True)
