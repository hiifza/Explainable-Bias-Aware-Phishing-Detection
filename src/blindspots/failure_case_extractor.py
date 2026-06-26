"""
src/blindspots/failure_case_extractor.py
------------------------------------------
Extracts all failure signals from the test set into a unified
FailureCaseSet for downstream analysis.

Failure signals (four tiers)
------------------------------
Tier 1 — Hard errors    : FP and FN (model is wrong)
Tier 2 — Soft errors    : Low-confidence correct predictions (uncertain)
Tier 3 — Explanation    : SHAP-LIME disagreement cases
Tier 4 — Combined       : Any sample flagged by 2+ tiers

Confidence thresholds
---------------------
  P(class) ≥ 0.95 → Green  (high confidence)
  P(class) ∈ [0.75, 0.95) → Yellow (moderate uncertainty)
  P(class) < 0.75          → Red    (high uncertainty / danger zone)

Public API
----------
    FailureCase         — dataclass for one flagged sample
    FailureCaseSet      — container with query/filter helpers
    extract_failure_cases(y_true, y_pred, y_proba, X_test,
                          agreement_df, feature_names) -> FailureCaseSet
"""

import sys
from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Optional

import numpy  as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Confidence zone thresholds
GREEN_THRESH  = 0.95
YELLOW_THRESH = 0.75

# ── FailureCase dataclass ─────────────────────────────────────────────────────

@dataclass
class FailureCase:
    """Unified failure case container for one test-set sample."""
    sample_idx       : int
    y_true           : int
    y_pred           : int
    y_proba_legit    : float    # P(legitimate)
    confidence       : float    # max(y_proba_legit, 1-y_proba_legit)
    confidence_zone  : str      # "green" / "yellow" / "red"
    is_fp            : bool     # FP: predicted legitimate, actually phishing
    is_fn            : bool     # FN: predicted phishing, actually legitimate
    is_error         : bool     # FP or FN
    tier             : int      # 1=hard error, 2=soft/uncertain, 3=explanation, 4=combined
    tiers_flagged    : list     # list of tier labels
    agreement_score  : float    # SHAP-LIME agreement (0-1, NaN if not computed)
    feature_values   : dict     = field(default_factory=dict)
    notes            : str      = ""

    @property
    def label_str(self) -> str:
        if self.is_fp: return "False Positive"
        if self.is_fn: return "False Negative"
        return "Correct-but-uncertain"

    @property
    def risk_level(self) -> str:
        if self.is_error:               return "CRITICAL"
        if self.confidence_zone == "red": return "HIGH"
        if self.confidence_zone == "yellow": return "MEDIUM"
        return "LOW"


class FailureCaseSet:
    """Container for all FailureCase objects with filter/query helpers."""

    def __init__(self, cases: list[FailureCase]) -> None:
        self.cases = cases
        self.df    = self._to_dataframe()

    def _to_dataframe(self) -> pd.DataFrame:
        rows = []
        for c in self.cases:
            rows.append({
                "sample_idx"      : c.sample_idx,
                "y_true"          : c.y_true,
                "y_pred"          : c.y_pred,
                "y_proba_legit"   : c.y_proba_legit,
                "confidence"      : c.confidence,
                "confidence_zone" : c.confidence_zone,
                "is_fp"           : c.is_fp,
                "is_fn"           : c.is_fn,
                "is_error"        : c.is_error,
                "tier"            : c.tier,
                "tiers_flagged"   : "|".join(c.tiers_flagged),
                "agreement_score" : c.agreement_score,
                "risk_level"      : c.risk_level,
                "label_str"       : c.label_str,
                **{f"feat_{k}": v for k,v in list(c.feature_values.items())[:10]},
            })
        return pd.DataFrame(rows).reset_index(drop=True)

    def filter(self, **kwargs) -> "FailureCaseSet":
        filtered = [c for c in self.cases
                    if all(getattr(c, k) == v for k, v in kwargs.items())]
        return FailureCaseSet(filtered)

    @property
    def errors(self)   -> list[FailureCase]: return [c for c in self.cases if c.is_error]
    @property
    def fp_cases(self) -> list[FailureCase]: return [c for c in self.cases if c.is_fp]
    @property
    def fn_cases(self) -> list[FailureCase]: return [c for c in self.cases if c.is_fn]
    @property
    def red_zone(self) -> list[FailureCase]: return [c for c in self.cases if c.confidence_zone=="red"]
    @property
    def yellow_zone(self)->list[FailureCase]:return [c for c in self.cases if c.confidence_zone=="yellow"]

    def __len__(self): return len(self.cases)
    def __iter__(self): return iter(self.cases)


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_failure_cases(
    y_true       : np.ndarray,
    y_pred       : np.ndarray,
    y_proba      : np.ndarray,    # P(legitimate)
    X_test       : pd.DataFrame,
    agreement_df : Optional[pd.DataFrame] = None,
    feature_names: Optional[list[str]] = None,
    max_soft_cases: int = 500,
) -> FailureCaseSet:
    """
    Extract all failure cases from the test set.

    Parameters
    ----------
    y_true, y_pred, y_proba : test-set arrays
    X_test           : feature DataFrame (aligned with arrays)
    agreement_df     : SHAP-LIME agreement DataFrame from M8 (optional)
    feature_names    : list of column names
    max_soft_cases   : max soft (uncertain) cases to include (for speed)

    Returns
    -------
    FailureCaseSet
    """
    y_true  = np.asarray(y_true)
    y_pred  = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)  # P(legitimate)
    n       = len(y_true)

    # Confidence = max(P(legit), P(phishing))
    confidence = np.maximum(y_proba, 1 - y_proba)

    def _zone(c: float) -> str:
        if c >= GREEN_THRESH:  return "green"
        if c >= YELLOW_THRESH: return "yellow"
        return "red"

    # Agreement lookup by sample_idx
    agree_map: dict[int, float] = {}
    if agreement_df is not None and "sample_id" in agreement_df.columns:
        agree_map = dict(zip(
            agreement_df["sample_id"].astype(int),
            agreement_df["agreement_score"].astype(float),
        ))

    # Track which indices to include
    flagged_set: dict[int, set] = {}   # idx -> set of tier labels

    # Tier 1: hard errors
    error_mask = y_pred != y_true
    for idx in np.where(error_mask)[0]:
        flagged_set.setdefault(int(idx), set()).add("tier1_error")

    # Tier 2: soft errors (correct but low confidence)
    soft_mask = (~error_mask) & (confidence < GREEN_THRESH)
    soft_indices = np.where(soft_mask)[0]
    rng = np.random.default_rng(42)
    if len(soft_indices) > max_soft_cases:
        soft_indices = rng.choice(soft_indices, size=max_soft_cases, replace=False)
    for idx in soft_indices:
        flagged_set.setdefault(int(idx), set()).add("tier2_uncertain")

    # Tier 3: SHAP-LIME disagreements (agreement < 0.4)
    for sid, ascore in agree_map.items():
        if ascore < 0.4 and sid < n:
            flagged_set.setdefault(int(sid), set()).add("tier3_explanation")

    # Build FailureCase objects
    cases: list[FailureCase] = []
    for idx, tier_tags in sorted(flagged_set.items()):
        if idx >= n:
            continue
        is_fp = bool((y_pred[idx] == 0) and (y_true[idx] == 1))
        is_fn = bool((y_pred[idx] == 1) and (y_true[idx] == 0))
        conf  = float(confidence[idx])
        zone  = _zone(conf)
        ascore= agree_map.get(idx, np.nan)

        tiers = sorted(tier_tags)
        if len(tier_tags) >= 2:
            tiers.append("tier4_combined")
        tier_num = 4 if len(tier_tags) >= 2 else int(min(tier_tags)[4])

        fvals: dict = {}
        if feature_names and idx < len(X_test):
            row = X_test.iloc[idx]
            fvals = {fn: float(row[fn]) for fn in (feature_names[:10]
                                                     if feature_names else [])}

        cases.append(FailureCase(
            sample_idx      = idx,
            y_true          = int(y_true[idx]),
            y_pred          = int(y_pred[idx]),
            y_proba_legit   = round(float(y_proba[idx]), 6),
            confidence      = round(conf, 6),
            confidence_zone = zone,
            is_fp           = is_fp,
            is_fn           = is_fn,
            is_error        = bool(y_pred[idx] != y_true[idx]),
            tier            = tier_num,
            tiers_flagged   = tiers,
            agreement_score = float(ascore) if not np.isnan(ascore) else np.nan,
            feature_values  = fvals,
        ))

    fcs = FailureCaseSet(cases)
    logger.info(
        f"Failure cases extracted: {len(fcs):,} total  "
        f"errors={len(fcs.errors)} (FP={len(fcs.fp_cases)} FN={len(fcs.fn_cases)})  "
        f"red={len(fcs.red_zone)}  yellow={len(fcs.yellow_zone)}"
    )
    return fcs
