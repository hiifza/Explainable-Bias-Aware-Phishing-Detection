"""
src/explainability/shap_explainer.py
--------------------------------------
Core SHAP value computation dispatcher for Module M7.1.

Provides a unified SHAPResult container and dispatcher that selects
the appropriate explainer backend:

  1. Native SHAP (requires: pip install shap>=0.44.0)
     - TreeExplainer  for RF, XGBoost, LightGBM, HistGBM
     - LinearExplainer for Logistic Regression
  2. FallbackExplainer (no external dependency)
     - Uses model.feature_importances_ / coef_ weighted by per-sample
       feature deviation from the training background mean.
     - Satisfies the SHAP additive property exactly:
         base_value + sum(shap_values[i]) == y_proba[i]  for all i

All downstream modules (shap_global, shap_local, shap_interactions)
operate on the unified SHAPResult and never import shap directly.

Public API
----------
    SHAPResult        — dataclass holding all SHAP outputs
    build_explainer(model, X_background)              -> SHAPResult-builder
    compute_shap_values(model, X_background, X_explain,
                        feature_names, sample_n)      -> SHAPResult
"""

import sys
from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Any, Optional

import numpy  as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── SHAP availability ─────────────────────────────────────────────────────────
try:
    import shap as _shap
    _SHAP_AVAILABLE = True
    logger.info(f"shap {_shap.__version__} available — using native SHAP")
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning(
        "shap not installed — using FallbackExplainer. "
        "Install with: pip install shap>=0.44.0"
    )

# ── Model type detection ──────────────────────────────────────────────────────
_TREE_TYPES = {
    "RandomForestClassifier",
    "HistGradientBoostingClassifier",
    "GradientBoostingClassifier",
    "XGBClassifier",
    "LGBMClassifier",
    "DecisionTreeClassifier",
    "ExtraTreesClassifier",
}
_LINEAR_TYPES = {
    "LogisticRegression",
    "LinearSVC",
    "RidgeClassifier",
    "SGDClassifier",
}


def _model_class(model: Any) -> str:
    return type(model).__name__


def _detect_model_type(model: Any) -> str:
    cls = _model_class(model)
    if cls in _TREE_TYPES:
        return "tree"
    if cls in _LINEAR_TYPES:
        return "linear"
    return "other"


# ── SHAPResult dataclass ──────────────────────────────────────────────────────

@dataclass
class SHAPResult:
    """
    Unified container for SHAP analysis outputs.

    Attributes
    ----------
    shap_values       : np.ndarray (n_samples × n_features)
                        SHAP values for class 1 (legitimate)
    expected_value    : float  — model output at background mean
    feature_names     : list[str]
    X_explained       : pd.DataFrame — rows used for SHAP computation
    y_pred            : np.ndarray   — binary predictions on X_explained
    y_proba           : np.ndarray   — probability scores on X_explained
    base_values       : np.ndarray   — expected_value broadcast to (n_samples,)
    is_native_shap    : bool
    model_type        : str  ("tree" | "linear" | "other" | "fallback")
    model_class       : str  (class name of the model)
    n_samples         : int
    n_features        : int
    """
    shap_values    : np.ndarray
    expected_value : float
    feature_names  : list
    X_explained    : pd.DataFrame
    y_pred         : np.ndarray
    y_proba        : np.ndarray
    base_values    : np.ndarray
    is_native_shap : bool
    model_type     : str
    model_class    : str
    n_samples      : int      = field(init=False)
    n_features     : int      = field(init=False)

    def __post_init__(self):
        self.n_samples  = self.shap_values.shape[0]
        self.n_features = self.shap_values.shape[1]

    def get_mean_abs_shap(self) -> pd.Series:
        """Return mean |SHAP| per feature, sorted descending."""
        return pd.Series(
            np.abs(self.shap_values).mean(axis=0),
            index=self.feature_names,
        ).sort_values(ascending=False)

    def get_feature_ranking(self) -> pd.DataFrame:
        """Return ranked DataFrame: feature, mean_shap, rank, relative_importance."""
        mean_abs = self.get_mean_abs_shap()
        total    = mean_abs.sum()
        df = pd.DataFrame({
            "feature"            : mean_abs.index,
            "mean_abs_shap"      : mean_abs.values.round(8),
            "rank"               : range(1, len(mean_abs) + 1),
            "relative_importance": (mean_abs.values / max(total, 1e-12)).round(6),
        })
        return df.reset_index(drop=True)


# ── Native SHAP ───────────────────────────────────────────────────────────────

def _native_shap(
    model       : Any,
    X_background: pd.DataFrame,
    X_explain   : pd.DataFrame,
    model_type  : str,
) -> tuple[np.ndarray, float]:
    """
    Compute SHAP values using the native shap library.

    Returns
    -------
    (shap_values, expected_value)  where shap_values has shape
    (n_samples, n_features) for class=1 (legitimate)
    """
    logger.info(
        f"Native SHAP: {model_type} explainer for "
        f"{_model_class(model)} on {len(X_explain):,} samples …"
    )

    if model_type == "tree":
        explainer = _shap.TreeExplainer(
            model,
            data=X_background.values,
            feature_perturbation="interventional",
        )
        sv = explainer.shap_values(X_explain.values, check_additivity=False)
        # sv is list of 2 arrays [class0, class1] or single array
        if isinstance(sv, list) and len(sv) == 2:
            shap_vals   = sv[1].astype(np.float64)
            exp_val     = float(explainer.expected_value[1])
        else:
            shap_vals   = np.asarray(sv).astype(np.float64)
            ev          = explainer.expected_value
            exp_val     = float(ev[1] if hasattr(ev, '__len__') else ev)

    elif model_type == "linear":
        explainer = _shap.LinearExplainer(model, X_background.values)
        sv        = explainer.shap_values(X_explain.values)
        if isinstance(sv, list):
            shap_vals = sv[1].astype(np.float64)
            exp_val   = float(explainer.expected_value[1])
        else:
            shap_vals = np.asarray(sv).astype(np.float64)
            exp_val   = float(explainer.expected_value)

    else:
        # KernelExplainer — slow; use small background
        bg_small  = _shap.sample(X_background.values, 200)
        explainer = _shap.KernelExplainer(
            model.predict_proba, bg_small
        )
        sv        = explainer.shap_values(
            X_explain.values, nsamples=200, silent=True
        )
        shap_vals = (sv[1] if isinstance(sv, list) else sv).astype(np.float64)
        ev        = explainer.expected_value
        exp_val   = float(ev[1] if hasattr(ev, '__len__') else ev)

    logger.info(
        f"Native SHAP complete: shap_values shape={shap_vals.shape}  "
        f"expected_value={exp_val:.6f}"
    )
    return shap_vals, exp_val


# ── Fallback explainer ────────────────────────────────────────────────────────

def _fallback_shap(
    model       : Any,
    X_background: pd.DataFrame,
    X_explain   : pd.DataFrame,
    model_type  : str,
) -> tuple[np.ndarray, float]:
    """
    Compute pseudo-SHAP values without the shap library.

    Algorithm
    ---------
    1. Obtain base importances from model internals.
    2. Compute per-feature deviation: X_explain - background_mean.
    3. Distribute each sample's total prediction contribution
       proportionally across features, weighted by |importance × deviation|.
    4. Satisfies additive property exactly:
         base_value + sum(shap_values[i]) == y_proba[i]
    """
    logger.info(
        f"FallbackExplainer for {_model_class(model)} "
        f"on {len(X_explain):,} samples …"
    )

    n_feats     = X_explain.shape[1]
    X_bg_arr    = X_background.values.astype(np.float64)
    X_exp_arr   = X_explain.values.astype(np.float64)
    bg_mean     = X_bg_arr.mean(axis=0)

    # ── base importances ─────────────────────────────────────────────────────
    if hasattr(model, "feature_importances_"):
        raw_imp = np.asarray(model.feature_importances_, dtype=np.float64)
    elif hasattr(model, "coef_"):
        coef    = np.asarray(model.coef_, dtype=np.float64).ravel()
        raw_imp = np.abs(coef)
    else:
        raw_imp = np.ones(n_feats, dtype=np.float64)

    # Normalise to sum = 1
    imp_sum = raw_imp.sum()
    norm_imp = raw_imp / max(imp_sum, 1e-12)

    # ── predictions ──────────────────────────────────────────────────────────
    y_proba     = model.predict_proba(X_explain)[:, 1]
    base_value  = float(model.predict_proba(X_background)[:, 1].mean())
    deltas      = y_proba - base_value          # (n_samples,)

    # ── per-sample feature weights ────────────────────────────────────────────
    deviations = X_exp_arr - bg_mean[np.newaxis, :]   # (n_samples, n_feats)
    # weight = |deviation| × importance
    weights    = np.abs(deviations) * norm_imp[np.newaxis, :]

    # normalise weights per sample so they sum to 1
    row_sums   = weights.sum(axis=1, keepdims=True)
    row_sums   = np.where(row_sums < 1e-12, 1.0, row_sums)
    w_norm     = weights / row_sums               # (n_samples, n_feats)

    # distribute total delta proportionally, preserve sign of deviation
    signs      = np.sign(deviations)
    signs[signs == 0] = 1.0
    shap_vals  = deltas[:, np.newaxis] * w_norm * signs

    # ── additive correction ───────────────────────────────────────────────────
    # Ensure sum(shap_vals[i]) == deltas[i] exactly
    row_totals = shap_vals.sum(axis=1, keepdims=True)
    row_totals = np.where(np.abs(row_totals) < 1e-12, 1.0, row_totals)
    shap_vals  = shap_vals * (deltas[:, np.newaxis] / row_totals)

    logger.info(
        f"FallbackExplainer complete: shape={shap_vals.shape}  "
        f"base_value={base_value:.6f}  "
        f"max_abs_shap={np.abs(shap_vals).max():.6f}"
    )
    return shap_vals.astype(np.float64), float(base_value)


# ── Public entry point ────────────────────────────────────────────────────────

def compute_shap_values(
    model       : Any,
    X_background: pd.DataFrame,
    X_explain   : pd.DataFrame,
    feature_names: list[str],
    sample_n    : int = 3000,
    random_state: int = 42,
) -> SHAPResult:
    """
    Compute SHAP values using native SHAP (if available) or fallback.

    Parameters
    ----------
    model        : fitted sklearn-compatible estimator
    X_background : training DataFrame for explainer background
    X_explain    : test DataFrame to explain (subsample if sample_n < len)
    feature_names: list of column names
    sample_n     : max rows to explain (for performance)
    random_state : reproducibility seed

    Returns
    -------
    SHAPResult
    """
    rng  = np.random.default_rng(random_state)
    n    = len(X_explain)

    # Subsample X_explain for SHAP computation
    if n > sample_n:
        idx       = rng.choice(n, size=sample_n, replace=False)
        X_sub     = X_explain.iloc[idx].reset_index(drop=True)
        logger.info(f"SHAP subsampled: {n:,} → {sample_n:,} test rows")
    else:
        X_sub = X_explain.copy()

    # Subsample background for speed
    bg_n = min(len(X_background), 2000)
    if len(X_background) > bg_n:
        bg_idx = rng.choice(len(X_background), size=bg_n, replace=False)
        X_bg   = X_background.iloc[bg_idx].reset_index(drop=True)
    else:
        X_bg = X_background.copy()

    mtype = _detect_model_type(model)

    if _SHAP_AVAILABLE:
        shap_vals, exp_val = _native_shap(model, X_bg, X_sub, mtype)
        is_native = True
    else:
        shap_vals, exp_val = _fallback_shap(model, X_bg, X_sub, mtype)
        is_native = False

    y_pred  = model.predict(X_sub)
    y_proba = model.predict_proba(X_sub)[:, 1]

    return SHAPResult(
        shap_values    = shap_vals,
        expected_value = exp_val,
        feature_names  = list(feature_names),
        X_explained    = X_sub,
        y_pred         = y_pred,
        y_proba        = y_proba,
        base_values    = np.full(len(X_sub), exp_val),
        is_native_shap = is_native,
        model_type     = mtype if is_native else "fallback",
        model_class    = _model_class(model),
    )
