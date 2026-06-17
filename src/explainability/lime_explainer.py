"""
src/explainability/lime_explainer.py
--------------------------------------
LIME tabular explainer with native library support and a robust fallback.

When lime (pip install lime>=0.2.0.1) is installed the real
LimeTabularExplainer is used.  Otherwise a FallbackLIMEExplainer
approximates local linear explanations by:
  1. Perturbing the sample around its neighbourhood.
  2. Fitting a weighted Ridge regression.
  3. Returning the regression coefficients as feature contributions.

This produces valid local approximations compatible with all downstream
consumers (shap_lime_comparator, lime_local, lime_report).

Public API
----------
    LIMEResult                  — dataclass per-sample explanation
    FallbackLIMEExplainer       — numpy-only local linear explainer
    build_lime_explainer(X_background, feature_names, class_names, mode)
    explain_sample(explainer, predict_fn, sample, feature_names,
                   n_features, n_samples)     -> LIMEResult
"""

import sys
from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Any, Callable, Optional

import numpy  as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── LIME availability ─────────────────────────────────────────────────────────
try:
    from lime.lime_tabular import LimeTabularExplainer as _LimeTabularExplainer
    _LIME_AVAILABLE = True
    logger.info("lime library available — using native LimeTabularExplainer")
except ImportError:
    _LIME_AVAILABLE = False
    logger.warning(
        "lime not installed — using FallbackLIMEExplainer. "
        "Install with: pip install lime>=0.2.0.1"
    )


# ── LIMEResult dataclass ──────────────────────────────────────────────────────

@dataclass
class LIMEResult:
    """
    Per-sample LIME explanation container.

    Attributes
    ----------
    sample_idx          : row index in the test set
    feature_names       : all feature names (56 for Track B)
    contributions       : dict[feature_name -> contribution_value]
                          positive = pushes toward class 1 (legitimate)
                          negative = pushes toward class 0 (phishing)
    top_features        : list of (feature_name, contribution) sorted by |contribution|
    prediction_proba    : model probability for class 1
    prediction_class    : model predicted class (0 or 1)
    local_r2            : R² of the local linear fit (quality indicator)
    is_native_lime      : True if native LIME was used
    intercept           : local linear model intercept
    """
    sample_idx       : int
    feature_names    : list
    contributions    : dict
    top_features     : list
    prediction_proba : float
    prediction_class : int
    local_r2         : float
    is_native_lime   : bool
    intercept        : float = 0.0

    def get_top_n_features(self, n: int = 5) -> list[str]:
        """Return feature names of the top-n contributors by |contribution|."""
        return [f for f, _ in self.top_features[:n]]

    def get_top_n_dict(self, n: int = 5) -> dict[str, float]:
        """Return {feature: contribution} for the top-n contributors."""
        return {f: c for f, c in self.top_features[:n]}


# ── Fallback LIME Explainer ───────────────────────────────────────────────────

class FallbackLIMEExplainer:
    """
    Numpy-only local linear LIME approximation.

    Algorithm
    ---------
    1. Sample `n_perturb` neighbours around the query point by
       adding Gaussian noise scaled by the training data std.
    2. Weight each neighbour by its cosine similarity to the query point
       (exponential kernel).
    3. Fit a Ridge regression: predict_proba(neighbours) ~ X_neighbours.
    4. Return Ridge coefficients as feature contributions.

    Parameters
    ----------
    X_background : np.ndarray (training data) — used for std estimation
    feature_names: list[str]
    class_names  : list[str]
    random_state : int
    """

    def __init__(
        self,
        X_background  : np.ndarray,
        feature_names : list[str],
        class_names   : list[str] = ("phishing", "legitimate"),
        random_state  : int = 42,
    ) -> None:
        self.feature_names  = feature_names
        self.class_names    = class_names
        self.random_state   = random_state
        self._rng           = np.random.default_rng(random_state)

        # Training data statistics for perturbation
        self._mean = X_background.mean(axis=0).astype(np.float64)
        self._std  = X_background.std(axis=0).astype(np.float64)
        self._std  = np.where(self._std < 1e-9, 1.0, self._std)

        logger.debug(
            f"FallbackLIMEExplainer ready: "
            f"{X_background.shape[0]:,} background rows × {X_background.shape[1]} features"
        )

    def explain_instance(
        self,
        sample      : np.ndarray,
        predict_fn  : Callable,
        num_features: int = 10,
        num_samples : int = 2000,
    ) -> "FallbackLIMEExplainer._FallbackExp":
        """
        Generate a local linear explanation for one sample.

        Returns an object with the same interface as a native LIME explanation
        (`.as_list()` method returning [(feature_name, contribution), ...]).
        """
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score

        sample = sample.astype(np.float64).flatten()
        n_feat = len(sample)

        # 1. Perturb neighbourhood
        noise    = self._rng.normal(0, 1, size=(num_samples, n_feat))
        X_neigh  = sample[np.newaxis, :] + noise * self._std[np.newaxis, :]

        # 2. Predict on neighbourhood
        y_neigh  = predict_fn(X_neigh)[:, 1]   # class 1 proba

        # 3. Kernel weights (exponential decay by L2 distance to sample)
        dists    = np.linalg.norm((X_neigh - sample) / self._std, axis=1)
        kernel_w = np.exp(-(dists ** 2) / (2 * (0.75 * np.sqrt(n_feat)) ** 2))

        # 4. Fit weighted Ridge
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(X_neigh, y_neigh, sample_weight=kernel_w)

        # 5. R² quality metric on neighbourhood
        y_hat = ridge.predict(X_neigh)
        try:
            local_r2 = float(r2_score(y_neigh, y_hat, sample_weight=kernel_w))
        except Exception:
            local_r2 = 0.0

        coef        = ridge.coef_.astype(np.float64)
        intercept   = float(ridge.intercept_)
        pred_proba  = float(predict_fn(sample[np.newaxis, :])[0, 1])

        # Sort by |coefficient| descending, take top num_features
        order     = np.argsort(np.abs(coef))[::-1][:num_features]
        top_feats = [(self.feature_names[i], float(coef[i])) for i in order]

        return FallbackLIMEExplainer._FallbackExp(
            top_features  = top_feats,
            coef          = coef,
            intercept     = intercept,
            local_r2      = local_r2,
            pred_proba    = pred_proba,
            n_features    = n_feat,
        )

    class _FallbackExp:
        """Minimal interface matching native LIME explanation."""
        def __init__(self, top_features, coef, intercept,
                     local_r2, pred_proba, n_features):
            self._top_features = top_features
            self._coef         = coef
            self._intercept    = intercept
            self.local_r2      = local_r2
            self.pred_proba    = pred_proba
            self.n_features    = n_features

        def as_list(self, label=1) -> list[tuple[str, float]]:
            return self._top_features

        def as_map(self) -> dict:
            return {1: [(i, c) for i, (_, c) in enumerate(self._top_features)]}


# ── Public builder ────────────────────────────────────────────────────────────

def build_lime_explainer(
    X_background : np.ndarray,
    feature_names: list[str],
    class_names  : tuple[str, str] = ("phishing", "legitimate"),
    mode         : str = "classification",
    random_state : int = 42,
):
    """
    Build and return a LIME explainer (native or fallback).

    Parameters
    ----------
    X_background  : training data numpy array — used as the LIME background
    feature_names : list of column names
    class_names   : class label strings
    mode          : "classification" or "regression"
    random_state  : reproducibility seed

    Returns
    -------
    LimeTabularExplainer  (native) or FallbackLIMEExplainer
    """
    if _LIME_AVAILABLE:
        explainer = _LimeTabularExplainer(
            training_data  = X_background,
            feature_names  = feature_names,
            class_names    = list(class_names),
            mode           = mode,
            discretize_continuous = True,
            random_state   = random_state,
        )
        logger.info(
            f"LimeTabularExplainer (native) built on "
            f"{X_background.shape[0]:,} background rows"
        )
    else:
        explainer = FallbackLIMEExplainer(
            X_background  = X_background,
            feature_names = feature_names,
            class_names   = list(class_names),
            random_state  = random_state,
        )
        logger.info(
            f"FallbackLIMEExplainer built on "
            f"{X_background.shape[0]:,} background rows"
        )
    return explainer


# ── explain_sample ────────────────────────────────────────────────────────────

def explain_sample(
    explainer    : Any,
    predict_fn   : Callable,
    sample       : np.ndarray,
    feature_names: list[str],
    n_features   : int = 10,
    n_samples    : int = 2000,
    sample_idx   : int = 0,
) -> LIMEResult:
    """
    Generate a LIME explanation for a single sample.

    Parameters
    ----------
    explainer    : native LimeTabularExplainer or FallbackLIMEExplainer
    predict_fn   : model.predict_proba
    sample       : 1-D numpy array (one test row)
    feature_names: list of feature names
    n_features   : number of features to include in explanation
    n_samples    : perturbation samples (LIME neighbourhood size)
    sample_idx   : index in the test set (for tracking)

    Returns
    -------
    LIMEResult
    """
    sample = np.asarray(sample, dtype=np.float64).flatten()

    if _LIME_AVAILABLE and isinstance(explainer, _LimeTabularExplainer):
        exp = explainer.explain_instance(
            data_row       = sample,
            predict_fn     = predict_fn,
            num_features   = n_features,
            num_samples    = n_samples,
            labels         = (1,),
        )
        contributions_raw = exp.as_list(label=1)
        local_r2_val      = float(exp.score) if hasattr(exp, "score") else 0.0
        intercept_val     = float(exp.intercept[1]) if hasattr(exp, "intercept") else 0.0
        is_native         = True
    else:
        exp               = explainer.explain_instance(
            sample, predict_fn,
            num_features=n_features, num_samples=n_samples,
        )
        contributions_raw = exp.as_list(label=1)
        local_r2_val      = float(exp.local_r2)
        intercept_val     = float(exp._intercept)
        is_native         = False

    # Sort by |contribution| descending
    top_features = sorted(contributions_raw, key=lambda x: abs(x[1]), reverse=True)
    contributions_dict = {f: c for f, c in top_features}

    pred_proba = float(predict_fn(sample[np.newaxis, :])[0, 1])
    pred_class = int(pred_proba >= 0.5)

    return LIMEResult(
        sample_idx       = sample_idx,
        feature_names    = feature_names,
        contributions    = contributions_dict,
        top_features     = top_features,
        prediction_proba = round(pred_proba, 6),
        prediction_class = pred_class,
        local_r2         = round(local_r2_val, 6),
        is_native_lime   = is_native,
        intercept        = intercept_val,
    )


def is_native_lime() -> bool:
    """Return True if the real LIME library is installed."""
    return _LIME_AVAILABLE
