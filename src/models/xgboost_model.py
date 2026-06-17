"""
src/models/xgboost_model.py
-----------------------------
XGBoost model configuration for the phishing detection project.

If xgboost is installed (pip install xgboost), the real XGBClassifier is
used.  Otherwise a sklearn HistGradientBoostingClassifier is used as a
drop-in fallback with equivalent hyperparameter spirit.

Install xgboost for production:
    pip install xgboost>=2.0.0
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "XGBoost"
MODEL_ID   = "xgboost"

# ── Try real XGBoost first ────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
    logger.debug("xgboost library found — using XGBClassifier")
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    _XGBOOST_AVAILABLE = False
    logger.warning(
        "xgboost not installed — falling back to "
        "sklearn.HistGradientBoostingClassifier. "
        "Install with: pip install xgboost>=2.0.0"
    )

# ── Hyperparameters ───────────────────────────────────────────────────────────
HYPERPARAMS_XGBOOST: dict = {
    "n_estimators"    : 300,
    "learning_rate"   : 0.05,
    "max_depth"       : 6,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "tree_method"     : "hist",    # fast histogram-based method
    "eval_metric"     : "auc",
    "use_label_encoder": False,
    "random_state"    : 42,
    "n_jobs"          : -1,
    "verbosity"       : 0,
}

HYPERPARAMS_FALLBACK: dict = {
    "max_iter"          : 300,
    "learning_rate"     : 0.05,
    "max_depth"         : 6,
    "min_samples_leaf"  : 10,
    "l2_regularization" : 1.0,
    "random_state"      : 42,
}


def get_model():
    """
    Return a configured, unfitted XGBoost (or fallback) estimator.

    Returns
    -------
    XGBClassifier  if xgboost is installed
    HistGradientBoostingClassifier  otherwise
    """
    if _XGBOOST_AVAILABLE:
        model = XGBClassifier(**HYPERPARAMS_XGBOOST)
        logger.debug(
            f"Created {MODEL_NAME} (XGBClassifier): "
            f"n_estimators={HYPERPARAMS_XGBOOST['n_estimators']}, "
            f"lr={HYPERPARAMS_XGBOOST['learning_rate']}"
        )
    else:
        model = HistGradientBoostingClassifier(**HYPERPARAMS_FALLBACK)
        logger.debug(
            f"Created {MODEL_NAME} (HistGB fallback): "
            f"max_iter={HYPERPARAMS_FALLBACK['max_iter']}, "
            f"lr={HYPERPARAMS_FALLBACK['learning_rate']}"
        )
    return model


def is_native() -> bool:
    """Return True if the real XGBoost library is available."""
    return _XGBOOST_AVAILABLE
