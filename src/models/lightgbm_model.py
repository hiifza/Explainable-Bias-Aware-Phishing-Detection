"""
src/models/lightgbm_model.py
------------------------------
LightGBM model configuration for the phishing detection project.

If lightgbm is installed (pip install lightgbm), the real LGBMClassifier is
used.  Otherwise a sklearn HistGradientBoostingClassifier is used as a
drop-in fallback.

Install lightgbm for production:
    pip install lightgbm>=4.0.0
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "LightGBM"
MODEL_ID   = "lightgbm"

# ── Try real LightGBM first ───────────────────────────────────────────────────
try:
    from lightgbm import LGBMClassifier
    _LGBM_AVAILABLE = True
    logger.debug("lightgbm library found — using LGBMClassifier")
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    _LGBM_AVAILABLE = False
    logger.warning(
        "lightgbm not installed — falling back to "
        "sklearn.HistGradientBoostingClassifier. "
        "Install with: pip install lightgbm>=4.0.0"
    )

# ── Hyperparameters ───────────────────────────────────────────────────────────
HYPERPARAMS_LGBM: dict = {
    "n_estimators"  : 300,
    "learning_rate" : 0.05,
    "num_leaves"    : 63,       # 2^(max_depth) - 1 for depth-6 equivalent
    "max_depth"     : -1,       # LightGBM uses num_leaves for depth control
    "min_child_samples": 20,
    "reg_alpha"     : 0.1,      # L1 regularisation
    "reg_lambda"    : 1.0,      # L2 regularisation
    "class_weight"  : "balanced",
    "n_jobs"        : -1,
    "random_state"  : 42,
    "verbose"       : -1,       # suppress training output
}

HYPERPARAMS_FALLBACK: dict = {
    "max_iter"          : 300,
    "learning_rate"     : 0.05,
    "max_depth"         : 6,
    "num_leaf_nodes"    : 63,
    "min_samples_leaf"  : 20,
    "l2_regularization" : 1.0,
    "random_state"      : 42,
}


def get_model():
    """
    Return a configured, unfitted LightGBM (or fallback) estimator.

    Returns
    -------
    LGBMClassifier  if lightgbm is installed
    HistGradientBoostingClassifier  otherwise
    """
    if _LGBM_AVAILABLE:
        model = LGBMClassifier(**HYPERPARAMS_LGBM)
        logger.debug(
            f"Created {MODEL_NAME} (LGBMClassifier): "
            f"n_estimators={HYPERPARAMS_LGBM['n_estimators']}, "
            f"num_leaves={HYPERPARAMS_LGBM['num_leaves']}"
        )
    else:
        # Use slightly different hyperparams from XGBoost fallback
        # by using more leaf nodes (num_leaf_nodes)
        params = {k: v for k, v in HYPERPARAMS_FALLBACK.items()
                  if k != "num_leaf_nodes"}
        model = HistGradientBoostingClassifier(
            **params,
            max_leaf_nodes=HYPERPARAMS_FALLBACK["num_leaf_nodes"],
        )
        logger.debug(
            f"Created {MODEL_NAME} (HistGB fallback): "
            f"max_iter={HYPERPARAMS_FALLBACK['max_iter']}, "
            f"max_leaf_nodes={HYPERPARAMS_FALLBACK['num_leaf_nodes']}"
        )
    return model


def is_native() -> bool:
    """Return True if the real LightGBM library is available."""
    return _LGBM_AVAILABLE
