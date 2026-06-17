"""
src/training/model_registry.py
--------------------------------
Central registry that maps model identifiers to their factory functions
and configuration metadata.

All training, evaluation, and SHAP/LIME modules import from here to
ensure every part of the project uses the same model objects.

Public API
----------
    get_all_models(track)      -> dict[str, estimator]
    get_model_by_id(model_id)  -> estimator
    MODEL_IDS                  -> list[str]
    MODEL_DISPLAY_NAMES        -> dict[str, str]
"""

import sys
from pathlib import Path
from typing  import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
import src.models.logistic_regression as _lr
import src.models.random_forest       as _rf
import src.models.xgboost_model       as _xgb
import src.models.lightgbm_model      as _lgbm

logger = get_logger(__name__)

# Ordered list of model identifiers (used for benchmark table row order)
MODEL_IDS: list[str] = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
]

# Human-readable display names (used in plots and reports)
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "random_forest"      : "Random Forest",
    "xgboost"            : "XGBoost",
    "lightgbm"           : "LightGBM",
}

# Module map for factory functions
_MODEL_FACTORIES: dict[str, Any] = {
    "logistic_regression": _lr,
    "random_forest"      : _rf,
    "xgboost"            : _xgb,
    "lightgbm"           : _lgbm,
}


def get_all_models(track: str = "B") -> dict[str, Any]:
    """
    Return a dict of unfitted model instances for the given track.

    Both tracks use identical model architectures; the difference is
    in the input feature set (57 vs 56 features).

    Parameters
    ----------
    track : "A" or "B"

    Returns
    -------
    dict[model_id -> unfitted estimator]
    """
    track = track.upper()
    assert track in ("A", "B"), f"track must be 'A' or 'B', got '{track}'"

    models = {}
    for mid, module in _MODEL_FACTORIES.items():
        models[mid] = module.get_model()
        logger.debug(
            f"Track {track}: registered '{mid}' "
            f"({type(models[mid]).__name__})"
        )

    logger.info(
        f"Model registry: {len(models)} models for Track {track}: "
        f"{list(models.keys())}"
    )
    return models


def get_model_by_id(model_id: str) -> Any:
    """
    Return a single unfitted model instance by ID.

    Parameters
    ----------
    model_id : one of MODEL_IDS

    Returns
    -------
    Unfitted sklearn-compatible estimator
    """
    if model_id not in _MODEL_FACTORIES:
        raise ValueError(
            f"Unknown model_id '{model_id}'. "
            f"Available: {MODEL_IDS}"
        )
    return _MODEL_FACTORIES[model_id].get_model()


def get_library_status() -> dict[str, str]:
    """
    Return the backend library status for each model.

    Returns
    -------
    dict[model_id -> "native" | "fallback"]
    """
    status = {
        "logistic_regression": "sklearn",
        "random_forest"      : "sklearn",
        "xgboost"            : "xgboost"  if _xgb.is_native()  else "sklearn-fallback",
        "lightgbm"           : "lightgbm" if _lgbm.is_native() else "sklearn-fallback",
    }
    for mid, lib in status.items():
        logger.info(f"  {MODEL_DISPLAY_NAMES[mid]:<26}: {lib}")
    return status
