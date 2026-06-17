"""
src/models/random_forest.py
---------------------------
Random Forest model definition.

Public API
----------
get_model() -> RandomForestClassifier
is_native() -> bool
get_metadata() -> dict
"""

from sklearn.ensemble import RandomForestClassifier

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)


MODEL_NAME = "Random Forest"
MODEL_ID = "random_forest"


def get_model():
    """
    Returns an unfitted Random Forest model.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    logger.info(f"{MODEL_NAME} initialized")

    return model


def is_native() -> bool:
    return True


def get_metadata() -> dict:
    return {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "library": "scikit-learn"
    }