"""
src/models/logistic_regression.py
---------------------------------
Logistic Regression model definition.

Public API
----------
get_model() -> LogisticRegression
is_native() -> bool
get_metadata() -> dict
"""

from sklearn.linear_model import LogisticRegression

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)


MODEL_NAME = "Logistic Regression"
MODEL_ID = "logistic_regression"


def get_model():
    """
    Returns an unfitted Logistic Regression model.
    """

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
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