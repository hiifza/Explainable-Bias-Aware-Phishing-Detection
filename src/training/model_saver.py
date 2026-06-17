"""
src/training/model_saver.py
----------------------------
Serialise and load trained models using joblib.

Public API
----------
    save_model(model, path)
    load_model(path)          -> estimator
    save_all_models(models, output_dir, track)
    load_all_models(output_dir, track) -> dict
"""

import sys
from pathlib import Path
from typing  import Any

import joblib

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger          import get_logger
from src.training.model_registry import MODEL_IDS

logger = get_logger(__name__)


def save_model(model: Any, path: str | Path) -> None:
    """
    Serialise a fitted model to disk.

    Parameters
    ----------
    model : fitted sklearn-compatible estimator
    path  : destination .pkl file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    size_kb = path.stat().st_size / 1024
    logger.info(f"Model saved → {path}  ({size_kb:.1f} KB)")


def load_model(path: str | Path) -> Any:
    """
    Load a serialised model from disk.

    Parameters
    ----------
    path : .pkl file path

    Returns
    -------
    Fitted estimator
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    logger.info(f"Model loaded ← {path}")
    return model


def save_all_models(
    models    : dict[str, Any],
    output_dir: str | Path,
    track     : str,
) -> dict[str, Path]:
    """
    Save all models in the dict to track-specific sub-directory.

    Parameters
    ----------
    models     : dict[model_id -> fitted estimator]
    output_dir : base directory (e.g. outputs/models)
    track      : "A" or "B"

    Returns
    -------
    dict[model_id -> saved Path]
    """
    track  = track.upper()
    subdir = Path(output_dir) / f"track_{track}"
    subdir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}
    for model_id, model in models.items():
        path = subdir / f"{model_id}.pkl"
        save_model(model, path)
        saved_paths[model_id] = path

    logger.info(
        f"Saved {len(saved_paths)} Track {track} models → {subdir}"
    )
    return saved_paths


def load_all_models(
    output_dir: str | Path,
    track     : str,
) -> dict[str, Any]:
    """
    Load all models for a given track from disk.

    Parameters
    ----------
    output_dir : base directory (e.g. outputs/models)
    track      : "A" or "B"

    Returns
    -------
    dict[model_id -> fitted estimator]
    """
    track  = track.upper()
    subdir = Path(output_dir) / f"track_{track}"

    models = {}
    for model_id in MODEL_IDS:
        path = subdir / f"{model_id}.pkl"
        if path.exists():
            models[model_id] = load_model(path)
        else:
            logger.warning(f"Model not found, skipping: {path}")

    logger.info(
        f"Loaded {len(models)}/{len(MODEL_IDS)} Track {track} models"
    )
    return models
