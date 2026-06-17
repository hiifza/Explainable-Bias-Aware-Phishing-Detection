"""
src/evaluation/confusion_analysis.py
--------------------------------------
Confusion matrix analysis and visualisation for all trained models.

Each model gets a 2-panel figure:
  [0] Raw confusion matrix          (absolute counts)
  [1] Normalised confusion matrix   (row-normalised proportions)

Public API
----------
    plot_single_confusion_matrix(y_true, y_pred, model_name, track, out_dir) -> Path
    plot_all_confusion_matrices(eval_results, out_dir)                        -> list[Path]
    plot_combined_confusion_grid(eval_results, track, out_dir)                -> Path
"""

import sys
from pathlib import Path
from typing  import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASS_LABELS = ["Phishing\n(0)", "Legitimate\n(1)"]


def _setup():
    sns.set_theme(style="white", font_scale=1.05)
    plt.rcParams.update({"figure.facecolor": "white", "savefig.dpi": 150})


def plot_single_confusion_matrix(
    y_true    : np.ndarray,
    y_pred    : np.ndarray,
    model_name: str,
    track     : str,
    out_dir   : Path,
) -> Path:
    """
    Generate and save a 2-panel (raw + normalised) confusion matrix figure.

    Parameters
    ----------
    y_true     : true labels
    y_pred     : predicted labels
    model_name : display name
    track      : "A" or "B"
    out_dir    : destination directory

    Returns
    -------
    Path to saved PNG
    """
    _setup()
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Confusion Matrix — {model_name} (Track {track.upper()})",
                 fontsize=12, fontweight="700", y=1.02)

    for ax, matrix, title, fmt, vmax, cmap in [
        (axes[0], cm,      "Raw counts",        "d",    cm.max(),  "Blues"),
        (axes[1], cm_norm, "Row-normalised",     ".3f",  1.0,       "Greens"),
    ]:
        sns.heatmap(
            matrix, ax=ax, annot=True, fmt=fmt,
            cmap=cmap, vmin=0, vmax=vmax,
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
            linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.75},
            annot_kws={"size": 11, "weight": "bold"},
        )
        ax.set_title(title, fontsize=11, fontweight="600")
        ax.set_ylabel("True label",      fontsize=10)
        ax.set_xlabel("Predicted label", fontsize=10)

    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    out_path  = out_dir / f"cm_{safe_name}_track{track.upper()}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.debug(f"Saved: {out_path.name}")
    return out_path


def plot_all_confusion_matrices(
    eval_results: list[dict],
    out_dir     : Path,
) -> list[Path]:
    """
    Generate individual confusion matrix figures for every model/track result.

    Parameters
    ----------
    eval_results : list of dicts, each containing:
                   'model', 'track', 'y_true', 'y_pred'
    out_dir      : destination directory

    Returns
    -------
    list of saved PNG paths
    """
    saved = []
    for r in eval_results:
        p = plot_single_confusion_matrix(
            y_true     = np.asarray(r["y_true"]),
            y_pred     = np.asarray(r["y_pred"]),
            model_name = r["model"],
            track      = r["track"],
            out_dir    = out_dir,
        )
        saved.append(p)

    logger.info(
        f"Confusion matrices saved: {len(saved)} figures → {out_dir}"
    )
    return saved


def plot_combined_confusion_grid(
    eval_results: list[dict],
    track       : str,
    out_dir     : Path,
) -> Path:
    """
    4-panel grid of raw confusion matrices for all models in one track.

    Parameters
    ----------
    eval_results : all results (filtered to given track internally)
    track        : "A" or "B"
    out_dir      : destination directory

    Returns
    -------
    Path to saved PNG
    """
    _setup()
    track_results = [r for r in eval_results
                     if r.get("track", "").upper() == track.upper()]

    n     = len(track_results)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 4.5, nrows * 4.2))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    fig.suptitle(f"Confusion Matrices — Track {track.upper()}",
                 fontsize=14, fontweight="700", y=1.01)

    for ax, r in zip(axes_flat, track_results):
        cm = confusion_matrix(np.asarray(r["y_true"]), np.asarray(r["y_pred"]))
        sns.heatmap(cm, ax=ax, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                    linewidths=0.5, linecolor="white", cbar=False,
                    annot_kws={"size": 11, "weight": "bold"})
        ax.set_title(r["model"], fontsize=11, fontweight="600")
        ax.set_ylabel("True", fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    out_path = out_dir / f"cm_grid_track{track.upper()}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")
    return out_path
