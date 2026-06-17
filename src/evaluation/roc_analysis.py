"""
src/evaluation/roc_analysis.py
--------------------------------
ROC and Precision-Recall curve analysis for all trained models.

Public API
----------
    plot_combined_roc(eval_results, track, out_dir)          -> Path
    plot_roc_all_tracks(eval_results, out_dir)                -> Path
    plot_combined_pr(eval_results, track, out_dir)            -> Path
    plot_auc_ranking(metrics_df, out_dir)                     -> Path
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_COLORS = ["#185FA5", "#E24B4A", "#0F6E56", "#854F0B",
                "#533AB7", "#993C1D", "#993556", "#444441"]


def _setup():
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({"figure.facecolor": "white", "savefig.dpi": 150})


# ── Combined ROC per track ────────────────────────────────────────────────────

def plot_combined_roc(
    eval_results: list[dict],
    track       : str,
    out_dir     : Path,
) -> Path:
    """
    Plot ROC curves for all models in one track on a single axis.

    Parameters
    ----------
    eval_results : list of result dicts with 'y_true', 'y_proba',
                   'model', 'track', 'roc_auc'
    track        : "A" or "B"
    out_dir      : destination directory

    Returns
    -------
    Path to saved PNG
    """
    _setup()
    track_results = [r for r in eval_results
                     if r.get("track", "").upper() == track.upper()]

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, r in enumerate(track_results):
        fpr, tpr, _ = roc_curve(
            np.asarray(r["y_true"]),
            np.asarray(r["y_proba"])
        )
        roc_auc_val = float(r.get("roc_auc", auc(fpr, tpr)))
        ax.plot(fpr, tpr,
                label=f"{r['model']} (AUC={roc_auc_val:.4f})",
                color=MODEL_COLORS[i % len(MODEL_COLORS)],
                linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.5000)")
    ax.fill_between([0, 1], [0, 1], alpha=0.03, color="gray")
    ax.set_xlabel("False Positive Rate",  fontsize=11)
    ax.set_ylabel("True Positive Rate",   fontsize=11)
    ax.set_title(f"ROC Curves — Track {track.upper()}",
                 fontsize=13, fontweight="700")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    sns.despine(ax=ax)
    plt.tight_layout()

    out_path = out_dir / f"roc_track{track.upper()}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")
    return out_path


def plot_roc_all_tracks(
    eval_results: list[dict],
    out_dir     : Path,
) -> Path:
    """
    Side-by-side ROC panels for Track A and Track B.

    Returns
    -------
    Path to saved PNG
    """
    _setup()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, track in zip(axes, ["A", "B"]):
        track_results = [r for r in eval_results
                         if r.get("track", "").upper() == track]
        for i, r in enumerate(track_results):
            fpr, tpr, _ = roc_curve(
                np.asarray(r["y_true"]),
                np.asarray(r["y_proba"])
            )
            ax.plot(fpr, tpr,
                    label=f"{r['model']} ({r.get('roc_auc', 0):.4f})",
                    color=MODEL_COLORS[i % len(MODEL_COLORS)], linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_title(f"ROC — Track {track}", fontsize=12, fontweight="700")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        sns.despine(ax=ax)

    plt.suptitle("ROC Curve Comparison — All Models",
                 fontsize=14, fontweight="700")
    plt.tight_layout()
    out_path = out_dir / "roc_all_tracks.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")
    return out_path


# ── Precision-Recall curves ───────────────────────────────────────────────────

def plot_combined_pr(
    eval_results: list[dict],
    track       : str,
    out_dir     : Path,
) -> Path:
    """
    Combined Precision-Recall curves for all models in one track.

    Returns
    -------
    Path to saved PNG
    """
    _setup()
    track_results = [r for r in eval_results
                     if r.get("track", "").upper() == track.upper()]

    fig, ax = plt.subplots(figsize=(7, 6))
    baseline = float(np.mean(np.asarray(track_results[0]["y_true"]) == 1)) \
        if track_results else 0.5

    for i, r in enumerate(track_results):
        y_true  = np.asarray(r["y_true"])
        y_proba = np.asarray(r["y_proba"])
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        ap_val = float(r.get("pr_auc",
                             average_precision_score(y_true, y_proba)))
        ax.plot(rec, prec,
                label=f"{r['model']} (AP={ap_val:.4f})",
                color=MODEL_COLORS[i % len(MODEL_COLORS)],
                linewidth=2)

    ax.axhline(baseline, color="k", linestyle="--", linewidth=1,
               label=f"Baseline (={baseline:.3f})")
    ax.set_xlabel("Recall",    fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(f"Precision-Recall Curves — Track {track.upper()}",
                 fontsize=13, fontweight="700")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    sns.despine(ax=ax)
    plt.tight_layout()

    out_path = out_dir / f"pr_track{track.upper()}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")
    return out_path


# ── AUC ranking chart ─────────────────────────────────────────────────────────

def plot_auc_ranking(
    metrics_df: pd.DataFrame,
    out_dir   : Path,
) -> Path:
    """
    Horizontal bar chart ranking all 8 models by ROC AUC.

    Parameters
    ----------
    metrics_df : DataFrame with columns model, track, roc_auc, pr_auc
    out_dir    : destination directory

    Returns
    -------
    Path to saved PNG
    """
    _setup()
    df = metrics_df.copy()
    df["model_track"] = df["model"] + " (Track " + df["track"] + ")"
    df = df.sort_values("roc_auc", ascending=True)

    colors = ["#E24B4A" if t == "A" else "#1D9E75"
              for t in df["track"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in [
        (axes[0], "roc_auc", "ROC AUC Ranking"),
        (axes[1], "pr_auc",  "PR AUC Ranking"),
    ]:
        bars = ax.barh(df["model_track"], df[metric],
                       color=colors, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, df[metric]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}", va="center", fontsize=9, fontweight="500")
        ax.set_xlabel(metric.upper().replace("_", " "))
        ax.set_title(title, fontsize=12, fontweight="700")
        ax.set_xlim(max(0, df[metric].min() - 0.05), 1.04)

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="#E24B4A", label="Track A (with URLSimilarityIndex)"),
            Patch(facecolor="#1D9E75", label="Track B (leakage-aware)"),
        ], fontsize=8, loc="lower right")
        sns.despine(ax=ax)

    plt.suptitle("Model AUC Rankings — All 8 Models",
                 fontsize=14, fontweight="700")
    plt.tight_layout()
    out_path = out_dir / "auc_ranking.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")
    return out_path
