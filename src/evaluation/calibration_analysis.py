"""
src/evaluation/calibration_analysis.py
----------------------------------------
Probability calibration analysis for all trained models.

A well-calibrated model predicts P(y=1|X=x) ≈ x for all x.
The reliability diagram visualises the gap between predicted
probabilities and observed class frequencies.

Calibration quality score = 1 − mean_absolute_error(
    fraction_of_positives, mean_predicted_value
)
Higher is better; 1.0 is perfect calibration.

Public API
----------
    compute_calibration_curve(y_true, y_proba, n_bins)  -> dict
    calibration_quality_score(y_true, y_proba)           -> float
    plot_calibration_curves(eval_results, track, out_dir) -> Path
    plot_probability_distribution(eval_results, track, out_dir) -> Path
    plot_calibration_comparison(eval_results, out_dir)    -> Path
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_COLORS = ["#185FA5", "#E24B4A", "#0F6E56", "#854F0B"]


def _setup():
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({"figure.facecolor": "white", "savefig.dpi": 150})


def compute_calibration_curve(
    y_true : np.ndarray,
    y_proba: np.ndarray,
    n_bins : int = 10,
) -> dict:
    """
    Compute the reliability diagram data.

    Returns
    -------
    dict  keys: fraction_of_positives, mean_predicted_value,
               calibration_quality, n_bins
    """
    fop, mpv = calibration_curve(y_true, y_proba,
                                  n_bins=n_bins, strategy="uniform")
    # Calibration quality: 1 - MAE between observed and predicted
    quality = float(1.0 - np.mean(np.abs(fop - mpv)))

    return {
        "fraction_of_positives": fop,
        "mean_predicted_value" : mpv,
        "calibration_quality"  : round(quality, 6),
        "n_bins"               : n_bins,
    }


def calibration_quality_score(
    y_true : np.ndarray,
    y_proba: np.ndarray,
    n_bins : int = 10,
) -> float:
    """Return a scalar calibration quality score in [0, 1]."""
    return compute_calibration_curve(y_true, y_proba, n_bins)["calibration_quality"]


def plot_calibration_curves(
    eval_results: list[dict],
    track       : str,
    out_dir     : Path,
) -> Path:
    """
    Reliability diagrams for all models in one track.

    Each panel shows observed positive fraction vs mean predicted probability
    with a perfectly calibrated diagonal for reference.

    Parameters
    ----------
    eval_results : list of result dicts with y_true, y_proba, model, track
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
    ncols = min(n, 2)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 6, nrows * 5))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    fig.suptitle(f"Reliability Diagrams — Track {track.upper()}",
                 fontsize=13, fontweight="700", y=1.01)

    for ax, r in zip(axes_flat, track_results):
        y_t = np.asarray(r["y_true"])
        y_p = np.asarray(r["y_proba"])

        try:
            cal = compute_calibration_curve(y_t, y_p, n_bins=10)
            ax.plot(cal["mean_predicted_value"],
                    cal["fraction_of_positives"],
                    "s-", color="#185FA5", linewidth=2,
                    markersize=6, label="Model calibration")
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
            ax.fill_between(
                cal["mean_predicted_value"],
                cal["fraction_of_positives"],
                cal["mean_predicted_value"],
                alpha=0.15, color="#185FA5",
            )
            ax.set_title(
                f"{r['model']}\n"
                f"Quality={cal['calibration_quality']:.4f}",
                fontsize=10, fontweight="600",
            )
        except Exception as e:
            ax.text(0.5, 0.5, f"Calibration error:\n{e}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.08)
        ax.legend(fontsize=8)
        sns.despine(ax=ax)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    out_path = out_dir / f"calibration_track{track.upper()}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")
    return out_path


def plot_probability_distribution(
    eval_results: list[dict],
    track       : str,
    out_dir     : Path,
) -> Path:
    """
    KDE plots of predicted probabilities split by true class.

    Well-separated distributions indicate confident, useful models.

    Returns
    -------
    Path to saved PNG
    """
    _setup()
    track_results = [r for r in eval_results
                     if r.get("track", "").upper() == track.upper()]

    n     = len(track_results)
    ncols = min(n, 2)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 6, nrows * 4))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    fig.suptitle(f"Predicted Probability Distributions — Track {track.upper()}",
                 fontsize=13, fontweight="700", y=1.01)

    for ax, r in zip(axes_flat, track_results):
        y_t = np.asarray(r["y_true"])
        y_p = np.asarray(r["y_proba"])

        ph_proba = y_p[y_t == 0]
        lg_proba = y_p[y_t == 1]

        try:
            ax.hist(ph_proba, bins=50, density=True, alpha=0.6,
                    color="#E24B4A", label=f"Phishing (n={len(ph_proba):,})",
                    edgecolor="white", linewidth=0.3)
            ax.hist(lg_proba, bins=50, density=True, alpha=0.6,
                    color="#1D9E75", label=f"Legitimate (n={len(lg_proba):,})",
                    edgecolor="white", linewidth=0.3)
        except Exception:
            pass

        ax.set_title(r["model"], fontsize=10, fontweight="600")
        ax.set_xlabel("P(Legitimate)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.axvline(0.5, color="#888", linestyle="--", linewidth=1)
        sns.despine(ax=ax)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    out_path = out_dir / f"prob_dist_track{track.upper()}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")
    return out_path


def plot_calibration_comparison(
    eval_results: list[dict],
    out_dir     : Path,
) -> Path:
    """
    Bar chart comparing calibration quality scores across all 8 models.

    Returns
    -------
    Path to saved PNG
    """
    _setup()
    rows = []
    for r in eval_results:
        try:
            score = calibration_quality_score(
                np.asarray(r["y_true"]),
                np.asarray(r["y_proba"]),
            )
        except Exception:
            score = 0.0
        rows.append({
            "model": r["model"],
            "track": r["track"],
            "calibration_quality": score,
        })

    df    = pd.DataFrame(rows).sort_values("calibration_quality", ascending=True)
    colors = ["#E24B4A" if t == "A" else "#1D9E75" for t in df["track"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    model_labels = df["model"] + " (Track " + df["track"] + ")"
    bars = ax.barh(model_labels, df["calibration_quality"],
                   color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, df["calibration_quality"]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9, fontweight="500")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#E24B4A", label="Track A"),
        Patch(facecolor="#1D9E75", label="Track B (primary)"),
    ], fontsize=9)
    ax.set_xlabel("Calibration Quality Score (1 = perfect)")
    ax.set_title("Calibration Quality — All Models",
                 fontsize=13, fontweight="700")
    ax.set_xlim(0, 1.1)
    sns.despine(ax=ax)
    plt.tight_layout()

    out_path = out_dir / "calibration_comparison.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")
    return out_path
