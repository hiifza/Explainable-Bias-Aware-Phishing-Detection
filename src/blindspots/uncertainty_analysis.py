"""
src/blindspots/uncertainty_analysis.py
----------------------------------------
Confidence Reliability Engine:

  Green Zone  → P(class) ≥ 0.95 : high confidence, very low error probability
  Yellow Zone → P(class) ∈ [0.75, 0.95) : moderate uncertainty
  Red Zone    → P(class) < 0.75 : high uncertainty / deployment risk

Computes per-zone:
  - Sample count and proportion
  - Empirical error rate within zone
  - Mean confidence
  - Calibration quality (predicted confidence vs empirical accuracy)

Public API
----------
    assign_confidence_zones(y_proba) -> np.ndarray[str]
    compute_zone_stats(y_true, y_pred, y_proba) -> dict
    plot_confidence_distribution(y_proba, y_true, y_pred, plots_dir) -> Path
    plot_zone_error_rates(zone_stats, plots_dir) -> Path
    run_uncertainty_analysis(y_true, y_pred, y_proba, plots_dir) -> dict
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

GREEN_THRESH  = 0.95
YELLOW_THRESH = 0.75

ZONE_COLORS = {"green": "#1D9E75", "yellow": "#EF9F27", "red": "#E24B4A"}


def _setup():
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({"figure.facecolor": "white", "savefig.dpi": 150})


def assign_confidence_zones(y_proba: np.ndarray) -> np.ndarray:
    """Assign confidence zones based on max class probability."""
    confidence = np.maximum(y_proba, 1 - y_proba)
    zones      = np.where(confidence >= GREEN_THRESH, "green",
                 np.where(confidence >= YELLOW_THRESH, "yellow", "red"))
    return zones


def compute_zone_stats(
    y_true : np.ndarray,
    y_pred : np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """
    Compute statistics for each confidence zone.

    Returns
    -------
    dict  with keys 'green', 'yellow', 'red', each containing:
        n, pct, mean_confidence, error_rate, n_errors,
        n_fp, n_fn, calibration_gap
    """
    zones      = assign_confidence_zones(y_proba)
    confidence = np.maximum(y_proba, 1 - y_proba)
    n_total    = len(y_true)

    stats: dict = {}
    for zone in ("green", "yellow", "red"):
        mask  = zones == zone
        n     = int(mask.sum())
        if n == 0:
            stats[zone] = {"zone": zone, "n": 0, "pct": 0.0, "mean_confidence": 0.0,
                           "error_rate": 0.0, "n_errors": 0, "n_fp": 0, "n_fn": 0,
                           "calibration_gap": 0.0}
            continue
        yt    = y_true[mask]; yp = y_pred[mask]; ypr = y_proba[mask]
        conf  = confidence[mask]
        n_err = int((yt != yp).sum())
        n_fp  = int(((yp==0) & (yt==1)).sum())
        n_fn  = int(((yp==1) & (yt==0)).sum())
        err_rate = n_err / max(n, 1)
        # Calibration gap: mean confidence - empirical accuracy
        emp_acc  = 1.0 - err_rate
        calib_gap= float(conf.mean()) - emp_acc
        stats[zone] = {
            "zone"           : zone,
            "n"              : n,
            "pct"            : round(n / n_total * 100, 4),
            "mean_confidence": round(float(conf.mean()), 6),
            "error_rate"     : round(err_rate, 6),
            "n_errors"       : n_err,
            "n_fp"           : n_fp,
            "n_fn"           : n_fn,
            "calibration_gap": round(calib_gap, 6),
        }
        logger.info(
            f"  {zone.upper():<6}: n={n:>6,} ({n/n_total*100:.2f}%)  "
            f"error_rate={err_rate:.6f}  mean_conf={conf.mean():.4f}"
        )
    return stats


def plot_confidence_distribution(
    y_proba  : np.ndarray,
    y_true   : np.ndarray,
    y_pred   : np.ndarray,
    plots_dir: Path,
) -> Path:
    """
    3-panel confidence distribution plot:
    [0] Overall histogram with zone coloring
    [1] Confidence by class (phishing vs legitimate)
    [2] Confidence of correct vs incorrect predictions
    """
    _setup()
    confidence = np.maximum(y_proba, 1 - y_proba)
    errors     = y_true != y_pred

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Confidence Distribution Analysis — Confidence Reliability Engine",
                 fontsize=13, fontweight="700")

    # Panel 0: Overall histogram with zone shading
    ax = axes[0]
    ax.hist(confidence, bins=50, color="#185FA5", alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvspan(0.00, YELLOW_THRESH, alpha=0.12, color="#E24B4A",  label=f"Red zone (<{YELLOW_THRESH})")
    ax.axvspan(YELLOW_THRESH, GREEN_THRESH, alpha=0.12, color="#EF9F27", label=f"Yellow zone ({YELLOW_THRESH}-{GREEN_THRESH})")
    ax.axvspan(GREEN_THRESH, 1.01, alpha=0.08, color="#1D9E75",  label=f"Green zone (≥{GREEN_THRESH})")
    ax.set_xlabel("Confidence = max(P(legit), P(phish))")
    ax.set_ylabel("Count")
    ax.set_title("Overall Confidence Distribution")
    ax.legend(fontsize=8); sns.despine(ax=ax)

    # Panel 1: By true class
    ax = axes[1]
    ph_conf = confidence[y_true == 0]
    lg_conf = confidence[y_true == 1]
    ax.hist(ph_conf, bins=40, alpha=0.65, color="#E24B4A",
            label=f"Phishing (n={len(ph_conf):,})", density=True, edgecolor="white")
    ax.hist(lg_conf, bins=40, alpha=0.65, color="#1D9E75",
            label=f"Legitimate (n={len(lg_conf):,})", density=True, edgecolor="white")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Density")
    ax.set_title("Confidence by True Class"); ax.legend(fontsize=8); sns.despine(ax=ax)

    # Panel 2: Correct vs incorrect
    ax = axes[2]
    correct_conf = confidence[~errors]
    wrong_conf   = confidence[errors]
    ax.hist(correct_conf, bins=40, alpha=0.65, color="#1D9E75",
            label=f"Correct (n={len(correct_conf):,})", density=True, edgecolor="white")
    if len(wrong_conf) > 0:
        ax.hist(wrong_conf, bins=max(5, len(wrong_conf)//2), alpha=0.85, color="#E24B4A",
                label=f"Errors (n={len(wrong_conf):,})", edgecolor="white")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Density")
    ax.set_title("Confidence: Correct vs Incorrect")
    ax.legend(fontsize=8); sns.despine(ax=ax)

    plt.tight_layout()
    out = plots_dir / "confidence_distribution.png"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out.name}")
    return out


def plot_zone_error_rates(
    zone_stats: dict,
    plots_dir : Path,
) -> Path:
    """Bar chart: error rate and sample count per confidence zone."""
    _setup()
    zones  = ["red", "yellow", "green"]
    counts = [zone_stats[z]["n"]          for z in zones]
    errors = [zone_stats[z]["error_rate"] for z in zones]
    confs  = [zone_stats[z]["mean_confidence"] for z in zones]
    colors = [ZONE_COLORS[z] for z in zones]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Confidence Zone Analysis", fontsize=12, fontweight="700")

    ax = axes[0]
    bars = ax.bar(["Red (<0.75)","Yellow (0.75-0.95)","Green (≥0.95)"],
                  counts, color=colors, edgecolor="white", linewidth=0.8)
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                f"{n:,}", ha="center", fontsize=9, fontweight="500")
    ax.set_ylabel("Sample count"); ax.set_title("Samples per Zone")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    sns.despine(ax=ax)

    ax = axes[1]
    bars2 = ax.bar(["Red","Yellow","Green"], errors, color=colors,
                   edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars2, errors):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0002,
                f"{v:.4f}", ha="center", fontsize=9, fontweight="500")
    ax.set_ylabel("Empirical error rate"); ax.set_title("Error Rate per Zone")
    sns.despine(ax=ax)

    plt.tight_layout()
    out = plots_dir / "zone_error_rates.png"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out.name}")
    return out


def run_uncertainty_analysis(
    y_true   : np.ndarray,
    y_pred   : np.ndarray,
    y_proba  : np.ndarray,
    plots_dir: str | Path = "outputs/plots/blindspot/confidence",
) -> dict:
    """
    Full confidence reliability analysis.

    Returns
    -------
    dict  keys: zone_stats, confidence_dist_plot, zone_error_plot,
               zones (np.ndarray of zone labels per sample),
               confidence (np.ndarray per sample)
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 55)
    logger.info("M10 — CONFIDENCE RELIABILITY ENGINE")
    logger.info("=" * 55)

    zone_stats = compute_zone_stats(y_true, y_pred, y_proba)
    zones      = assign_confidence_zones(y_proba)
    confidence = np.maximum(y_proba, 1 - y_proba)

    conf_plot  = plot_confidence_distribution(y_proba, y_true, y_pred, plots_dir)
    zone_plot  = plot_zone_error_rates(zone_stats, plots_dir)

    return {
        "zone_stats"          : zone_stats,
        "confidence_dist_plot": conf_plot,
        "zone_error_plot"     : zone_plot,
        "zones"               : zones,
        "confidence"          : confidence,
    }
