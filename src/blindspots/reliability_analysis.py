"""
src/blindspots/reliability_analysis.py
-----------------------------------------
Determines whether SHAP-LIME disagreement correlates with increased
error probability, uncertainty, and overall deployment risk.

Questions answered
------------------
Q1: Does low agreement → higher error rate?
Q2: Does low agreement → lower confidence?
Q3: Does low agreement → higher severity score?

Public API
----------
    compute_agreement_reliability(agreement_df, severity_df, zone_stats) -> dict
    plot_agreement_vs_error(agreement_df, plots_dir)                      -> Path
    plot_reliability_summary(reliability_stats, plots_dir)                -> Path
    run_reliability_analysis(agreement_df, severity_df, zone_stats,
                              plots_dir, reports_dir)                     -> dict
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

AGREE_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
AGREE_LABELS = ["0.0-0.2","0.2-0.4","0.4-0.6","0.6-0.8","0.8-1.0"]


def _setup():
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({"figure.facecolor":"white","savefig.dpi":150})


def compute_agreement_reliability(
    agreement_df : pd.DataFrame,
    severity_df  : pd.DataFrame,
    zone_stats   : dict,
) -> dict:
    """
    Compute correlation between SHAP-LIME agreement and error/uncertainty.

    Returns
    -------
    dict  keys: q1_answer, q2_answer, q3_answer, bin_stats_df,
               corr_agreement_severity, corr_agreement_confidence
    """
    if agreement_df.empty or "agreement_score" not in agreement_df.columns:
        return {"q1_answer":"No agreement data","q2_answer":"—","q3_answer":"—"}

    # Merge with severity
    has_agree = (not agreement_df.empty) and ("sample_id" in agreement_df.columns)
    has_sev   = (not severity_df.empty) and ("sample_idx" in severity_df.columns)
    if has_agree and has_sev:
        sev_cols = [c for c in ["sample_idx","severity_score_norm","confidence","is_error"] if c in severity_df.columns]
        merged = agreement_df.merge(severity_df[sev_cols], left_on="sample_id", right_on="sample_idx", how="inner")
    elif has_agree:
        merged = agreement_df.copy()
    else:
        merged = pd.DataFrame()

    scores = merged["agreement_score"].fillna(0.5)

    # Bin by agreement level
    merged["agree_bin"] = pd.cut(scores, bins=AGREE_BINS, labels=AGREE_LABELS, right=False)

    bin_stats = (
        merged.groupby("agree_bin", observed=True)
        .agg(
            n            = ("agreement_score","count"),
            mean_agree   = ("agreement_score","mean"),
            error_rate   = ("is_error","mean") if "is_error" in merged.columns else ("agreement_score","count"),
            mean_confidence = ("confidence","mean") if "confidence" in merged.columns else ("agreement_score","mean"),
            mean_severity= ("severity_score_norm","mean") if "severity_score_norm" in merged.columns else ("agreement_score","mean"),
        )
        .reset_index()
    )

    # Correlations
    agree_col = merged["agreement_score"]
    corr_sev  = float(agree_col.corr(merged["severity_score_norm"])) \
        if "severity_score_norm" in merged.columns and len(merged) > 3 else 0.0
    corr_conf = float(agree_col.corr(merged["confidence"])) \
        if "confidence" in merged.columns and len(merged) > 3 else 0.0

    # Q1: Does low agreement → higher error rate?
    if "is_error" in merged.columns:
        low_agree  = merged[merged["agreement_score"] < 0.4]["is_error"].mean()
        high_agree = merged[merged["agreement_score"] >= 0.6]["is_error"].mean()
        q1 = f"Low-agree error rate={low_agree:.4f}  High-agree error rate={high_agree:.4f}  " \
             f"({'Yes, risk higher' if low_agree>high_agree else 'No clear effect'})"
    else:
        q1 = "Error data not available in agreement_df"

    # Q2: Does low agreement → lower confidence?
    if "confidence" in merged.columns:
        low_conf  = merged[merged["agreement_score"] < 0.4]["confidence"].mean()
        high_conf = merged[merged["agreement_score"] >= 0.6]["confidence"].mean()
        q2 = f"Low-agree mean confidence={low_conf:.4f}  High-agree={high_conf:.4f}  " \
             f"({'Yes, less confident' if low_conf<high_conf else 'No clear effect'})"
    else:
        q2 = "Confidence data not available"

    # Q3: Does low agreement → higher severity?
    if "severity_score_norm" in merged.columns:
        low_sev  = merged[merged["agreement_score"] < 0.4]["severity_score_norm"].mean()
        high_sev = merged[merged["agreement_score"] >= 0.6]["severity_score_norm"].mean()
        q3 = f"Low-agree mean severity={low_sev:.4f}  High-agree={high_sev:.4f}  " \
             f"({'Yes, higher risk' if low_sev>high_sev else 'No clear effect'})"
    else:
        q3 = "Severity data not available"

    logger.info(f"Reliability Q1: {q1}")
    logger.info(f"Reliability Q2: {q2}")
    logger.info(f"Reliability Q3: {q3}")

    return {
        "q1_answer"              : q1,
        "q2_answer"              : q2,
        "q3_answer"              : q3,
        "bin_stats_df"           : bin_stats,
        "corr_agreement_severity": round(corr_sev,  4),
        "corr_agreement_confidence": round(corr_conf, 4),
        "merged_df"              : merged,
    }


def plot_agreement_vs_error(
    agreement_df: pd.DataFrame,
    severity_df : pd.DataFrame,
    plots_dir   : Path,
) -> Path:
    """3-panel: agreement distribution, agreement vs severity scatter, bin error rates."""
    _setup()
    has_agree = (not agreement_df.empty) and ("sample_id" in agreement_df.columns) and ("agreement_score" in agreement_df.columns)
    has_sev   = (not severity_df.empty) and ("sample_idx" in severity_df.columns)
    # Early return if no agreement data to plot
    if not has_agree:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No SHAP-LIME agreement data available",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        out = plots_dir / "agreement_reliability.png"
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out
    if has_agree and has_sev:
        sev_cols = [c for c in ["sample_idx","severity_score_norm","confidence","is_error"] if c in severity_df.columns]
        merged = agreement_df.merge(severity_df[sev_cols], left_on="sample_id", right_on="sample_idx", how="inner")
    elif has_agree:
        merged = agreement_df.copy()
    else:
        merged = pd.DataFrame()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("SHAP-LIME Agreement vs Error & Uncertainty",
                 fontsize=13, fontweight="700")

    # Panel 0: Agreement score histogram
    ax = axes[0]
    ax.hist(agreement_df["agreement_score"].fillna(0.5), bins=10, range=(0,1),
            color="#185FA5", edgecolor="white", linewidth=0.8)
    ax.axvline(0.4, color="#E24B4A", linestyle="--", linewidth=1.5,
               label="High-disagree threshold")
    ax.set_xlabel("Agreement score"); ax.set_ylabel("Count")
    ax.set_title("Agreement Distribution"); ax.legend(fontsize=8); sns.despine(ax=ax)

    # Panel 1: Agreement vs severity scatter
    ax = axes[1]
    if "severity_score_norm" in merged.columns:
        sc = ax.scatter(merged["agreement_score"], merged["severity_score_norm"],
                        c=merged["confidence"] if "confidence" in merged.columns else "#185FA5",
                        cmap="RdYlGn", alpha=0.6, s=20, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="Confidence")
    ax.set_xlabel("Agreement score"); ax.set_ylabel("Severity score")
    ax.set_title("Agreement vs Severity"); sns.despine(ax=ax)

    # Panel 2: Binned error rates
    ax = axes[2]
    if "is_error" in merged.columns:
        merged["agree_bin"] = pd.cut(merged["agreement_score"].fillna(0.5),
                                     bins=AGREE_BINS, labels=AGREE_LABELS, right=False)
        bin_err = merged.groupby("agree_bin", observed=True)["is_error"].mean()
        colors  = ["#E24B4A" if v > 0.01 else "#1D9E75" for v in bin_err.values]
        ax.bar(AGREE_LABELS[:len(bin_err)], bin_err.values,
               color=colors, edgecolor="white")
        for i, v in enumerate(bin_err.values):
            ax.text(i, v+0.001, f"{v:.4f}", ha="center", fontsize=8)
    ax.set_xlabel("Agreement bin"); ax.set_ylabel("Error rate")
    ax.set_title("Error Rate by Agreement Level"); sns.despine(ax=ax)

    plt.tight_layout()
    out = plots_dir / "agreement_reliability.png"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig); logger.info(f"Saved: {out.name}"); return out


def run_reliability_analysis(
    agreement_df: pd.DataFrame,
    severity_df : pd.DataFrame,
    zone_stats  : dict,
    plots_dir   : str | Path = "outputs/plots/blindspot/reliability",
    reports_dir : str | Path = "outputs/reports",
) -> dict:
    """Full reliability analysis pipeline."""
    plots_dir   = Path(plots_dir); reports_dir = Path(reports_dir)
    plots_dir.mkdir(parents=True, exist_ok=True); reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("M10 — SHAP-LIME RELIABILITY ANALYSIS")

    rel_stats = compute_agreement_reliability(agreement_df, severity_df, zone_stats)
    agree_plot= plot_agreement_vs_error(agreement_df, severity_df, plots_dir)

    # Save bin stats
    if "bin_stats_df" in rel_stats and not rel_stats["bin_stats_df"].empty:
        rel_stats["bin_stats_df"].to_csv(reports_dir/"reliability_bin_stats.csv", index=False)
        logger.info("Saved: reliability_bin_stats.csv")

    return {
        "reliability_stats" : rel_stats,
        "agreement_plot"    : agree_plot,
    }
