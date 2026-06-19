"""
src/bias/performance_investigator.py
--------------------------------------
Investigates and visualises why the phishing detection model achieves
near-perfect performance (ROC-AUC ≈ 1.0, Accuracy ≈ 99.98%).

Seven required visualisations
------------------------------
1.  Top Feature Dominance Chart          — top-10 SHAP relative contributions
2.  Dataset Separability Chart           — class purity of dominant features
3.  Feature Dominance vs Remaining       — top-3 / top-10 / all comparison
4.  Cumulative SHAP Importance Curve     — top 1/3/5/10/20/all
5.  Feature Contribution Distribution    — |SHAP| histogram showing concentration
6.  URLSimilarityIndex Impact            — per-class USI distribution
7.  HTTPS Impact                         — IsHTTPS per-class analysis

Public API
----------
    investigate_near_perfect_performance(shap_result, X_test_raw, y_true,
                                          y_pred, y_proba, plots_dir) -> dict
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy  as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _setup():
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({"figure.facecolor": "white", "savefig.dpi": 150})


def _save(fig, path: Path, name: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {name}")
    return path


# ── 1. Top Feature Dominance Chart ───────────────────────────────────────────

def plot_top_feature_dominance(
    shap_values  : np.ndarray,
    feature_names: list[str],
    plots_dir    : Path,
    top_n        : int = 10,
) -> Path:
    """Bar chart of top-N SHAP features with percentage relative contribution."""
    _setup()
    mean_abs  = np.abs(shap_values).mean(axis=0)
    total     = mean_abs.sum()
    order     = np.argsort(mean_abs)[::-1][:top_n]
    top_feats = [feature_names[i] for i in order]
    top_vals  = mean_abs[order]
    top_pct   = top_vals / max(total, 1e-12) * 100
    others_pct= (mean_abs[np.argsort(mean_abs)[::-1][top_n:]].sum() / max(total,1e-12))*100

    colors = ["#E24B4A" if i==0 else "#EF9F27" if i<3 else "#378ADD"
              for i in range(top_n)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(top_n), top_pct, color=colors,
                  edgecolor="white", linewidth=0.8)
    for i, (bar, pct) in enumerate(zip(bars, top_pct)):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3, f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="600")

    ax.set_xticks(range(top_n))
    ax.set_xticklabels(top_feats, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Relative contribution (%)")
    ax.set_title(
        f"Top-{top_n} Feature Dominance (SHAP)\n"
        f"Top-{top_n} explain {top_pct.sum():.1f}% of total model output; "
        f"remaining {others_pct:.1f}% spread across {len(feature_names)-top_n} features",
        fontsize=12, fontweight="700",
    )

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#E24B4A", label="Top 1 (dominant)"),
        Patch(facecolor="#EF9F27", label="Top 2–3"),
        Patch(facecolor="#378ADD", label="Top 4–10"),
    ], fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()
    return _save(fig, plots_dir/"top_feature_dominance.png", "top_feature_dominance.png")


# ── 2. Dataset Separability Chart ─────────────────────────────────────────────

def plot_dataset_separability(
    X_test_raw   : pd.DataFrame,
    y_true       : np.ndarray,
    feature_names: list[str],
    shap_values  : np.ndarray,
    plots_dir    : Path,
    top_n        : int = 5,
) -> Path:
    """
    For the top-N SHAP features: show class-separated KDE distributions.
    Clean separation → explains high performance.
    """
    _setup()
    mean_abs  = np.abs(shap_values).mean(axis=0)
    order     = np.argsort(mean_abs)[::-1][:top_n]
    top_feats = [feature_names[i] for i in order]

    fig, axes = plt.subplots(1, top_n, figsize=(top_n * 4.5, 4))
    if top_n == 1: axes = [axes]

    for ax, feat in zip(axes, top_feats):
        if feat not in X_test_raw.columns:
            ax.set_visible(False)
            continue
        ph_vals = X_test_raw.loc[y_true == 0, feat].dropna()
        lg_vals = X_test_raw.loc[y_true == 1, feat].dropna()

        try:
            ph_vals.plot.kde(ax=ax, color="#E24B4A",
                              label=f"Phishing (n={len(ph_vals):,})",
                              linewidth=2)
            lg_vals.plot.kde(ax=ax, color="#1D9E75",
                              label=f"Legit (n={len(lg_vals):,})",
                              linewidth=2)
        except Exception:
            ax.hist(ph_vals, bins=40, density=True, alpha=0.6,
                    color="#E24B4A", label="Phishing")
            ax.hist(lg_vals, bins=40, density=True, alpha=0.6,
                    color="#1D9E75", label="Legitimate")

        # Overlap metric (simple)
        ax.set_title(feat, fontsize=9, fontweight="600")
        ax.set_xlabel("Feature value")
        ax.legend(fontsize=7)
        sns.despine(ax=ax)

    fig.suptitle(
        "Dataset Separability — Class Distributions of Top Features\n"
        "(Non-overlapping distributions → strong separation → near-perfect accuracy)",
        fontsize=12, fontweight="700",
    )
    plt.tight_layout()
    return _save(fig, plots_dir/"dataset_separability.png", "dataset_separability.png")


# ── 3. Feature Dominance vs Remaining ────────────────────────────────────────

def plot_dominance_vs_remaining(
    shap_values  : np.ndarray,
    feature_names: list[str],
    plots_dir    : Path,
) -> Path:
    """Stacked bar comparing importance share of top-1/3/5/10 vs rest."""
    _setup()
    mean_abs = np.abs(shap_values).mean(axis=0)
    total    = mean_abs.sum()
    order    = np.argsort(mean_abs)[::-1]

    buckets  = [1, 3, 5, 10, 20]
    labels   = [f"Top {b}" for b in buckets]
    top_pcts = []
    for b in buckets:
        pct = mean_abs[order[:b]].sum() / max(total, 1e-12) * 100
        top_pcts.append(pct)
    rest_pcts= [100 - p for p in top_pcts]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(buckets))
    w = 0.55

    b1 = ax.bar(x, top_pcts,  width=w, label="Top-N features",
                color="#185FA5", edgecolor="white")
    b2 = ax.bar(x, rest_pcts, width=w, bottom=top_pcts,
                label="Remaining features", color="#B5D4F4", edgecolor="white")

    for bar, pct in zip(b1, top_pcts):
        ax.text(bar.get_x() + bar.get_width()/2,
                pct/2, f"{pct:.1f}%",
                ha="center", va="center", fontsize=10,
                fontweight="700", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of total SHAP importance")
    ax.set_ylim(0, 115)
    ax.set_title(
        "Feature Dominance vs Remaining Features\n"
        "(How concentrated is the model's decision-making?)",
        fontsize=12, fontweight="700",
    )
    ax.legend(fontsize=10)
    sns.despine(ax=ax)
    plt.tight_layout()
    return _save(fig, plots_dir/"dominance_vs_remaining.png", "dominance_vs_remaining.png")


# ── 4. Cumulative SHAP Importance Curve ──────────────────────────────────────

def plot_cumulative_shap_curve(
    shap_values  : np.ndarray,
    feature_names: list[str],
    plots_dir    : Path,
) -> Path:
    """
    Line chart: cumulative % of total SHAP importance as features are added
    in order of decreasing importance.
    """
    _setup()
    n_feats  = len(feature_names)
    mean_abs = np.abs(shap_values).mean(axis=0)
    total    = mean_abs.sum()
    order    = np.argsort(mean_abs)[::-1]
    cumsum   = np.cumsum(mean_abs[order]) / max(total, 1e-12) * 100
    x        = np.arange(1, n_feats + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, cumsum, color="#185FA5", linewidth=2.5)
    ax.fill_between(x, cumsum, alpha=0.12, color="#185FA5")

    # Mark key milestones
    milestones = [1, 3, 5, 10, 20]
    for m in milestones:
        if m <= n_feats:
            val = cumsum[m-1]
            ax.axvline(m, color="#E24B4A", linewidth=1, linestyle="--", alpha=0.6)
            ax.annotate(f"Top {m}\n{val:.1f}%",
                        xy=(m, val), xytext=(m+0.8, val-8),
                        fontsize=8, fontweight="600", color="#E24B4A",
                        arrowprops=dict(arrowstyle="->", color="#E24B4A",
                                        lw=1.2))

    ax.axhline(80, color="#0F6E56", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(n_feats*0.85, 81, "80% threshold", fontsize=8, color="#0F6E56")
    ax.axhline(95, color="#854F0B", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(n_feats*0.85, 96, "95% threshold", fontsize=8, color="#854F0B")

    ax.set_xlabel("Number of features (sorted by importance)")
    ax.set_ylabel("Cumulative % of total SHAP importance")
    ax.set_xlim(1, n_feats)
    ax.set_ylim(0, 105)
    ax.set_title(
        "Cumulative SHAP Importance Curve\n"
        "(Steep early rise → model relies on very few features → explains near-perfect AUC)",
        fontsize=12, fontweight="700",
    )
    sns.despine(ax=ax)
    plt.tight_layout()
    return _save(fig, plots_dir/"cumulative_shap_curve.png", "cumulative_shap_curve.png")


# ── 5. Feature Contribution Distribution ─────────────────────────────────────

def plot_contribution_distribution(
    shap_values  : np.ndarray,
    feature_names: list[str],
    plots_dir    : Path,
) -> Path:
    """
    Histogram of mean |SHAP| values across all features.
    A heavy right tail shows a small number of features dominate.
    """
    _setup()
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]
    top3     = [feature_names[i] for i in order[:3]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: histogram
    ax = axes[0]
    ax.hist(mean_abs, bins=20, color="#185FA5", edgecolor="white",
            linewidth=0.7, alpha=0.85)
    for topf in top3:
        idx = feature_names.index(topf)
        ax.axvline(mean_abs[idx], color="#E24B4A", linewidth=1.8, linestyle="--")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Number of features")
    ax.set_title("Distribution of Feature Importance\n(|SHAP| concentration)")
    sns.despine(ax=ax)

    # Right: log-scale sorted bar
    ax = axes[1]
    sorted_vals = np.sort(mean_abs)[::-1]
    ax.bar(range(len(sorted_vals)), sorted_vals,
           color=["#E24B4A" if i < 3 else "#378ADD" for i in range(len(sorted_vals))],
           edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("Feature rank (1 = most important)")
    ax.set_ylabel("Mean |SHAP| (log scale)")
    ax.set_title("Importance Decay by Rank\n(steep drop → extreme concentration)")
    sns.despine(ax=ax)

    fig.suptitle(
        "Feature Contribution Distribution — Explaining Near-Perfect Performance\n"
        f"Top-3 features: {', '.join(top3)}",
        fontsize=12, fontweight="700",
    )
    plt.tight_layout()
    return _save(fig, plots_dir/"contribution_distribution.png", "contribution_distribution.png")


# ── 6. URLSimilarityIndex Impact ─────────────────────────────────────────────

def plot_usi_impact(
    X_test_raw: pd.DataFrame,
    y_true    : np.ndarray,
    plots_dir : Path,
) -> Path:
    """
    4-panel analysis of URLSimilarityIndex leakage and its impact.
    Shows per-class distribution and how it would classify samples alone.
    """
    _setup()
    col = "URLSimilarityIndex"
    if col not in X_test_raw.columns:
        logger.warning("URLSimilarityIndex not in raw test data — skipping plot 6")
        return plots_dir / "usi_impact.png"

    ph_vals = X_test_raw.loc[y_true == 0, col].dropna()
    lg_vals = X_test_raw.loc[y_true == 1, col].dropna()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "URLSimilarityIndex Impact Analysis (Track A only — excluded from Track B)\n"
        "ALL legitimate sites have URLSimilarityIndex = 100.0",
        fontsize=12, fontweight="700",
    )

    # Panel 1: Histogram
    ax = axes[0]
    ax.hist(ph_vals, bins=60, alpha=0.7, color="#E24B4A",
            label=f"Phishing (n={len(ph_vals):,})", density=True)
    ax.axvline(100, color="#1D9E75", linewidth=3,
               label=f"Legitimate (ALL = 100.0, n={len(lg_vals):,})")
    ax.set_xlabel("URLSimilarityIndex")
    ax.set_ylabel("Density")
    ax.set_title("Distribution by Class")
    ax.legend(fontsize=8)
    sns.despine(ax=ax)

    # Panel 2: Box plot
    ax = axes[1]
    bp = ax.boxplot([ph_vals, lg_vals], tick_labels=["Phishing (0)","Legitimate (1)"],
                    patch_artist=True, notch=False,
                    medianprops={"linewidth":2.5,"color":"#333"},
                    flierprops={"marker":".","markersize":1.5,"alpha":0.2})
    for patch, color in zip(bp["boxes"], ["#E24B4A","#1D9E75"]):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_title("Box Plot by Class")
    ax.set_ylabel("URLSimilarityIndex")
    sns.despine(ax=ax)

    # Panel 3: Solo classification accuracy
    ax = axes[2]
    thresholds = np.linspace(0, 100, 100)
    accs = []
    for t in thresholds:
        pred_legit = (X_test_raw[col].fillna(0) >= t).astype(int)
        acc = (pred_legit.values == y_true).mean()
        accs.append(max(acc, 1-acc))
    ax.plot(thresholds, accs, color="#185FA5", linewidth=2)
    ax.axhline(max(accs), color="#E24B4A", linestyle="--",
               label=f"Max accuracy: {max(accs):.4f}")
    ax.set_xlabel("Threshold for predicting Legitimate")
    ax.set_ylabel("Accuracy (solo predictor)")
    ax.set_title("URLSimilarityIndex as Solo Predictor")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

    plt.tight_layout()
    return _save(fig, plots_dir/"usi_impact.png", "usi_impact.png")


# ── 7. HTTPS Impact Visualization ────────────────────────────────────────────

def plot_https_impact(
    X_test_raw: pd.DataFrame,
    y_true    : np.ndarray,
    y_proba   : np.ndarray,
    plots_dir : Path,
) -> Path:
    """
    3-panel HTTPS analysis: class distribution, model score split, FPR/FNR.
    ALL legitimate sites have IsHTTPS=1 → explains strong model performance.
    """
    _setup()
    col = "IsHTTPS"
    if col not in X_test_raw.columns:
        logger.warning("IsHTTPS not in raw test data — skipping plot 7")
        return plots_dir / "https_impact.png"

    https_vals = X_test_raw[col].fillna(0).astype(int).values

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        "IsHTTPS Impact Analysis — All Legitimate Sites Use HTTPS\n"
        "(Advisory leakage: IsHTTPS=1 → high P(Legitimate) for model)",
        fontsize=12, fontweight="700",
    )

    # Panel 1: HTTPS rate per class
    ax = axes[0]
    ph_https  = https_vals[y_true==0].mean()*100
    lg_https  = https_vals[y_true==1].mean()*100
    bars = ax.bar(["Phishing (0)","Legitimate (1)"], [ph_https, lg_https],
                  color=["#E24B4A","#1D9E75"], edgecolor="white", width=0.4)
    for bar, v in zip(bars, [ph_https, lg_https]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{v:.1f}%", ha="center", fontsize=11, fontweight="600")
    ax.set_ylabel("% using HTTPS"); ax.set_ylim(0, 115)
    ax.set_title("HTTPS Adoption Rate by Class"); sns.despine(ax=ax)

    # Panel 2: Model probability score split by HTTPS
    ax = axes[1]
    p_http  = y_proba[https_vals==0]
    p_https = y_proba[https_vals==1]
    ax.hist(p_http,  bins=50, alpha=0.65, color="#E24B4A",
            label=f"Non-HTTPS (n={len(p_http):,})", density=True)
    ax.hist(p_https, bins=50, alpha=0.65, color="#1D9E75",
            label=f"HTTPS (n={len(p_https):,})", density=True)
    ax.set_xlabel("P(Legitimate)"); ax.set_ylabel("Density")
    ax.set_title("Model Probability by HTTPS Status")
    ax.legend(fontsize=8); sns.despine(ax=ax)

    # Panel 3: FPR and FNR by HTTPS group
    ax = axes[2]
    groups, fpr_vals, fnr_vals = [], [], []
    for grp_name, https_flag in [("Non-HTTPS",0),("HTTPS",1)]:
        mask  = https_vals==https_flag
        if mask.sum()<10: continue
        yt,yp = y_true[mask], (y_proba[mask]>=0.5).astype(int)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(yt, yp, labels=[0,1])
        tn,fp,fn,tp = cm.ravel() if cm.shape==(2,2) else (0,0,0,0)
        groups.append(grp_name)
        fpr_vals.append(fp/max(fp+tn,1))
        fnr_vals.append(fn/max(fn+tp,1))
    x = np.arange(len(groups)); w=0.35
    ax.bar(x-w/2, fpr_vals, width=w, label="FPR", color="#EF9F27", edgecolor="white")
    ax.bar(x+w/2, fnr_vals, width=w, label="FNR", color="#E24B4A", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Rate"); ax.set_title("FPR and FNR by HTTPS Group")
    ax.legend(fontsize=9); sns.despine(ax=ax)

    plt.tight_layout()
    return _save(fig, plots_dir/"https_impact.png", "https_impact.png")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def investigate_near_perfect_performance(
    shap_values  : np.ndarray,
    feature_names: list[str],
    X_test_raw   : pd.DataFrame,
    y_true       : np.ndarray,
    y_pred       : np.ndarray,
    y_proba      : np.ndarray,
    plots_dir    : str | Path = "outputs/plots/bias/performance_investigation",
) -> dict:
    """
    Run all 7 performance investigation visualisations and return summary stats.

    Returns
    -------
    dict  keys: top1_pct, top3_pct, top5_pct, top10_pct,
               n_features_80pct, n_features_95pct,
               plot_paths (dict of 7 Paths),
               dominance_summary (dict)
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 55)
    logger.info("M9 — NEAR-PERFECT PERFORMANCE INVESTIGATION")
    logger.info("=" * 55)

    mean_abs = np.abs(shap_values).mean(axis=0)
    total    = mean_abs.sum()
    order    = np.argsort(mean_abs)[::-1]
    cumsum   = np.cumsum(mean_abs[order]) / max(total, 1e-12) * 100

    top1_pct  = float(cumsum[0])
    top3_pct  = float(cumsum[2])  if len(cumsum) > 2  else top1_pct
    top5_pct  = float(cumsum[4])  if len(cumsum) > 4  else top3_pct
    top10_pct = float(cumsum[9])  if len(cumsum) > 9  else top5_pct

    n80  = int(np.searchsorted(cumsum, 80) + 1)
    n95  = int(np.searchsorted(cumsum, 95) + 1)

    logger.info(f"  Top-1  feature : {top1_pct:.1f}% of importance")
    logger.info(f"  Top-3  features: {top3_pct:.1f}%")
    logger.info(f"  Top-5  features: {top5_pct:.1f}%")
    logger.info(f"  Top-10 features: {top10_pct:.1f}%")
    logger.info(f"  Features for 80% importance: {n80}")
    logger.info(f"  Features for 95% importance: {n95}")

    # Generate all 7 plots
    paths = {}
    paths["1_dominance"]    = plot_top_feature_dominance(shap_values, feature_names, plots_dir)
    paths["2_separability"] = plot_dataset_separability(X_test_raw, y_true,
                                                         feature_names, shap_values, plots_dir)
    paths["3_dom_vs_rest"]  = plot_dominance_vs_remaining(shap_values, feature_names, plots_dir)
    paths["4_cumulative"]   = plot_cumulative_shap_curve(shap_values, feature_names, plots_dir)
    paths["5_distribution"] = plot_contribution_distribution(shap_values, feature_names, plots_dir)
    paths["6_usi_impact"]   = plot_usi_impact(X_test_raw, y_true, plots_dir)
    paths["7_https_impact"] = plot_https_impact(X_test_raw, y_true, y_proba, plots_dir)

    logger.info(f"Performance investigation plots saved: {len(paths)}")

    return {
        "top1_pct"          : round(top1_pct,  2),
        "top3_pct"          : round(top3_pct,  2),
        "top5_pct"          : round(top5_pct,  2),
        "top10_pct"         : round(top10_pct, 2),
        "n_features_80pct"  : n80,
        "n_features_95pct"  : n95,
        "top_feature"       : feature_names[order[0]],
        "top3_features"     : [feature_names[order[i]] for i in range(min(3,len(order)))],
        "plot_paths"        : paths,
    }
