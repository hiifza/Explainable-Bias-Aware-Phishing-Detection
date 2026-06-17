"""
src/eda/leakage_analyzer.py
----------------------------
Quantifies and visualises the two leakage-flagged features for M2.1:
  - URLSimilarityIndex  (critical — ALL legitimate = 100.0)
  - IsHTTPS             (advisory — ALL legitimate = IsHTTPS=1)

For each feature the module computes:
  - Class-conditional statistics (mean, std, min, max per label)
  - AUROC as a solo predictor (separation power)
  - Chi-square or ANOVA p-value
  - Evidence visualisations

Public API
----------
    analyze_urlsimilarity(df)              -> dict
    analyze_https(df)                      -> dict
    compute_separation_power(df, feature)  -> float   (AUROC)
    run_leakage_analysis(df, out_dir, plots_dir) -> dict
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
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.eda.visualization_manager import save_figure, setup_plot_style

logger = get_logger(__name__)

TARGET           = "label"
PHISHING_COLOR   = "#E24B4A"
LEGITIMATE_COLOR = "#1D9E75"
LEAKAGE_COLOR    = "#A32D2D"


# ── Separation power ──────────────────────────────────────────────────────────

def compute_separation_power(
    df     : pd.DataFrame,
    feature: str,
) -> float:
    """
    Compute the AUROC of *feature* as a solo predictor of *label*.

    AUROC = 1.0 means perfect separation; 0.5 means no better than chance.

    Parameters
    ----------
    df      : DataFrame containing feature and TARGET
    feature : column name

    Returns
    -------
    float — AUROC score
    """
    from sklearn.metrics import roc_auc_score

    if feature not in df.columns:
        return float("nan")

    scores = df[feature].fillna(df[feature].median()).values
    labels = df[TARGET].values

    try:
        auroc = roc_auc_score(labels, scores)
        # Ensure it's > 0.5 by convention (flip if needed)
        auroc = max(auroc, 1 - auroc)
    except Exception as exc:
        logger.warning(f"AUROC computation failed for {feature}: {exc}")
        auroc = float("nan")

    return float(auroc)


# ── URLSimilarityIndex ────────────────────────────────────────────────────────

def analyze_urlsimilarity(df: pd.DataFrame) -> dict:
    """
    Full statistical analysis of URLSimilarityIndex leakage.

    Returns
    -------
    dict with keys:
        all_legit_100, phishing_min/mean/max/std,
        legit_min/mean/max/std, auroc, anova_f, anova_p,
        n_phishing, n_legitimate
    """
    feat = "URLSimilarityIndex"
    if feat not in df.columns:
        logger.warning(f"'{feat}' not in DataFrame — skipping")
        return {}

    ph_vals  = df[df[TARGET] == 0][feat].dropna()
    lg_vals  = df[df[TARGET] == 1][feat].dropna()

    all_legit_100 = bool((lg_vals == 100).all())
    any_legit_ne  = bool((lg_vals != 100).any())

    # ANOVA
    f_stat, p_val = scipy_stats.f_oneway(ph_vals, lg_vals)

    # AUROC
    auroc = compute_separation_power(df, feat)

    result = {
        "feature"        : feat,
        "all_legit_100"  : all_legit_100,
        "any_legit_ne100": any_legit_ne,
        "phishing_min"   : float(ph_vals.min()),
        "phishing_mean"  : round(float(ph_vals.mean()), 4),
        "phishing_max"   : float(ph_vals.max()),
        "phishing_std"   : round(float(ph_vals.std()), 4),
        "legit_min"      : float(lg_vals.min()),
        "legit_mean"     : round(float(lg_vals.mean()), 4),
        "legit_max"      : float(lg_vals.max()),
        "legit_std"      : round(float(lg_vals.std()), 6),
        "auroc"          : round(auroc, 6),
        "anova_f"        : round(float(f_stat), 2),
        "anova_p"        : float(p_val),
        "n_phishing"     : int(len(ph_vals)),
        "n_legitimate"   : int(len(lg_vals)),
    }

    logger.info(f"URLSimilarityIndex AUROC          : {auroc:.6f}")
    logger.info(f"  All legitimate = 100.0          : {all_legit_100}")
    logger.info(f"  Phishing mean                   : {result['phishing_mean']}")
    logger.info(f"  Legitimate mean                 : {result['legit_mean']}")
    return result


# ── IsHTTPS ───────────────────────────────────────────────────────────────────

def analyze_https(df: pd.DataFrame) -> dict:
    """
    Full statistical analysis of IsHTTPS advisory leakage.

    Returns
    -------
    dict with keys:
        all_legit_https1, phishing_https_pct, legit_https_pct,
        auroc, chi2, chi2_p, contingency_table
    """
    feat = "IsHTTPS"
    if feat not in df.columns:
        logger.warning(f"'{feat}' not in DataFrame — skipping")
        return {}

    ph_vals = df[df[TARGET] == 0][feat].dropna()
    lg_vals = df[df[TARGET] == 1][feat].dropna()

    all_legit_1 = bool((lg_vals == 1).all())

    ph_https_pct = float((ph_vals == 1).mean() * 100)
    lg_https_pct = float((lg_vals == 1).mean() * 100)

    # Chi-square test
    ct = pd.crosstab(df[TARGET], df[feat])
    chi2_stat, chi2_p, _, _ = scipy_stats.chi2_contingency(ct)

    # AUROC
    auroc = compute_separation_power(df, feat)

    result = {
        "feature"           : feat,
        "all_legit_https1"  : all_legit_1,
        "phishing_https_pct": round(ph_https_pct, 2),
        "legit_https_pct"   : round(lg_https_pct, 2),
        "auroc"             : round(auroc, 6),
        "chi2"              : round(float(chi2_stat), 2),
        "chi2_p"            : float(chi2_p),
        "n_phishing"        : int(len(ph_vals)),
        "n_legitimate"      : int(len(lg_vals)),
        "contingency_table" : ct.to_dict(),
    }

    logger.info(f"IsHTTPS AUROC                     : {auroc:.6f}")
    logger.info(f"  All legitimate have HTTPS=1     : {all_legit_1}")
    logger.info(f"  Phishing sites using HTTPS      : {ph_https_pct:.1f}%")
    logger.info(f"  Legitimate sites using HTTPS    : {lg_https_pct:.1f}%")
    return result


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_urlsimilarity_evidence(
    df       : pd.DataFrame,
    usi_stats: dict,
    plots_dir: Path,
) -> None:
    """
    4-panel leakage evidence figure for URLSimilarityIndex.

    Panels
    ------
    [0] Histogram by class
    [1] Boxplot by class
    [2] Cumulative distribution by class
    [3] KDE — phishing only (legitimate is a spike at 100)
    """
    setup_plot_style()
    feat    = "URLSimilarityIndex"
    ph_data = df[df[TARGET] == 0][feat].dropna()
    lg_data = df[df[TARGET] == 1][feat].dropna()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"URLSimilarityIndex — Leakage Evidence\n"
        f"(AUROC={usi_stats.get('auroc',0):.4f}  |  All Legitimate = 100.0: "
        f"{usi_stats.get('all_legit_100',False)})",
        fontsize=12, fontweight="700",
    )

    # Panel 0: Histogram by class
    ax = axes[0, 0]
    ax.hist(ph_data, bins=80, alpha=0.75, color=PHISHING_COLOR,
            label=f"Phishing (0)  n={len(ph_data):,}", edgecolor="white", linewidth=0.3)
    ax.axvline(100, color=LEGITIMATE_COLOR, linewidth=2.5,
               label=f"Legitimate (1) — all = 100.0  n={len(lg_data):,}")
    ax.set_xlabel("URLSimilarityIndex")
    ax.set_ylabel("Count")
    ax.set_title("Distribution by Class")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    sns.despine(ax=ax)

    # Panel 1: Boxplot by class
    ax = axes[0, 1]
    bp = ax.boxplot([ph_data, lg_data],
                    vert=True, tick_labels=["Phishing (0)", "Legitimate (1)"],
                    patch_artist=True, notch=False,
                    medianprops={"linewidth": 2.5, "color": "#333"},
                    flierprops={"marker": ".", "markersize": 1.2, "alpha": 0.2})
    for patch, color in zip(bp["boxes"], [PHISHING_COLOR, LEGITIMATE_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel("URLSimilarityIndex")
    ax.set_title("Boxplot by Class")
    sns.despine(ax=ax)

    # Panel 2: Empirical CDF by class
    ax = axes[1, 0]
    for data, color, label in [
        (ph_data, PHISHING_COLOR,   "Phishing (0)"),
        (lg_data, LEGITIMATE_COLOR, "Legitimate (1)"),
    ]:
        sorted_d = np.sort(data)
        cdf      = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax.plot(sorted_d, cdf, color=color, linewidth=2, label=label)
    ax.set_xlabel("URLSimilarityIndex")
    ax.set_ylabel("Cumulative proportion")
    ax.set_title("Empirical CDF by Class")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

    # Panel 3: Phishing KDE (legitimate is degenerate at 100)
    ax = axes[1, 1]
    try:
        ph_data.plot.kde(ax=ax, color=PHISHING_COLOR, linewidth=2, label="Phishing KDE")
    except Exception:
        ax.hist(ph_data, bins=60, density=True, color=PHISHING_COLOR,
                alpha=0.7, edgecolor="white", label="Phishing")
    ax.axvline(100, color=LEAKAGE_COLOR, linewidth=3, linestyle="--",
               label="Legitimate spike at 100.0")
    ax.set_xlabel("URLSimilarityIndex")
    ax.set_ylabel("Density")
    ax.set_title("KDE (Phishing) + Legitimate spike at 100")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

    plt.tight_layout()
    save_figure(fig, plots_dir / "leakage_urlsimilarity.png")


def plot_https_evidence(
    df          : pd.DataFrame,
    https_stats : dict,
    plots_dir   : Path,
) -> None:
    """
    3-panel evidence figure for IsHTTPS advisory leakage.

    Panels
    ------
    [0] Grouped bar: HTTPS rate by class
    [1] Confusion-style 2×2 count table
    [2] Pie charts: HTTPS split within each class
    """
    setup_plot_style()
    feat    = "IsHTTPS"
    ph_data = df[df[TARGET] == 0][feat].dropna()
    lg_data = df[df[TARGET] == 1][feat].dropna()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"IsHTTPS — Advisory Leakage Evidence\n"
        f"(AUROC={https_stats.get('auroc',0):.4f}  |  "
        f"Phishing HTTPS: {https_stats.get('phishing_https_pct',0):.1f}%  |  "
        f"Legitimate HTTPS: {https_stats.get('legit_https_pct',0):.1f}%)",
        fontsize=12, fontweight="700",
    )

    # Panel 0: HTTPS rate by class
    ax = axes[0]
    ph_pct = https_stats.get("phishing_https_pct", 0)
    lg_pct = https_stats.get("legit_https_pct", 0)
    bars   = ax.bar(["Phishing (0)", "Legitimate (1)"],
                    [ph_pct, lg_pct],
                    color=[PHISHING_COLOR, LEGITIMATE_COLOR],
                    edgecolor="white", width=0.45)
    for bar, v in zip(bars, [ph_pct, lg_pct]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8, f"{v:.1f}%",
                ha="center", fontsize=11, fontweight="500")
    ax.set_ylabel("% using HTTPS")
    ax.set_ylim(0, 115)
    ax.set_title("HTTPS Adoption Rate by Class")
    sns.despine(ax=ax)

    # Panel 1: Stacked bar showing HTTP vs HTTPS within each class
    ax = axes[1]
    ph_http  = 100 - ph_pct
    lg_http  = 100 - lg_pct
    x        = np.array([0, 1])
    w        = 0.4
    ax.bar(x, [ph_http,  lg_http],  width=w, label="HTTP (0)",  color="#B0BEC5", edgecolor="white")
    ax.bar(x, [ph_pct,   lg_pct],   width=w, label="HTTPS (1)", color="#378ADD", edgecolor="white",
           bottom=[ph_http, lg_http])
    ax.set_xticks(x)
    ax.set_xticklabels(["Phishing (0)", "Legitimate (1)"])
    ax.set_ylabel("Percentage (%)")
    ax.set_title("HTTP vs HTTPS Composition")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

    # Panel 2: Two pies — IsHTTPS within each class
    ax = axes[2]
    ax.axis("off")
    # Draw two mini pies using inset axes
    ax_ph = fig.add_axes([0.68, 0.20, 0.14, 0.55])
    ax_lg = fig.add_axes([0.84, 0.20, 0.14, 0.55])
    for mini_ax, data, title, color in [
        (ax_ph, ph_data, "Phishing", PHISHING_COLOR),
        (ax_lg, lg_data, "Legitimate", LEGITIMATE_COLOR),
    ]:
        vc  = data.value_counts().sort_index()
        v0  = int(vc.get(0, 0))
        v1  = int(vc.get(1, 0))
        mini_ax.pie([v0, v1], colors=["#B0BEC5", color],
                    startangle=90, autopct="%1.0f%%",
                    wedgeprops={"edgecolor": "white", "linewidth": 1.2},
                    textprops={"fontsize": 8})
        mini_ax.set_title(title, fontsize=9, fontweight="600")

    plt.tight_layout()
    save_figure(fig, plots_dir / "leakage_https.png")


def plot_leakage_comparison(
    usi_stats  : dict,
    https_stats: dict,
    plots_dir  : Path,
) -> None:
    """Side-by-side AUROC comparison bar for both leakage features."""
    setup_plot_style()

    features = ["URLSimilarityIndex\n(critical)", "IsHTTPS\n(advisory)"]
    aurocs   = [usi_stats.get("auroc", 0), https_stats.get("auroc", 0)]
    colors   = [LEAKAGE_COLOR, "#EF9F27"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(features, aurocs, color=colors, edgecolor="white", width=0.4)
    ax.axhline(0.5, color="#888", linestyle="--", linewidth=1, label="Random baseline (0.5)")
    ax.axhline(1.0, color="#185FA5", linestyle=":",  linewidth=1, label="Perfect (1.0)")
    for bar, v in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005, f"{v:.4f}",
                ha="center", fontsize=11, fontweight="600")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("AUROC (solo predictor)")
    ax.set_title("Leakage Feature Separation Power\n(AUROC as solo predictor)",
                 fontsize=12, fontweight="600")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()
    save_figure(fig, plots_dir / "leakage_auroc_comparison.png")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_leakage_analysis(
    df        : pd.DataFrame,
    output_dir: str | Path = "outputs/reports",
    plots_dir : str | Path = "outputs/plots/eda",
) -> dict:
    """
    Execute the full leakage analysis pipeline.

    Parameters
    ----------
    df         : clean DataFrame from M1.1
    output_dir : CSV output directory
    plots_dir  : plot output directory

    Returns
    -------
    dict  keys: urlsimilarity (dict), https (dict)
    """
    output_dir = Path(output_dir)
    plots_dir  = Path(plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("M2.1 — LEAKAGE ANALYSIS MODULE")
    logger.info("=" * 60)

    # 1. URLSimilarityIndex
    usi_stats = analyze_urlsimilarity(df)
    plot_urlsimilarity_evidence(df, usi_stats, plots_dir)

    # 2. IsHTTPS
    https_stats = analyze_https(df)
    plot_https_evidence(df, https_stats, plots_dir)

    # 3. Comparison bar
    plot_leakage_comparison(usi_stats, https_stats, plots_dir)

    # 4. Save CSV report
    rows = []
    for stats in [usi_stats, https_stats]:
        for k, v in stats.items():
            if isinstance(v, dict):
                continue      # skip nested dicts (contingency table)
            rows.append({"feature": stats.get("feature",""), "metric": k, "value": v})
    pd.DataFrame(rows).to_csv(output_dir / "leakage_analysis.csv", index=False)
    logger.info("Saved: leakage_analysis.csv")

    # 5. Validation assertions
    assert usi_stats.get("all_legit_100", False), \
        "URLSimilarityIndex leakage NOT confirmed — recheck dataset"
    assert usi_stats.get("auroc", 0) > 0.95, \
        f"URLSimilarityIndex AUROC unexpectedly low: {usi_stats.get('auroc')}"

    logger.info("LEAKAGE ANALYSIS COMPLETE")

    return {
        "urlsimilarity": usi_stats,
        "https"        : https_stats,
    }
