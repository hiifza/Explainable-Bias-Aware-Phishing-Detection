"""
src/eda/eda_analyzer.py
-----------------------
Dataset overview computation, TLD analysis, and statistical
feature-importance pre-screening for Module M2.1.

Public API
----------
    compute_dataset_overview(df, feature_lists)  -> dict
    compute_tld_analysis(df)                     -> dict
    compute_feature_prescreening(df, track_B)    -> dict
    run_eda_overview(df, feature_lists, out_dir, plots_dir) -> dict
"""

import sys
from pathlib import Path
from typing import Optional

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
from src.eda.visualization_manager import save_figure, setup_plot_style

logger = get_logger(__name__)

TARGET = "label"
TLD_COL = "TLD"


# ── Dataset overview ──────────────────────────────────────────────────────────

def compute_dataset_overview(
    df           : pd.DataFrame,
    track_B_feats: list[str],
) -> dict:
    """
    Compute comprehensive dataset statistics.

    Parameters
    ----------
    df             : full clean DataFrame (from M1.1)
    track_B_feats  : list of Track B feature names

    Returns
    -------
    dict with keys: n_rows, n_cols, n_features, n_numeric, n_categorical,
                    n_binary, total_missing, memory_mb, class_distribution,
                    dtype_counts
    """
    logger.info("Computing dataset overview …")

    feat_df  = df[track_B_feats]
    n_rows   = len(df)
    n_cols   = df.shape[1]

    # Feature type counts
    n_numeric     = int(feat_df.select_dtypes(include=[np.number]).shape[1])
    n_categorical = int(feat_df.select_dtypes(include=["object"]).shape[1])
    binary_cols   = [c for c in feat_df.columns
                     if feat_df[c].nunique() == 2 and c != TARGET]
    n_binary      = len(binary_cols)

    # Missing values
    total_missing = int(df.isnull().sum().sum())

    # Memory
    memory_mb = df.memory_usage(deep=True).sum() / 1_000_000

    # Class distribution
    vc = df[TARGET].value_counts().sort_index()
    ph = int(vc.get(0, 0))
    lg = int(vc.get(1, 0))
    class_dist = {
        "phishing_count"   : ph,
        "legitimate_count" : lg,
        "phishing_pct"     : round(ph / n_rows * 100, 4),
        "legitimate_pct"   : round(lg / n_rows * 100, 4),
        "imbalance_ratio"  : round(ph / max(lg, 1), 6),
    }

    # Dtype counts
    dtype_counts = {str(k): int(v)
                    for k, v in feat_df.dtypes.value_counts().items()}

    overview = {
        "n_rows"            : n_rows,
        "n_cols"            : n_cols,
        "n_features"        : len(track_B_feats),
        "n_numeric"         : n_numeric,
        "n_categorical"     : n_categorical,
        "n_binary"          : n_binary,
        "total_missing"     : total_missing,
        "memory_mb"         : round(memory_mb, 2),
        "class_distribution": class_dist,
        "dtype_counts"      : dtype_counts,
    }

    logger.info(f"  Rows        : {n_rows:,}")
    logger.info(f"  Track B feat: {len(track_B_feats)}")
    logger.info(f"  Missing     : {total_missing}")
    logger.info(f"  Memory      : {memory_mb:.1f} MB")
    logger.info(f"  Phishing    : {ph:,} ({class_dist['phishing_pct']:.2f}%)")
    logger.info(f"  Legitimate  : {lg:,} ({class_dist['legitimate_pct']:.2f}%)")

    return overview


def save_overview_csv(overview: dict, output_path: str | Path) -> None:
    """Flatten the overview dict and save to CSV."""
    cd  = overview.pop("class_distribution", {})
    dtc = overview.pop("dtype_counts", {})

    rows = [{"metric": k, "value": v} for k, v in overview.items()]
    rows += [{"metric": f"class_{k}", "value": v} for k, v in cd.items()]
    rows += [{"metric": f"dtype_{k}", "value": v} for k, v in dtc.items()]

    # Restore nested keys so caller still has them
    overview["class_distribution"] = cd
    overview["dtype_counts"]       = dtc

    pd.DataFrame(rows).to_csv(output_path, index=False)
    logger.info(f"Saved: {Path(output_path).name}")


def plot_class_distribution(
    df       : pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Bar chart of class distribution with counts and percentages."""
    setup_plot_style()
    vc     = df[TARGET].value_counts().sort_index()
    labels = ["Phishing (0)", "Legitimate (1)"]
    counts = [vc.get(0, 0), vc.get(1, 0)]
    colors = ["#E24B4A", "#1D9E75"]
    pcts   = [c / len(df) * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars    = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    for bar, cnt, pct in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{cnt:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="500")
    ax.set_ylabel("Sample count")
    ax.set_title("Class Distribution", fontsize=13, fontweight="700")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_ylim(0, max(counts) * 1.2)
    sns.despine(ax=ax)
    plt.tight_layout()
    save_figure(fig, plots_dir / "class_distribution.png")


def plot_dtype_summary(
    df           : pd.DataFrame,
    track_B_feats: list[str],
    plots_dir    : Path,
) -> None:
    """Pie chart of feature dtype distribution."""
    setup_plot_style()
    feat_df   = df[track_B_feats]
    dtype_vc  = feat_df.dtypes.value_counts()
    labels    = [str(d) for d in dtype_vc.index]
    sizes     = dtype_vc.values
    colors    = ["#185FA5", "#0F6E56", "#854F0B", "#533AB7"][:len(labels)]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(sizes, labels=[f"{l}\n({s})" for l, s in zip(labels, sizes)],
           colors=colors, autopct="%1.0f%%", startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.set_title("Feature Data Types (Track B)", fontsize=12, fontweight="600")
    plt.tight_layout()
    save_figure(fig, plots_dir / "dtype_summary.png")


# ── TLD analysis ──────────────────────────────────────────────────────────────

def compute_tld_analysis(df: pd.DataFrame) -> dict:
    """
    Compute TLD frequency, phishing rate, legitimate rate, and long-tail stats.

    Returns
    -------
    dict with keys: n_unique_tlds, top_tlds (DataFrame), long_tail_stats
    """
    logger.info("Computing TLD analysis …")

    if TLD_COL not in df.columns:
        logger.warning(f"Column '{TLD_COL}' not found — skipping TLD analysis")
        return {}

    tld_group = (
        df.groupby(TLD_COL)[TARGET]
        .agg(
            count       = "count",
            phishing_count = lambda x: (x == 0).sum(),
            legitimate_count = lambda x: (x == 1).sum(),
        )
        .reset_index()
    )
    tld_group["phishing_rate"]   = tld_group["phishing_count"]   / tld_group["count"]
    tld_group["legitimate_rate"] = tld_group["legitimate_count"] / tld_group["count"]
    tld_group = tld_group.sort_values("count", ascending=False).reset_index(drop=True)

    # Long-tail: TLDs with < 100 samples
    long_tail = tld_group[tld_group["count"] < 100]

    n_unique = df[TLD_COL].nunique()
    logger.info(f"  Unique TLDs    : {n_unique}")
    logger.info(f"  Long-tail TLDs : {len(long_tail)} (< 100 samples)")
    logger.info(f"  Top TLD        : {tld_group.iloc[0][TLD_COL]}  ({tld_group.iloc[0]['count']:,} samples)")

    return {
        "n_unique_tlds"  : n_unique,
        "top_tlds"       : tld_group,
        "long_tail_count": len(long_tail),
        "long_tail_stats": long_tail.describe().to_dict(),
    }


def save_tld_csv(tld_results: dict, output_path: str | Path) -> None:
    """Save TLD analysis table to CSV."""
    tld_df = tld_results.get("top_tlds", pd.DataFrame())
    if len(tld_df) == 0:
        return
    tld_df.to_csv(output_path, index=False)
    logger.info(f"Saved: {Path(output_path).name}")


def plot_tld_phishing_rate(
    tld_results: dict,
    plots_dir  : Path,
    top_n      : int = 30,
) -> None:
    """
    Horizontal bar chart: phishing rate for the top_n most frequent TLDs.
    TLDs are ordered by sample count so the most common appear at top.
    """
    setup_plot_style()
    tld_df = tld_results.get("top_tlds", pd.DataFrame())
    if len(tld_df) == 0:
        return

    top = tld_df.head(top_n).copy()
    top = top.sort_values("phishing_rate", ascending=True)

    colors = ["#E24B4A" if r >= 0.7 else "#EF9F27" if r >= 0.4 else "#1D9E75"
              for r in top["phishing_rate"]]

    fig, ax = plt.subplots(figsize=(9, max(6, top_n * 0.32)))
    bars = ax.barh(top[TLD_COL], top["phishing_rate"] * 100,
                   color=colors, edgecolor="white", linewidth=0.6)
    ax.axvline(50, color="#888", linestyle="--", linewidth=1, label="50% baseline")
    for bar, cnt in zip(bars, top["count"]):
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"n={cnt:,}", va="center", fontsize=7.5, color="#666")
    ax.set_xlabel("Phishing rate (%)")
    ax.set_title(f"Phishing Rate by TLD — Top {top_n} most frequent",
                 fontsize=12, fontweight="600")
    ax.set_xlim(0, 115)
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()
    save_figure(fig, plots_dir / "tld_phishing_rate.png")


def plot_tld_frequency(
    tld_results: dict,
    plots_dir  : Path,
    top_n      : int = 25,
) -> None:
    """Bar chart of the top_n TLDs by sample count, split by class."""
    setup_plot_style()
    tld_df = tld_results.get("top_tlds", pd.DataFrame())
    if len(tld_df) == 0:
        return

    top = tld_df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(top))
    w = 0.38
    ax.bar(x - w / 2, top["phishing_count"],   width=w, label="Phishing (0)",   color="#E24B4A", edgecolor="white")
    ax.bar(x + w / 2, top["legitimate_count"], width=w, label="Legitimate (1)", color="#1D9E75", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(top[TLD_COL], rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Sample count")
    ax.set_title(f"Top {top_n} TLDs — Sample Count by Class", fontsize=12, fontweight="600")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    sns.despine(ax=ax)
    plt.tight_layout()
    save_figure(fig, plots_dir / "tld_frequency.png")


# ── Feature pre-screening ─────────────────────────────────────────────────────

def compute_feature_prescreening(
    df           : pd.DataFrame,
    track_B_feats: list[str],
    sample_n     : int = 50_000,
) -> dict:
    """
    Compute Mutual Information, ANOVA F-score, and (where applicable)
    Chi-Square scores for all Track B features.

    Parameters
    ----------
    df             : full clean DataFrame
    track_B_feats  : Track B feature list (49 features)
    sample_n       : max rows for MI/ANOVA computation (speed)

    Returns
    -------
    dict with key 'mutual_information' (DataFrame with columns:
         feature, mi_score, anova_f, anova_p, mi_rank, anova_rank)
    """
    from sklearn.feature_selection import (
        mutual_info_classif,
        f_classif,
        chi2,
    )

    logger.info(f"Computing feature pre-screening (sample_n={sample_n:,}) …")

    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), size=min(sample_n, len(df)), replace=False)
    df_s = df.iloc[idx].reset_index(drop=True)

    # Only numeric Track B features (TLD is categorical — skip for MI/ANOVA)
    num_feats = [c for c in track_B_feats
                 if df[c].dtype in [np.int64, np.float64, "int64", "float64"]]
    cat_feats = [c for c in track_B_feats if c not in num_feats]

    X_num = df_s[num_feats].fillna(0).values
    y     = df_s[TARGET].values

    # ── Mutual Information
    mi_scores = mutual_info_classif(X_num, y, discrete_features="auto",
                                    random_state=42)

    # ── ANOVA F-score
    f_scores, p_values = f_classif(X_num, y)
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0)
    p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0)

    mi_series = pd.Series(mi_scores, index=num_feats).sort_values(ascending=False)
    f_series  = pd.Series(f_scores,  index=num_feats).sort_values(ascending=False)

    result_df = pd.DataFrame({
        "feature"    : num_feats,
        "mi_score"   : mi_scores,
        "anova_f"    : f_scores,
        "anova_p"    : p_values,
    })
    result_df["mi_rank"]    = result_df["mi_score"].rank(ascending=False).astype(int)
    result_df["anova_rank"] = result_df["anova_f"].rank(ascending=False).astype(int)
    result_df = result_df.sort_values("mi_score", ascending=False).reset_index(drop=True)

    # ── Chi-Square on binary features (non-negative)
    bin_feats  = [c for c in num_feats if df[c].nunique() <= 2]
    chi2_rows  = []
    if bin_feats:
        X_bin = df_s[bin_feats].fillna(0).clip(lower=0).values
        chi2_scores, chi2_p = chi2(X_bin, y)
        chi2_rows = [
            {"feature": f, "chi2_score": float(s), "chi2_p": float(p)}
            for f, s, p in zip(bin_feats, chi2_scores, chi2_p)
        ]
    chi2_df = pd.DataFrame(chi2_rows).sort_values("chi2_score", ascending=False) \
        if chi2_rows else pd.DataFrame()

    logger.info(f"  MI computed for {len(num_feats)} numeric features")
    logger.info(f"  Top-5 MI features: {list(result_df['feature'].head(5))}")
    logger.info(f"  Chi-Square binary features: {len(bin_feats)}")

    return {
        "mutual_information": result_df,
        "chi2"              : chi2_df,
        "categorical_feats" : cat_feats,
        "n_numeric"         : len(num_feats),
    }


def save_prescreening_csv(ps_results: dict, output_path: str | Path) -> None:
    """Save feature prescreening results to CSV."""
    mi_df = ps_results.get("mutual_information", pd.DataFrame())
    if len(mi_df) == 0:
        return
    mi_df.to_csv(output_path, index=False)
    logger.info(f"Saved: {Path(output_path).name}")


def plot_mutual_information(
    ps_results: dict,
    plots_dir : Path,
) -> None:
    """Horizontal bar chart of mutual information scores (Track B)."""
    setup_plot_style()
    mi_df = ps_results.get("mutual_information", pd.DataFrame())
    if len(mi_df) == 0:
        return

    mi_plot = mi_df.sort_values("mi_score", ascending=True)
    q75 = mi_plot["mi_score"].quantile(0.75)
    q50 = mi_plot["mi_score"].quantile(0.50)
    colors = [
        "#E24B4A" if v >= q75 else "#EF9F27" if v >= q50 else "#B0BEC5"
        for v in mi_plot["mi_score"]
    ]

    fig, ax = plt.subplots(figsize=(9, 13))
    ax.barh(mi_plot["feature"], mi_plot["mi_score"],
            color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mutual Information Score")
    ax.set_title("Feature Importance Pre-Screening — Mutual Information\n"
                 "(Track B numeric features, 50k sample)",
                 fontsize=12, fontweight="600")
    sns.despine(ax=ax)
    plt.tight_layout()
    save_figure(fig, plots_dir / "prescreening_mutual_info.png")


def plot_anova_prescreening(
    ps_results: dict,
    plots_dir : Path,
    top_n     : int = 25,
) -> None:
    """Bar chart of top ANOVA F-scores."""
    setup_plot_style()
    mi_df = ps_results.get("mutual_information", pd.DataFrame())
    if len(mi_df) == 0:
        return

    top = mi_df.nlargest(top_n, "anova_f").sort_values("anova_f", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["anova_f"], color="#533AB7",
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("ANOVA F-Score")
    ax.set_title(f"Top {top_n} Features — ANOVA F-Score",
                 fontsize=12, fontweight="600")
    sns.despine(ax=ax)
    plt.tight_layout()
    save_figure(fig, plots_dir / "prescreening_anova.png")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_eda_overview(
    df           : pd.DataFrame,
    track_B_feats: list[str],
    output_dir   : str | Path = "outputs/reports",
    plots_dir    : str | Path = "outputs/plots/eda",
) -> dict:
    """
    Run the complete dataset overview, TLD analysis, and prescreening.

    Parameters
    ----------
    df             : clean DataFrame from M1.1
    track_B_feats  : list of Track B feature names
    output_dir     : CSV output directory
    plots_dir      : plot output directory

    Returns
    -------
    dict with keys: overview, tld, prescreening
    """
    output_dir = Path(output_dir)
    plots_dir  = Path(plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 60
    logger.info(sep)
    logger.info("M2.1 — EDA OVERVIEW MODULE")
    logger.info(sep)

    # 1. Dataset overview
    overview = compute_dataset_overview(df, track_B_feats)
    save_overview_csv(overview, output_dir / "dataset_overview.csv")
    plot_class_distribution(df, plots_dir)
    plot_dtype_summary(df, track_B_feats, plots_dir)

    # 2. TLD analysis
    tld_results = compute_tld_analysis(df)
    save_tld_csv(tld_results, output_dir / "tld_analysis.csv")
    plot_tld_phishing_rate(tld_results, plots_dir)
    plot_tld_frequency(tld_results, plots_dir)

    # 3. Feature prescreening
    ps_results = compute_feature_prescreening(df, track_B_feats)
    save_prescreening_csv(ps_results, output_dir / "feature_prescreening.csv")
    plot_mutual_information(ps_results, plots_dir)
    plot_anova_prescreening(ps_results, plots_dir)

    logger.info(sep)
    logger.info("EDA OVERVIEW MODULE COMPLETE")
    logger.info(sep)

    return {
        "overview"     : overview,
        "tld"          : tld_results,
        "prescreening" : ps_results,
    }
