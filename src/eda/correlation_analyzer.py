"""
src/eda/correlation_analyzer.py
--------------------------------
Pearson and Spearman correlation analysis for Module M2.1.

Public API
----------
    compute_pearson_matrix(df, features)         -> pd.DataFrame
    compute_spearman_matrix(df, features)        -> pd.DataFrame
    get_top_positive_correlations(matrix, n)     -> pd.DataFrame
    get_top_negative_correlations(matrix, n)     -> pd.DataFrame
    build_correlation_network(matrix, threshold) -> pd.DataFrame
    run_correlation_analysis(df, features, out_dir, plots_dir) -> dict
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
from src.eda.visualization_manager import save_figure, setup_plot_style

logger = get_logger(__name__)

TARGET = "label"


# ── Matrix computation ────────────────────────────────────────────────────────

def compute_pearson_matrix(
    df      : pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix for numeric features.

    Parameters
    ----------
    df       : DataFrame containing the features
    features : list of column names (numeric only)

    Returns
    -------
    pd.DataFrame  — symmetric correlation matrix
    """
    num_feats = [c for c in features
                 if df[c].dtype in [np.int64, np.float64, "int64", "float64"]]
    logger.info(f"Computing Pearson correlation ({len(num_feats)} numeric features) …")
    matrix = df[num_feats].corr(method="pearson")
    logger.info("Pearson matrix computed ✓")
    return matrix


def compute_spearman_matrix(
    df      : pd.DataFrame,
    features: list[str],
    sample_n: int = 50_000,
) -> pd.DataFrame:
    """
    Compute Spearman rank correlation matrix.

    Spearman is more robust to outliers and monotonic non-linear
    relationships than Pearson.  We sample for speed on large frames.

    Parameters
    ----------
    df       : DataFrame
    features : numeric feature names
    sample_n : max rows to use (Spearman is O(n log n))

    Returns
    -------
    pd.DataFrame  — symmetric rank-correlation matrix
    """
    num_feats = [c for c in features
                 if df[c].dtype in [np.int64, np.float64, "int64", "float64"]]

    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), size=min(sample_n, len(df)), replace=False)
    df_s = df.iloc[idx][num_feats]

    logger.info(
        f"Computing Spearman correlation ({len(num_feats)} features, "
        f"{len(df_s):,} rows) …"
    )
    matrix = df_s.corr(method="spearman")
    logger.info("Spearman matrix computed ✓")
    return matrix


# ── Pair extraction ───────────────────────────────────────────────────────────

def _extract_pairs(
    matrix   : pd.DataFrame,
    positive : bool = True,
    n        : int  = 20,
    exclude_self: bool = True,
) -> pd.DataFrame:
    """
    Extract the top-n correlated pairs from a correlation matrix.

    Parameters
    ----------
    matrix       : square correlation DataFrame
    positive     : True → top positive  |  False → top negative (most negative)
    n            : number of pairs to return
    exclude_self : exclude diagonal (self-correlation)
    """
    mat = matrix.copy()
    mat_vals = mat.values.copy()          # ensure writable before fill_diagonal
    if exclude_self:
        np.fill_diagonal(mat_vals, np.nan)
    mat = pd.DataFrame(mat_vals, index=mat.index, columns=mat.columns)

    # Upper triangle only to avoid duplicates
    upper_mask = np.triu(np.ones(mat.shape, dtype=bool), k=1)
    mat_upper  = mat.where(upper_mask)

    stacked = (
        mat_upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feat_A", "level_1": "feat_B", 0: "pearson_r"})
    )
    stacked = stacked.dropna(subset=["pearson_r"])

    if positive:
        result = stacked.nlargest(n, "pearson_r")
    else:
        result = stacked.nsmallest(n, "pearson_r")

    result["abs_r"] = result["pearson_r"].abs()
    return result.reset_index(drop=True)


def get_top_positive_correlations(
    matrix: pd.DataFrame,
    n     : int = 20,
) -> pd.DataFrame:
    """Return top-n positive feature-feature correlation pairs."""
    pairs = _extract_pairs(matrix, positive=True, n=n)
    logger.info(f"Top-{n} positive pairs extracted (max r={pairs['pearson_r'].max():.4f})")
    return pairs


def get_top_negative_correlations(
    matrix: pd.DataFrame,
    n     : int = 20,
) -> pd.DataFrame:
    """Return top-n negative feature-feature correlation pairs."""
    pairs = _extract_pairs(matrix, positive=False, n=n)
    logger.info(f"Top-{n} negative pairs extracted (min r={pairs['pearson_r'].min():.4f})")
    return pairs


def build_correlation_network(
    matrix   : pd.DataFrame,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Build a full correlation network edge table — all pairs above threshold.
    Used for correlation_analysis.csv output.

    Parameters
    ----------
    matrix    : Pearson correlation matrix
    threshold : |r| threshold for inclusion

    Returns
    -------
    pd.DataFrame  columns: feat_A, feat_B, pearson_r, abs_r, strength
    """
    mat      = matrix.copy()
    mat_vals = mat.values.copy()
    np.fill_diagonal(mat_vals, np.nan)
    mat   = pd.DataFrame(mat_vals, index=mat.index, columns=mat.columns)
    upper = mat.where(np.triu(np.ones(mat.shape, dtype=bool), k=1))

    stacked = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feat_A", "level_1": "feat_B", 0: "pearson_r"})
        .dropna(subset=["pearson_r"])
    )
    stacked["abs_r"] = stacked["pearson_r"].abs()
    stacked = stacked[stacked["abs_r"] >= threshold].copy()

    def _strength(r: float) -> str:
        if   r >= 0.90: return "very_high"
        elif r >= 0.75: return "high"
        elif r >= 0.50: return "medium"
        else:           return "low"

    stacked["strength"] = stacked["abs_r"].apply(_strength)
    stacked = stacked.sort_values("abs_r", ascending=False).reset_index(drop=True)

    n_high = int((stacked["abs_r"] >= 0.75).sum())
    logger.info(
        f"Correlation network: {len(stacked)} pairs |r|≥{threshold}  "
        f"({n_high} high-corr |r|≥0.75)"
    )
    return stacked


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    matrix    : pd.DataFrame,
    title     : str,
    output_path: Path,
    annot_threshold: float = 0.60,
) -> None:
    """
    Plot a lower-triangle correlation heatmap.

    Cells with |r| < annot_threshold are not annotated to keep the
    chart readable at high feature counts.

    Parameters
    ----------
    matrix             : square correlation DataFrame
    title              : chart title
    output_path        : destination PNG file
    annot_threshold    : |r| below which cell annotations are suppressed
    """
    setup_plot_style()
    n = len(matrix)

    # Lower triangle mask
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)

    # Selective annotation: show value only if |r| >= threshold
    annot_array = np.full(matrix.shape, "", dtype=object)
    for i in range(n):
        for j in range(n):
            if not mask[i, j] and abs(matrix.iloc[i, j]) >= annot_threshold:
                annot_array[i, j] = f"{matrix.iloc[i, j]:.2f}"

    fig, ax = plt.subplots(figsize=(max(14, n * 0.38), max(12, n * 0.34)))
    sns.heatmap(
        matrix, ax=ax, mask=mask,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.25, linecolor="#EBEBEB",
        cbar_kws={"shrink": 0.55, "label": "Correlation r"},
        annot=annot_array if n <= 55 else False,
        fmt="",
        annot_kws={"size": 7},
    )
    ax.set_title(title, fontsize=13, fontweight="600", pad=14)
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()
    save_figure(fig, output_path)


def plot_top_corr_pairs(
    top_pos  : pd.DataFrame,
    top_neg  : pd.DataFrame,
    plots_dir: Path,
    n        : int = 20,
) -> None:
    """Side-by-side bar charts of top positive and negative corr pairs."""
    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, df, title, color in [
        (axes[0], top_pos, f"Top {n} Positive Correlations", "#185FA5"),
        (axes[1], top_neg, f"Top {n} Negative Correlations", "#E24B4A"),
    ]:
        if df is None or len(df) == 0:
            ax.set_visible(False)
            continue
        plot_df = df.head(n).copy()
        labels  = [f"{r.feat_A}\n↔ {r.feat_B}" for _, r in plot_df.iterrows()]
        values  = plot_df["pearson_r"].abs().values
        ax.barh(labels, values, color=color, edgecolor="white", linewidth=0.5)
        for val, label in zip(values, labels):
            ax.text(val + 0.005, labels.index(label),
                    f"{val:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("|r|")
        ax.set_title(title, fontsize=12, fontweight="600")
        sns.despine(ax=ax)

    plt.suptitle("Top Correlation Pairs (Pearson)", fontsize=13, fontweight="700", y=1.01)
    plt.tight_layout()
    save_figure(fig, plots_dir / "top_correlation_pairs.png")


def plot_target_correlations(
    df        : pd.DataFrame,
    features  : list[str],
    plots_dir : Path,
) -> None:
    """
    Horizontal bar chart of |Pearson r| between each feature and the
    target label, coloured by strength.
    Includes URLSimilarityIndex as a reference bar (labelled as leakage).
    """
    setup_plot_style()

    all_feats = features + ["URLSimilarityIndex"] \
        if "URLSimilarityIndex" in df.columns else features

    num_feats = [c for c in all_feats
                 if df[c].dtype in [np.int64, np.float64, "int64", "float64"]]

    corr_vals = (
        df[num_feats + [TARGET]]
        .corr()[TARGET]
        .drop(TARGET)
        .abs()
        .sort_values(ascending=True)
    )

    colors = []
    for feat in corr_vals.index:
        if feat == "URLSimilarityIndex":
            colors.append("#A32D2D")          # critical leakage — dark red
        elif corr_vals[feat] >= 0.60:
            colors.append("#E24B4A")
        elif corr_vals[feat] >= 0.40:
            colors.append("#EF9F27")
        else:
            colors.append("#378ADD")

    fig, ax = plt.subplots(figsize=(9, max(10, len(corr_vals) * 0.27)))
    ax.barh(corr_vals.index, corr_vals.values,
            color=colors, edgecolor="white", linewidth=0.4)
    for v, name in zip(corr_vals.values, corr_vals.index):
        ax.text(v + 0.004, name, f"{v:.3f}", va="center", fontsize=8)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#A32D2D", label="Critical leakage (Track A only)"),
        Patch(facecolor="#E24B4A", label="|r| ≥ 0.60 — strong"),
        Patch(facecolor="#EF9F27", label="0.40 ≤ |r| < 0.60 — moderate"),
        Patch(facecolor="#378ADD", label="|r| < 0.40 — weak"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
    ax.set_xlabel("|Pearson r| with label")
    ax.set_title("Feature Correlation with Target Label\n(Track B + URLSimilarityIndex for reference)",
                 fontsize=12, fontweight="600")
    ax.set_xlim(0, max(corr_vals.values) * 1.15)
    sns.despine(ax=ax)
    plt.tight_layout()
    save_figure(fig, plots_dir / "target_correlation_full.png")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_correlation_analysis(
    df        : pd.DataFrame,
    features  : list[str],
    output_dir: str | Path = "outputs/reports",
    plots_dir : str | Path = "outputs/plots/eda",
) -> dict:
    """
    Execute the full correlation analysis pipeline.

    Parameters
    ----------
    df         : clean DataFrame
    features   : Track B feature list
    output_dir : CSV output path
    plots_dir  : plot output directory

    Returns
    -------
    dict  keys: pearson_matrix, spearman_matrix, top_positive,
                top_negative, network, n_high_pairs
    """
    output_dir = Path(output_dir)
    plots_dir  = Path(plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("M2.1 — CORRELATION ANALYSIS MODULE")
    logger.info("=" * 60)

    # 1. Compute matrices
    pearson_mat  = compute_pearson_matrix(df, features)
    spearman_mat = compute_spearman_matrix(df, features)

    # 2. Top pairs
    top_pos = get_top_positive_correlations(pearson_mat, n=20)
    top_neg = get_top_negative_correlations(pearson_mat, n=20)

    # 3. Network table → CSV
    network = build_correlation_network(pearson_mat, threshold=0.30)
    network.to_csv(output_dir / "correlation_analysis.csv", index=False)
    logger.info(f"Saved: correlation_analysis.csv")

    n_high = int((network["abs_r"] >= 0.75).sum())

    # 4. Plots
    plot_correlation_heatmap(
        pearson_mat,
        "Pearson Correlation Heatmap (Track B numeric features)",
        plots_dir / "correlation_heatmap_pearson.png",
    )
    plot_correlation_heatmap(
        spearman_mat,
        "Spearman Rank Correlation Heatmap (Track B numeric features)",
        plots_dir / "correlation_heatmap_spearman.png",
    )
    plot_top_corr_pairs(top_pos, top_neg, plots_dir)
    plot_target_correlations(df, features, plots_dir)

    logger.info("CORRELATION ANALYSIS COMPLETE")

    return {
        "pearson_matrix" : pearson_mat,
        "spearman_matrix": spearman_mat,
        "top_positive"   : top_pos,
        "top_negative"   : top_neg,
        "network"        : network,
        "n_high_pairs"   : n_high,
    }
