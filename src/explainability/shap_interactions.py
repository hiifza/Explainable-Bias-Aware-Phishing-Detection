"""
src/explainability/shap_interactions.py
-----------------------------------------
SHAP interaction analysis: pairwise feature interaction values,
interaction heatmap, and top-20 interaction pairs.

When native SHAP is available and the model is tree-based,
TreeExplainer.shap_interaction_values() is used.
Otherwise, interactions are approximated as the outer product of
SHAP values per sample, averaged over all samples.

Public API
----------
    compute_interaction_values(result, X, sample_n) -> np.ndarray
    get_top_interaction_pairs(interaction_matrix, feature_names, top_n) -> pd.DataFrame
    plot_interaction_heatmap(interaction_matrix, feature_names, out_dir, top_n) -> Path
    plot_top_interaction_pairs(pairs_df, out_dir) -> Path
    run_interaction_analysis(result, X_background, plots_dir, reports_dir) -> dict
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

from src.utils.logger                  import get_logger
from src.explainability.shap_explainer import SHAPResult

logger = get_logger(__name__)

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


def _setup():
    sns.set_theme(style="white", font_scale=1.0)
    plt.rcParams.update({"figure.facecolor": "white", "savefig.dpi": 150})


# ── Interaction values ────────────────────────────────────────────────────────

def compute_interaction_values(
    result      : SHAPResult,
    X_background: pd.DataFrame,
    sample_n    : int = 500,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute a (n_features × n_features) interaction matrix.

    Native SHAP (tree models): uses shap_interaction_values() then averages
    |interaction[i,j]| over samples → matrix[i,j].

    Fallback: approximates as outer product of mean SHAP values.
    Matrix[i,j] = mean_i(|shap_i|) * mean_j(|shap_j|) * cross_corr(shap_i, shap_j).

    Parameters
    ----------
    result       : SHAPResult
    X_background : training data (for tree explainer context)
    sample_n     : rows to use for interaction computation
    random_state : seed

    Returns
    -------
    np.ndarray  shape (n_features, n_features)  — symmetric, diag excluded
    """
    n_feats = result.n_features
    rng     = np.random.default_rng(random_state)

    # Subsample
    n_use = min(sample_n, result.n_samples)
    idx   = rng.choice(result.n_samples, size=n_use, replace=False)
    sv    = result.shap_values[idx]          # (n_use, n_feats)

    if _SHAP_AVAILABLE and result.model_type == "tree":
        try:
            n_bg  = min(500, len(X_background))
            bg_idx= rng.choice(len(X_background), size=n_bg, replace=False)
            X_bg  = X_background.iloc[bg_idx]
            X_sub = result.X_explained.iloc[idx]

            explainer   = _shap.TreeExplainer(
                result.fitted_model if hasattr(result, "fitted_model")
                else _get_model_from_result(result),
                data=X_bg.values,
                feature_perturbation="interventional",
            )
            shap_ival   = explainer.shap_interaction_values(X_sub.values)
            # shape: (n_samples, n_feats, n_feats) or list thereof
            if isinstance(shap_ival, list):
                shap_ival = shap_ival[1]  # class 1

            interaction_matrix = np.abs(shap_ival).mean(axis=0)
            np.fill_diagonal(interaction_matrix, 0)
            logger.info("SHAP interaction values: native TreeExplainer ✓")
            return interaction_matrix

        except Exception as e:
            logger.warning(f"Native interaction failed ({e}), using fallback")

    # ── Fallback: outer product approximation ────────────────────────────────
    # Interaction[i,j] = correlation between shap_i and shap_j
    # weighted by their marginal importances
    mean_abs     = np.abs(sv).mean(axis=0)           # (n_feats,)
    outer        = np.outer(mean_abs, mean_abs)       # (n_feats, n_feats)

    # Cross-correlation matrix of SHAP columns
    sv_centred   = sv - sv.mean(axis=0, keepdims=True)
    sv_std       = sv_centred.std(axis=0)
    sv_std       = np.where(sv_std < 1e-12, 1.0, sv_std)
    sv_norm      = sv_centred / sv_std

    corr_matrix  = (sv_norm.T @ sv_norm) / max(n_use - 1, 1)
    corr_matrix  = np.abs(corr_matrix)

    interaction_matrix = outer * corr_matrix
    np.fill_diagonal(interaction_matrix, 0)

    logger.info("SHAP interaction values: fallback outer-product approximation ✓")
    return interaction_matrix


def _get_model_from_result(result: SHAPResult):
    """Try to recover the fitted model from the result object."""
    raise AttributeError("Fitted model not stored in SHAPResult")


# ── Top interaction pairs ─────────────────────────────────────────────────────

def get_top_interaction_pairs(
    interaction_matrix: np.ndarray,
    feature_names     : list[str],
    top_n             : int = 20,
) -> pd.DataFrame:
    """
    Extract top-n feature pairs by interaction strength.

    Returns
    -------
    pd.DataFrame  columns: rank, feature_A, feature_B, interaction_strength
    """
    n = len(feature_names)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append({
                "feature_A"          : feature_names[i],
                "feature_B"          : feature_names[j],
                "interaction_strength": float(interaction_matrix[i, j]),
            })

    df = (
        pd.DataFrame(rows)
        .sort_values("interaction_strength", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ── Interaction heatmap ───────────────────────────────────────────────────────

def plot_interaction_heatmap(
    interaction_matrix: np.ndarray,
    feature_names     : list[str],
    out_dir           : Path,
    top_n_features    : int = 20,
) -> Path:
    """
    Heatmap of interaction strengths for the top-n features by marginal
    importance (sum of row interactions).
    """
    _setup()
    n    = interaction_matrix.shape[0]
    k    = min(top_n_features, n)

    # Select top-k features by total interaction (row sums)
    row_sums = interaction_matrix.sum(axis=1)
    top_idx  = np.argsort(row_sums)[::-1][:k]
    sub_mat  = interaction_matrix[np.ix_(top_idx, top_idx)]
    sub_names= [feature_names[i] for i in top_idx]

    fig, ax  = plt.subplots(figsize=(max(10, k * 0.5), max(9, k * 0.46)))
    sns.heatmap(
        sub_mat, ax=ax,
        xticklabels=sub_names,
        yticklabels=sub_names,
        cmap="YlOrRd", square=True,
        linewidths=0.3, linecolor="white",
        cbar_kws={"shrink": 0.65, "label": "Interaction strength"},
        annot=(k <= 15),
        fmt=".3f",
        annot_kws={"size": 7},
    )
    ax.set_title(
        f"SHAP Interaction Heatmap — Top {k} Features",
        fontsize=13, fontweight="700", pad=12,
    )
    ax.tick_params(axis="x", rotation=55, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "interaction_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")
    return out_path


def plot_top_interaction_pairs(
    pairs_df: pd.DataFrame,
    out_dir : Path,
    top_n   : int = 20,
) -> Path:
    """
    Horizontal bar chart of top interaction pairs.
    """
    _setup()
    df = pairs_df.head(top_n).copy()
    df["pair"] = df["feature_A"] + "\n↔ " + df["feature_B"]
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    ax.barh(df["pair"], df["interaction_strength"],
            color="#533AB7", edgecolor="white", linewidth=0.5)
    for _, row in df.iterrows():
        ax.text(row["interaction_strength"] + 1e-5, row["pair"],
                f"{row['interaction_strength']:.4f}",
                va="center", fontsize=8)
    ax.set_xlabel("Interaction strength")
    ax.set_title(f"Top {top_n} SHAP Feature Interaction Pairs",
                 fontsize=12, fontweight="700")
    sns.despine(ax=ax)
    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "top_interaction_pairs.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")
    return out_path


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_interaction_analysis(
    result      : SHAPResult,
    X_background: pd.DataFrame,
    plots_dir   : str | Path = "outputs/plots/shap/interactions",
    reports_dir : str | Path = "outputs/reports",
    sample_n    : int = 500,
    top_n_pairs : int = 20,
) -> dict:
    """
    Full interaction analysis: compute matrix, save heatmap and top-pairs
    chart, write CSV.

    Returns
    -------
    dict  keys: interaction_matrix, top_pairs_df, heatmap_path, pairs_path,
               pairs_csv_path
    """
    plots_dir   = Path(plots_dir)
    reports_dir = Path(reports_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 55)
    logger.info("M7.1 — SHAP INTERACTION ANALYSIS")
    logger.info("=" * 55)

    # 1. Compute interaction matrix
    int_matrix = compute_interaction_values(result, X_background, sample_n)

    # 2. Top pairs
    top_pairs = get_top_interaction_pairs(
        int_matrix, result.feature_names, top_n=top_n_pairs
    )
    pairs_csv = reports_dir / "shap_interaction_pairs.csv"
    top_pairs.to_csv(pairs_csv, index=False)
    logger.info(f"Saved: shap_interaction_pairs.csv  ({len(top_pairs)} rows)")

    # 3. Plots
    heatmap_p = plot_interaction_heatmap(int_matrix, result.feature_names,
                                         plots_dir)
    pairs_p   = plot_top_interaction_pairs(top_pairs, plots_dir)

    logger.info("Interaction analysis complete")
    return {
        "interaction_matrix": int_matrix,
        "top_pairs_df"      : top_pairs,
        "heatmap_path"      : heatmap_p,
        "pairs_path"        : pairs_p,
        "pairs_csv_path"    : pairs_csv,
    }
