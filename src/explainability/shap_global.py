"""
src/explainability/shap_global.py
-----------------------------------
Global SHAP explainability: summary plots, feature importance bar charts,
and dependence plots for the top features.

All plots are implemented in pure matplotlib/seaborn so they work
whether native SHAP or the fallback explainer was used.

Public API
----------
    plot_beeswarm(result, out_dir, max_display)          -> Path
    plot_importance_bar(result, out_dir, max_display)    -> Path
    plot_dependence(result, feature, out_dir)            -> Path
    plot_top_dependence(result, out_dir, top_n)          -> list[Path]
    save_feature_ranking_csv(result, out_dir)            -> Path
    run_global_analysis(result, plots_dir, reports_dir)  -> dict
"""

import sys
from pathlib import Path
from typing  import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm     as cm
import numpy  as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger                import get_logger
from src.explainability.shap_explainer import SHAPResult

logger = get_logger(__name__)


def _setup():
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({"figure.facecolor": "white", "savefig.dpi": 150})


# ── Beeswarm summary ──────────────────────────────────────────────────────────

def plot_beeswarm(
    result     : SHAPResult,
    out_dir    : Path,
    max_display: int = 20,
) -> Path:
    """
    SHAP summary beeswarm plot: each dot = one sample, coloured by feature
    value, x-axis = SHAP value.

    Returns
    -------
    Path to saved PNG
    """
    _setup()
    shap_vals  = result.shap_values
    feat_names = result.feature_names
    X_vals     = result.X_explained.values

    mean_abs   = np.abs(shap_vals).mean(axis=0)
    top_idx    = np.argsort(mean_abs)[::-1][:max_display]
    # Reverse for bottom-up plot
    top_idx    = top_idx[::-1]

    n_feats    = len(top_idx)
    fig, ax    = plt.subplots(figsize=(10, max(6, n_feats * 0.32)))

    # Normalise feature values for colouring (0=low, 1=high)
    for plot_row, feat_i in enumerate(top_idx):
        sv     = shap_vals[:, feat_i]
        fv_raw = X_vals[:, feat_i].astype(float)
        fv_min, fv_max = fv_raw.min(), fv_raw.max()
        fv_norm = (fv_raw - fv_min) / max(fv_max - fv_min, 1e-12)

        # Jitter y-axis for visibility
        jitter = np.random.default_rng(feat_i).uniform(-0.3, 0.3, size=len(sv))
        y_pos  = plot_row + jitter

        ax.scatter(sv, y_pos,
                   c=fv_norm, cmap="RdBu_r", vmin=0, vmax=1,
                   alpha=0.5, s=5, rasterized=True)

    ax.set_yticks(range(n_feats))
    ax.set_yticklabels([feat_names[i] for i in top_idx], fontsize=9)
    ax.axvline(0, color="#888", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value\n(← Phishing  |  Legitimate →)", fontsize=11)
    ax.set_title(
        f"SHAP Summary (Beeswarm) — Top {max_display} Features\n"
        f"({result.model_class}, Track B, "
        f"{'native' if result.is_native_shap else 'fallback'} SHAP, "
        f"n={result.n_samples:,})",
        fontsize=12, fontweight="700",
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Feature value\n(low → high)", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"], fontsize=8)

    sns.despine(ax=ax)
    plt.tight_layout()

    out_path = Path(out_dir) / "summary_beeswarm.png"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")
    return out_path


# ── Importance bar chart ──────────────────────────────────────────────────────

def plot_importance_bar(
    result     : SHAPResult,
    out_dir    : Path,
    max_display: int = 25,
) -> Path:
    """
    Horizontal bar chart of mean |SHAP| per feature (top max_display).
    """
    _setup()
    ranking   = result.get_feature_ranking().head(max_display)
    # Reverse for bottom-to-top
    ranking   = ranking.iloc[::-1]

    colors = [
        "#E24B4A" if r <= 5 else "#EF9F27" if r <= 10 else "#378ADD"
        for r in ranking["rank"]
    ]

    fig, ax = plt.subplots(figsize=(9, max(6, max_display * 0.32)))
    bars = ax.barh(ranking["feature"], ranking["mean_abs_shap"],
                   color=colors[::-1], edgecolor="white", linewidth=0.5)
    for bar, rel in zip(bars, ranking["relative_importance"][::-1]):
        ax.text(bar.get_width() + 0.0002,
                bar.get_y() + bar.get_height() / 2,
                f"{rel*100:.1f}%",
                va="center", fontsize=8)

    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(
        f"Global Feature Importance (Mean |SHAP|)\n"
        f"{result.model_class} — Track B — Top {max_display}",
        fontsize=12, fontweight="700",
    )
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#E24B4A", label="Top 5"),
        Patch(facecolor="#EF9F27", label="Top 6–10"),
        Patch(facecolor="#378ADD", label="Top 11+"),
    ], fontsize=9, loc="lower right")
    sns.despine(ax=ax)
    plt.tight_layout()

    out_path = Path(out_dir) / "global_importance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")
    return out_path


# ── Dependence plot ───────────────────────────────────────────────────────────

def plot_dependence(
    result     : SHAPResult,
    feature    : str,
    out_dir    : Path,
    color_feat : Optional[str] = None,
) -> Optional[Path]:
    """
    Scatter plot of feature value vs SHAP value for one feature,
    coloured by a second feature (auto-selected if color_feat is None).
    """
    _setup()
    feat_names = result.feature_names
    if feature not in feat_names:
        logger.warning(f"Feature '{feature}' not found — skipping dependence plot")
        return None

    feat_idx  = feat_names.index(feature)
    feat_vals = result.X_explained[feature].values.astype(float)
    shap_vals = result.shap_values[:, feat_idx]

    # Auto-select colour feature (highest interaction proxy = max correlation)
    if color_feat is None:
        corrs = []
        for j, fn in enumerate(feat_names):
            if fn != feature:
                sv_j = result.shap_values[:, j]
                c    = abs(np.corrcoef(shap_vals, sv_j)[0, 1]) if len(shap_vals) > 2 else 0
                corrs.append((fn, c))
        if corrs:
            color_feat = max(corrs, key=lambda x: x[1])[0]
        else:
            color_feat = feature

    cf_idx  = feat_names.index(color_feat) if color_feat in feat_names else feat_idx
    cf_vals = result.X_explained.iloc[:, cf_idx].values.astype(float)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(feat_vals, shap_vals,
                    c=cf_vals, cmap="RdBu_r",
                    alpha=0.6, s=8, rasterized=True)
    ax.axhline(0, color="#888", linestyle="--", linewidth=0.8)
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel(f"SHAP value for {feature}", fontsize=11)
    ax.set_title(f"Dependence Plot: {feature}\n(coloured by {color_feat})",
                 fontsize=12, fontweight="700")
    plt.colorbar(sc, ax=ax, label=color_feat, shrink=0.7)
    sns.despine(ax=ax)
    plt.tight_layout()

    safe = feature.lower().replace(" ", "_").replace("/", "_")
    out_path = Path(out_dir) / f"{safe}.png"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.debug(f"Dependence saved: {out_path.name}")
    return out_path


def plot_top_dependence(
    result : SHAPResult,
    out_dir: Path,
    top_n  : int = 10,
) -> list[Path]:
    """Generate dependence plots for the top-n most important features."""
    ranking = result.get_feature_ranking().head(top_n)
    saved   = []
    for feat in ranking["feature"]:
        p = plot_dependence(result, feat, out_dir)
        if p:
            saved.append(p)
    logger.info(f"Dependence plots saved: {len(saved)}")
    return saved


# ── Feature ranking CSV ───────────────────────────────────────────────────────

def save_feature_ranking_csv(
    result    : SHAPResult,
    output_dir: Path,
) -> Path:
    """Save full feature ranking to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df   = result.get_feature_ranking()
    path = output_dir / "shap_feature_ranking.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved: shap_feature_ranking.csv  ({len(df)} features)")
    return path


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_global_analysis(
    result      : SHAPResult,
    plots_dir   : str | Path,
    reports_dir : str | Path,
    max_display : int = 20,
    top_dep_n   : int = 10,
) -> dict:
    """
    Run all global SHAP analyses and save all artefacts.

    Returns
    -------
    dict  keys: beeswarm_path, importance_path, dependence_paths,
               ranking_csv_path, feature_ranking_df
    """
    plots_dir   = Path(plots_dir)
    reports_dir = Path(reports_dir)

    logger.info("=" * 55)
    logger.info("M7.1 — GLOBAL SHAP ANALYSIS")
    logger.info("=" * 55)

    beeswarm_p   = plot_beeswarm(result, plots_dir, max_display)
    importance_p = plot_importance_bar(result, plots_dir, max_display)
    dep_paths    = plot_top_dependence(result, plots_dir / "dependence", top_dep_n)
    ranking_csv  = save_feature_ranking_csv(result, reports_dir)
    ranking_df   = result.get_feature_ranking()

    logger.info(
        f"Global analysis complete: "
        f"beeswarm + importance bar + {len(dep_paths)} dependence plots"
    )
    return {
        "beeswarm_path"    : beeswarm_p,
        "importance_path"  : importance_p,
        "dependence_paths" : dep_paths,
        "ranking_csv_path" : ranking_csv,
        "feature_ranking_df": ranking_df,
    }
