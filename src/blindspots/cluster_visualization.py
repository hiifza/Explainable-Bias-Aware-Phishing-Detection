"""
src/blindspots/cluster_visualization.py
-----------------------------------------
Cluster visualizations: PCA maps, UMAP maps (if available),
confidence heatmaps, and failure density plots.

Public API
----------
    plot_pca_cluster_map(archetype_result, fcs, plots_dir)         -> Path
    plot_umap_cluster_map(X_fail, labels, fcs, plots_dir)          -> Optional[Path]
    plot_confidence_heatmap(archetype_result, severity_df, plots_dir) -> Path
    plot_failure_density(fcs, X_test, feature_names, plots_dir)    -> Path
    run_cluster_visualization(archetype_result, fcs, severity_df,
                               X_test, feature_names, plots_dir)   -> dict
"""

import sys
from pathlib import Path
from typing  import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy  as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger                          import get_logger
from src.blindspots.failure_case_extractor     import FailureCaseSet
from src.blindspots.failure_archetype_discovery import ArchetypeResult

logger = get_logger(__name__)

# Try UMAP
try:
    from umap import UMAP as _UMAP
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False


def _setup():
    sns.set_theme(style="white", font_scale=1.0)
    plt.rcParams.update({"figure.facecolor": "white", "savefig.dpi": 150})


def plot_pca_cluster_map(
    archetype_result: ArchetypeResult,
    fcs             : FailureCaseSet,
    plots_dir       : Path,
) -> Path:
    """
    Scatter plot of failure cases in PCA space, coloured by cluster/archetype.
    Errors marked with ✕, uncertain samples with ○.
    """
    _setup()
    ar      = archetype_result
    if ar.X_pca.shape[1] < 2 or len(ar.X_pca) == 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "Insufficient data for PCA plot",
                ha="center", va="center", transform=ax.transAxes)
        out = plots_dir / "pca_cluster_map.png"; plots_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); return out

    pca_x   = ar.X_pca[:, 0]
    pca_y   = ar.X_pca[:, 1]
    labels  = ar.labels
    indices = set(ar.sample_indices.tolist())

    # Map sample_idx → error flag
    error_map = {c.sample_idx: c.is_error for c in fcs}
    is_error  = np.array([error_map.get(idx, False) for idx in ar.sample_indices])

    # Colour per cluster
    n_clusters = ar.n_clusters

    import matplotlib

    cmap = matplotlib.colormaps["tab10"]
    colors = [cmap((labels[i] % 10) / 10) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(9, 7))

    # Non-error cases
    ne_mask = ~is_error
    if ne_mask.any():
        ax.scatter(pca_x[ne_mask], pca_y[ne_mask],
                   c=[colors[i] for i in np.where(ne_mask)[0]],
                   alpha=0.55, s=30, edgecolors="white", linewidths=0.3,
                   marker="o", label="Uncertain (correct)", zorder=2)

    # Error cases
    er_mask = is_error
    if er_mask.any():
        ax.scatter(pca_x[er_mask], pca_y[er_mask],
                   c=[colors[i] for i in np.where(er_mask)[0]],
                   alpha=0.95, s=120, edgecolors="#333", linewidths=1.5,
                   marker="X", label="Hard errors (FP/FN)", zorder=5)

    # Cluster centroids
    for m in ar.cluster_meta:
        k = m["cluster"]
        ax.scatter(m["pca_centroid_x"], m["pca_centroid_y"],
                   c=[cmap(k)], s=200, marker="*", edgecolors="#333",
                   linewidths=1.0, zorder=6)
        ax.annotate(m["label"][:25], (m["pca_centroid_x"], m["pca_centroid_y"]),
                    fontsize=7, fontweight="600",
                    xytext=(6, 4), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="#ccc", alpha=0.85))

    # Legend
    legend_handles = [
        mpatches.Patch(color=cmap(k), label=f"[{k}] {m['label'][:20]}")
        for k, m in enumerate(ar.cluster_meta)
    ]
    legend_handles += [
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#888",
                   markersize=8, label="Uncertain (correct)"),
        plt.Line2D([0],[0], marker="X", color="w", markerfacecolor="#333",
                   markersize=10, label="Hard errors"),
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, loc="best",
              bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.set_xlabel(f"PCA Component 1")
    ax.set_ylabel(f"PCA Component 2")
    ax.set_title(
        f"Failure Archetype Cluster Map (PCA)\n"
        f"{ar.n_clusters} clusters  silhouette={ar.silhouette_score:.3f}  "
        f"PCA_var={ar.explained_var:.2%}",
        fontsize=12, fontweight="700",
    )
    sns.despine(ax=ax)
    plt.tight_layout()

    out = plots_dir / "pca_cluster_map.png"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out.name}")
    return out


def plot_umap_cluster_map(
    X_fail   : np.ndarray,
    labels   : np.ndarray,
    fcs      : FailureCaseSet,
    plots_dir: Path,
) -> Optional[Path]:
    """UMAP 2-D projection (requires umap-learn). Returns None if unavailable."""
    if not _UMAP_AVAILABLE:
        logger.info("UMAP not available — skipping UMAP plot (pip install umap-learn)")
        return None

    _setup()
    try:
        reducer = _UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(X_fail)-1))
        X_2d    = reducer.fit_transform(X_fail)
    except Exception as e:
        logger.warning(f"UMAP failed: {e}")
        return None

    error_idx = {c.sample_idx for c in fcs if c.is_error}
    cmap      = plt.cm.get_cmap("tab10", max(len(set(labels)), 1))
    colors    = [cmap(l) for l in labels]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.6, s=25,
               edgecolors="white", linewidths=0.3)
    ax.set_title("UMAP Failure Case Projection",
                 fontsize=12, fontweight="700")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    sns.despine(ax=ax); plt.tight_layout()

    out = plots_dir / "umap_cluster_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out.name}")
    return out


def plot_confidence_heatmap(
    archetype_result: ArchetypeResult,
    severity_df     : pd.DataFrame,
    plots_dir       : Path,
) -> Path:
    """
    Heatmap: cluster × metric (error_rate, mean_confidence, mean_severity).
    """
    _setup()
    ar = archetype_result
    if not ar.cluster_meta:
        return plots_dir / "confidence_heatmap.png"

    # Merge severity_df with archetype_df
    merged = ar.archetype_df.merge(
        severity_df[["sample_idx","severity_score_norm","confidence"]],
        on="sample_idx", how="left",
    )

    heat_data = {}
    for m in ar.cluster_meta:
        k   = m["cluster"]
        sub = merged[merged["cluster"] == k]
        heat_data[m["label"][:20]] = {
            "error_rate"     : round(sub["is_error"].mean(), 4) if "is_error" in sub else 0,
            "mean_confidence": round(sub["confidence"].mean(), 4) if "confidence" in sub else 0,
            "mean_severity"  : round(sub["severity_score_norm"].mean(), 4) if "severity_score_norm" in sub else 0,
            "n_samples"      : len(sub),
        }

    hdf = pd.DataFrame(heat_data).T[["error_rate","mean_confidence","mean_severity","n_samples"]]

    fig, ax = plt.subplots(figsize=(8, max(3, len(hdf) * 0.6)))
    sns.heatmap(hdf.drop(columns=["n_samples"]), ax=ax,
                cmap="RdYlGn_r", annot=True, fmt=".3f",
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.7}, annot_kws={"size": 10})
    ax.set_title("Cluster Risk Heatmap\n(error_rate · confidence · severity)",
                 fontsize=12, fontweight="700")
    ax.tick_params(axis="x", rotation=15); ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()

    out = plots_dir / "confidence_heatmap.png"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out.name}")
    return out


def plot_failure_density(
    fcs          : FailureCaseSet,
    X_test       : pd.DataFrame,
    feature_names: list[str],
    plots_dir    : Path,
    top_features : int = 4,
) -> Path:
    """
    KDE density plots: failure cases vs all test samples for top features.
    """
    _setup()
    fail_idx    = np.array([c.sample_idx for c in fcs if c.sample_idx < len(X_test)])
    all_idx     = np.arange(len(X_test))

    # Use features 0-3 if no feature importance available
    feat_subset = feature_names[:top_features]

    fig, axes = plt.subplots(1, len(feat_subset), figsize=(len(feat_subset)*4.5, 4))
    if len(feat_subset) == 1: axes = [axes]

    for ax, feat in zip(axes, feat_subset):
        if feat not in X_test.columns: continue
        all_vals  = X_test.iloc[all_idx][feat].dropna()
        fail_vals = X_test.iloc[fail_idx][feat].dropna() if len(fail_idx) > 0 else pd.Series()

        try:
            all_vals.plot.kde(ax=ax, color="#185FA5", linewidth=2, label="All test")
        except Exception:
            ax.hist(all_vals, bins=30, density=True, color="#185FA5", alpha=0.5, label="All")
        if len(fail_vals) >= 2:
            try:
                fail_vals.plot.kde(ax=ax, color="#E24B4A", linewidth=2, label="Failures")
            except Exception:
                ax.hist(fail_vals, bins=10, density=True, color="#E24B4A", alpha=0.7, label="Failures")

        ax.set_title(feat, fontsize=9, fontweight="600")
        ax.set_xlabel("Value"); ax.legend(fontsize=8); sns.despine(ax=ax)

    fig.suptitle("Failure Density vs All Test Samples\n"
                 "(deviation shows where failures cluster in feature space)",
                 fontsize=12, fontweight="700")
    plt.tight_layout()

    out = plots_dir / "failure_density.png"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out.name}")
    return out


def run_cluster_visualization(
    archetype_result: ArchetypeResult,
    fcs             : FailureCaseSet,
    severity_df     : pd.DataFrame,
    X_test          : pd.DataFrame,
    feature_names   : list[str],
    plots_dir       : str | Path = "outputs/plots/blindspot/clusters",
) -> dict:
    """Run all cluster visualizations and return paths."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    pca_p  = plot_pca_cluster_map(archetype_result, fcs, plots_dir)
    heat_p = plot_confidence_heatmap(archetype_result, severity_df, plots_dir)
    dens_p = plot_failure_density(fcs, X_test, feature_names, plots_dir)

    # UMAP (optional)
    umap_p = None
    if len(archetype_result.X_pca) >= 5:
        from sklearn.preprocessing import StandardScaler
        X_fail = X_test.iloc[archetype_result.sample_indices][feature_names].fillna(0).values
        umap_p = plot_umap_cluster_map(X_fail, archetype_result.labels, fcs, plots_dir)

    return {
        "pca_plot"       : pca_p,
        "heatmap_plot"   : heat_p,
        "density_plot"   : dens_p,
        "umap_plot"      : umap_p,
    }
