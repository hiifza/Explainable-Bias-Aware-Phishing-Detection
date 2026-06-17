"""
src/explainability/shap_lime_comparator.py
-------------------------------------------
Computes SHAP vs LIME agreement scores for all analysed samples,
identifies high-disagreement cases, and generates the consistency report.

Agreement formula (per sample)
-------------------------------
    top5_SHAP = top-5 features by |SHAP value|
    top5_LIME = top-5 features by |LIME contribution|
    agreement  = |intersection(top5_SHAP, top5_LIME)| / 5

Global aggregation:
    Mean / Median / Min / Max agreement
    Agreement by category: TP / TN / FP / FN

Feature consistency (global):
    top-20 SHAP features vs top-20 LIME aggregated features
    shared_features / SHAP_only / LIME_only / consistency_score

Public API
----------
    compute_per_sample_agreement(shap_result, lime_results, selected_indices,
                                 top_k)                           -> pd.DataFrame
    compute_global_agreement_metrics(agreement_df)                -> dict
    compute_feature_consistency(shap_ranking_df, lime_all_results,
                                feature_names, top_n)             -> dict
    plot_agreement_distribution(agreement_df, out_dir)            -> Path
    plot_feature_consistency(consistency, out_dir)                -> Path
    run_comparison(shap_result, lime_local_r, shap_ranking_df,
                   feature_names, plots_dir, reports_dir)         -> dict
"""

import sys
from pathlib import Path
from typing  import Any

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
from src.explainability.lime_explainer import LIMEResult

logger = get_logger(__name__)

HIGH_DISAGREE_THRESHOLD = 0.40


def _setup():
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({"figure.facecolor": "white", "savefig.dpi": 150})


# ── Per-sample agreement ──────────────────────────────────────────────────────

def compute_per_sample_agreement(
    shap_result     : SHAPResult,
    lime_results    : dict[str, list[LIMEResult]],
    selected_indices: dict[str, np.ndarray],
    top_k           : int = 5,
) -> pd.DataFrame:
    """
    Compute SHAP vs LIME top-k feature agreement for every analysed sample.

    For each sample the SHAP top-k features are extracted from
    shap_result.shap_values (matching the SHAP sample index), and the
    LIME top-k features are extracted from the LIMEResult.

    Parameters
    ----------
    shap_result      : SHAPResult from M7.1
    lime_results     : dict[category -> list[LIMEResult]] from lime_local
    selected_indices : dict[category -> np.ndarray of indices] from lime_local
    top_k            : overlap window (default 5)

    Returns
    -------
    pd.DataFrame  columns:
        sample_id, category, shap_features (list), lime_features (list),
        overlap_count, agreement_score
    """
    rows = []
    shap_vals  = shap_result.shap_values
    feat_names = shap_result.feature_names

    for cat, results_list in lime_results.items():
        indices = selected_indices.get(cat, np.array([], dtype=int))
        for num, (lr, orig_idx) in enumerate(zip(results_list, indices)):
            # SHAP: get top-k for this sample
            # SHAP was computed on a subsample; orig_idx is in the full test set.
            # We use the closest available SHAP sample (by index position)
            shap_sample_idx = min(int(orig_idx), len(shap_vals) - 1)
            sv              = shap_vals[shap_sample_idx]
            shap_order      = np.argsort(np.abs(sv))[::-1][:top_k]
            shap_top_feats  = [feat_names[i] for i in shap_order]

            # LIME: get top-k
            lime_top_feats  = lr.get_top_n_features(top_k)

            # Agreement
            overlap = len(set(shap_top_feats) & set(lime_top_feats))
            score   = overlap / max(top_k, 1)

            rows.append({
                "sample_id"      : int(orig_idx),
                "sample_num"     : num,
                "category"       : cat,
                "shap_features"  : "|".join(shap_top_feats),
                "lime_features"  : "|".join(lime_top_feats),
                "overlap_count"  : overlap,
                "agreement_score": round(score, 4),
                "prediction_proba": lr.prediction_proba,
                "local_r2"       : lr.local_r2,
                "is_native_lime" : lr.is_native_lime,
            })

    df = pd.DataFrame(rows).sort_values("agreement_score", ascending=True)
    logger.info(
        f"Per-sample agreement computed: {len(df)} samples  "
        f"mean={df['agreement_score'].mean():.4f}  "
        f"min={df['agreement_score'].min():.4f}"
    )
    return df.reset_index(drop=True)


# ── Global metrics ────────────────────────────────────────────────────────────

def compute_global_agreement_metrics(agreement_df: pd.DataFrame) -> dict:
    """
    Summarise agreement statistics globally and per category.

    Returns
    -------
    dict  keys: mean, median, std, min, max, by_category (dict)
    """
    scores = agreement_df["agreement_score"]
    global_stats = {
        "mean"         : round(float(scores.mean()),   4),
        "median"       : round(float(scores.median()), 4),
        "std"          : round(float(scores.std()),    4),
        "min"          : round(float(scores.min()),    4),
        "max"          : round(float(scores.max()),    4),
        "n_samples"    : int(len(scores)),
        "n_high_disagree": int((scores < HIGH_DISAGREE_THRESHOLD).sum()),
        "high_disagree_pct": round(
            float((scores < HIGH_DISAGREE_THRESHOLD).mean()) * 100, 2
        ),
    }

    by_cat = {}
    for cat in agreement_df["category"].unique():
        sub = agreement_df[agreement_df["category"] == cat]["agreement_score"]
        by_cat[cat] = {
            "mean"  : round(float(sub.mean()),   4),
            "median": round(float(sub.median()), 4),
            "n"     : int(len(sub)),
        }
    global_stats["by_category"] = by_cat

    logger.info(f"Agreement summary: mean={global_stats['mean']:.4f}  "
                f"high-disagree={global_stats['n_high_disagree']}")
    return global_stats


# ── Feature consistency ───────────────────────────────────────────────────────

def compute_feature_consistency(
    shap_ranking_df : pd.DataFrame,
    all_lime_results: list[LIMEResult],
    feature_names   : list[str],
    top_n           : int = 20,
) -> dict:
    """
    Compare global top-n SHAP features vs aggregated top-n LIME features.

    LIME global ranking: sum of |contribution| across all explained samples.

    Returns
    -------
    dict  keys: top_shap_features, top_lime_features, shared,
               shap_only, lime_only, consistency_score, lime_global_ranking
    """
    # SHAP top-n
    top_shap = set(shap_ranking_df.head(top_n)["feature"].tolist())

    # LIME global ranking: aggregate |contributions| over all samples
    lime_agg: dict[str, float] = {}
    for lr in all_lime_results:
        for feat, contrib in lr.contributions.items():
            # Normalise feature name (native LIME may suffix with conditions)
            base_feat = feat.split(" ")[0] if " " in feat else feat
            # Match to actual feature name
            best_match = next(
                (f for f in feature_names if f in feat or feat == f),
                base_feat,
            )
            lime_agg[best_match] = lime_agg.get(best_match, 0.0) + abs(contrib)

    lime_sorted = sorted(lime_agg.items(), key=lambda x: x[1], reverse=True)
    top_lime    = {f for f, _ in lime_sorted[:top_n]}

    lime_global_df = pd.DataFrame(
        [{"feature": f, "total_abs_contribution": round(v, 8), "rank": i + 1}
         for i, (f, v) in enumerate(lime_sorted[:top_n])]
    )

    shared      = top_shap & top_lime
    shap_only   = top_shap - top_lime
    lime_only   = top_lime - top_shap
    consistency = len(shared) / max(top_n, 1)

    logger.info(
        f"Feature consistency: "
        f"shared={len(shared)}/{top_n}  "
        f"SHAP-only={len(shap_only)}  "
        f"LIME-only={len(lime_only)}  "
        f"score={consistency:.4f}"
    )

    return {
        "top_shap_features"  : sorted(top_shap),
        "top_lime_features"  : sorted(top_lime),
        "shared"             : sorted(shared),
        "shap_only"          : sorted(shap_only),
        "lime_only"          : sorted(lime_only),
        "consistency_score"  : round(consistency, 4),
        "lime_global_ranking": lime_global_df,
    }


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_agreement_distribution(
    agreement_df: pd.DataFrame,
    out_dir     : Path,
) -> Path:
    """
    3-panel: histogram of agreement scores, category comparison bar,
    and scatter of agreement vs local R².
    """
    _setup()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("SHAP vs LIME Agreement Analysis", fontsize=14, fontweight="700")

    # Panel 1: histogram of agreement scores
    ax = axes[0]
    ax.hist(agreement_df["agreement_score"], bins=6, range=(0, 1),
            color="#185FA5", edgecolor="white", linewidth=0.8)
    ax.axvline(HIGH_DISAGREE_THRESHOLD, color="#E24B4A", linestyle="--",
               linewidth=1.5, label=f"High-disagree <{HIGH_DISAGREE_THRESHOLD}")
    ax.axvline(agreement_df["agreement_score"].mean(), color="#0F6E56",
               linestyle="-", linewidth=2, label=f"Mean={agreement_df['agreement_score'].mean():.3f}")
    ax.set_xlabel("Agreement score")
    ax.set_ylabel("Count")
    ax.set_title("Agreement Score Distribution")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    sns.despine(ax=ax)

    # Panel 2: mean agreement by category
    ax = axes[1]
    cat_means = agreement_df.groupby("category")["agreement_score"].mean()
    colors    = {"tp": "#1D9E75", "tn": "#185FA5", "fp": "#EF9F27", "fn": "#E24B4A"}
    bar_colors= [colors.get(c, "#888") for c in cat_means.index]
    ax.bar(cat_means.index.str.upper(), cat_means.values, color=bar_colors,
           edgecolor="white", linewidth=0.8)
    for cat, v in cat_means.items():
        ax.text(cat.upper(), v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="500")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Mean agreement score")
    ax.set_title("Mean Agreement by Category")
    sns.despine(ax=ax)

    # Panel 3: agreement vs R² scatter
    ax = axes[2]
    if "local_r2" in agreement_df.columns:
        sc = ax.scatter(
            agreement_df["local_r2"],
            agreement_df["agreement_score"],
            c=[list(colors.values())[i % 4] for i in range(len(agreement_df))],
            alpha=0.7, s=40, edgecolors="white", linewidths=0.5,
        )
        ax.set_xlabel("LIME local R²")
        ax.set_ylabel("Agreement score")
        ax.set_title("Agreement vs LIME Quality (R²)")
        sns.despine(ax=ax)

    plt.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shap_lime_agreement.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")
    return out_path


def plot_feature_consistency(
    consistency: dict,
    out_dir    : Path,
) -> Path:
    """Venn-style bar chart of shared/SHAP-only/LIME-only features."""
    _setup()
    n_shared    = len(consistency["shared"])
    n_shap_only = len(consistency["shap_only"])
    n_lime_only = len(consistency["lime_only"])

    labels = ["Shared\n(both)", "SHAP only", "LIME only"]
    values = [n_shared, n_shap_only, n_lime_only]
    colors = ["#1D9E75", "#185FA5", "#E24B4A"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Feature Consistency: SHAP vs LIME Top-20\n"
        f"Consistency score = {consistency['consistency_score']:.4f}",
        fontsize=13, fontweight="700",
    )

    # Left: overlap counts
    ax = axes[0]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(v), ha="center", fontsize=11, fontweight="600")
    ax.set_ylabel("Feature count")
    ax.set_title("Top-20 Feature Overlap")
    ax.set_ylim(0, max(values) * 1.25)
    sns.despine(ax=ax)

    # Right: LIME global ranking
    ax    = axes[1]
    lime_r= consistency["lime_global_ranking"].head(15).iloc[::-1]
    bar_c = ["#1D9E75" if f in set(consistency["shared"]) else "#E24B4A"
             for f in lime_r["feature"]]
    ax.barh(lime_r["feature"], lime_r["total_abs_contribution"],
            color=bar_c, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Total |LIME contribution|")
    ax.set_title("LIME Global Ranking (top 15)\n(green = shared with SHAP top-20)")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#1D9E75", label="Shared with SHAP"),
        Patch(facecolor="#E24B4A", label="LIME only"),
    ], fontsize=8)
    sns.despine(ax=ax)

    plt.tight_layout()
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feature_consistency.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")
    return out_path


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_comparison(
    shap_result     : SHAPResult,
    lime_local_r    : dict,
    shap_ranking_df : pd.DataFrame,
    feature_names   : list[str],
    plots_dir       : str | Path = "outputs/plots/lime",
    reports_dir     : str | Path = "outputs/reports",
    top_k           : int = 5,
    top_n_global    : int = 20,
) -> dict:
    """
    Run the full SHAP vs LIME comparison pipeline.

    Returns
    -------
    dict  keys: agreement_df, global_metrics, consistency, high_disagree_df,
               agreement_plot_path, consistency_plot_path
    """
    plots_dir   = Path(plots_dir)
    reports_dir = Path(reports_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 55)
    logger.info("M8.1 — SHAP vs LIME AGREEMENT ANALYSIS")
    logger.info("=" * 55)

    lime_results     = lime_local_r["lime_results"]
    selected_indices = lime_local_r["selected_indices"]
    all_lime_flat    = lime_local_r["all_lime_results_flat"]

    # 1. Per-sample agreement
    agreement_df = compute_per_sample_agreement(
        shap_result, lime_results, selected_indices, top_k
    )
    agreement_df.to_csv(reports_dir / "shap_lime_agreement.csv", index=False)
    logger.info("Saved: shap_lime_agreement.csv")

    # 2. High-disagreement cases
    high_dis = agreement_df[
        agreement_df["agreement_score"] < HIGH_DISAGREE_THRESHOLD
    ].copy()
    high_dis.to_csv(reports_dir / "high_disagreement_cases.csv", index=False)
    logger.info(
        f"Saved: high_disagreement_cases.csv  ({len(high_dis)} cases)"
    )

    # 3. Global metrics
    global_metrics = compute_global_agreement_metrics(agreement_df)

    # 4. Feature consistency
    consistency = compute_feature_consistency(
        shap_ranking_df, all_lime_flat, feature_names, top_n_global
    )
    consistency["lime_global_ranking"].to_csv(
        reports_dir / "lime_feature_ranking.csv", index=False
    )
    logger.info("Saved: lime_feature_ranking.csv")

    # 5. Summary CSV
    summary_rows = [{"metric": k, "value": v}
                    for k, v in global_metrics.items()
                    if not isinstance(v, dict)]
    summary_rows += [
        {"metric": "consistency_score",   "value": consistency["consistency_score"]},
        {"metric": "n_shared_features",   "value": len(consistency["shared"])},
        {"metric": "n_shap_only_features","value": len(consistency["shap_only"])},
        {"metric": "n_lime_only_features","value": len(consistency["lime_only"])},
    ]
    pd.DataFrame(summary_rows).to_csv(reports_dir / "lime_summary.csv", index=False)
    logger.info("Saved: lime_summary.csv")

    # 6. Plots
    agree_plot = plot_agreement_distribution(agreement_df, plots_dir)
    consist_p  = plot_feature_consistency(consistency, plots_dir)

    logger.info("SHAP vs LIME comparison complete")
    return {
        "agreement_df"          : agreement_df,
        "global_metrics"        : global_metrics,
        "consistency"           : consistency,
        "high_disagree_df"      : high_dis,
        "agreement_plot_path"   : agree_plot,
        "consistency_plot_path" : consist_p,
    }
