"""
src/explainability/lime_local.py
----------------------------------
Generates LIME local explanations for automatically selected
True Positive, True Negative, False Positive, and False Negative samples.

For each selected sample:
  - Feature contribution bar chart (saved to outputs/plots/lime/<cat>/)
  - JSON explanation report (saved to outputs/reports/lime_local_explanations/)

Public API
----------
    select_samples(model, X_test, y_test, n_per_class, random_state) -> dict
    plot_lime_contribution(lime_result, category, out_dir, sample_num) -> Path
    save_lime_explanation_json(lime_result, category, out_dir, sample_num, y_true) -> Path
    run_lime_local(explainer, predict_fn, X_test, y_test, feature_names,
                   n_per_class, plots_dir, reports_dir)               -> dict
"""

import json
import sys
from pathlib import Path
from typing  import Any, Callable, Optional

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
from src.explainability.lime_explainer import LIMEResult, explain_sample

logger = get_logger(__name__)

CATEGORY_COLORS = {
    "tp": "#1D9E75", "tn": "#185FA5",
    "fp": "#EF9F27", "fn": "#E24B4A",
}
CATEGORY_LABELS = {
    "tp": "True Positive  (correct: Legitimate)",
    "tn": "True Negative  (correct: Phishing)",
    "fp": "False Positive (error: Legitimate → Phishing)",
    "fn": "False Negative (error: Phishing → Legitimate)",
}


def _setup():
    sns.set_theme(style="white", font_scale=1.0)
    plt.rcParams.update({"figure.facecolor": "white", "savefig.dpi": 150})


# ── Sample selection ──────────────────────────────────────────────────────────

def select_samples(
    model       : Any,
    X_test      : np.ndarray,
    y_test      : np.ndarray,
    n_per_class : int = 10,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """
    Identify row indices in X_test for TP, TN, FP, FN samples.

    Parameters
    ----------
    model        : fitted estimator
    X_test       : test feature matrix (numpy)
    y_test       : true labels (numpy)
    n_per_class  : samples per category
    random_state : seed

    Returns
    -------
    dict  {'tp': idx_array, 'tn': ..., 'fp': ..., 'fn': ...}
    """
    y_pred = model.predict(X_test)
    rng    = np.random.default_rng(random_state)

    cats = {
        "tp": np.where((y_pred == 1) & (y_test == 1))[0],
        "tn": np.where((y_pred == 0) & (y_test == 0))[0],
        "fp": np.where((y_pred == 0) & (y_test == 1))[0],
        "fn": np.where((y_pred == 1) & (y_test == 0))[0],
    }
    selected = {}
    for cat, indices in cats.items():
        n      = min(n_per_class, len(indices))
        chosen = rng.choice(indices, size=n, replace=False) \
            if n > 0 else np.array([], dtype=int)
        selected[cat] = chosen
        logger.info(f"  LIME {cat.upper()}: {len(indices):,} available → {n} selected")

    return selected


# ── Feature contribution plot ─────────────────────────────────────────────────

def plot_lime_contribution(
    lime_result : LIMEResult,
    category    : str,
    out_dir     : Path,
    sample_num  : int = 0,
    top_k       : int = 10,
) -> Path:
    """
    Horizontal bar chart of LIME feature contributions for one sample.

    Positive contributions → push toward Legitimate.
    Negative contributions → push toward Phishing.
    """
    _setup()
    color       = CATEGORY_COLORS.get(category, "#185FA5")
    cat_label   = CATEGORY_LABELS.get(category, category.upper())
    top_feats   = lime_result.top_features[:top_k]

    if not top_feats:
        logger.warning(f"No LIME contributions for sample {sample_num}")
        return Path(out_dir) / f"{category}_{sample_num:02d}.png"

    features = [f for f, _ in top_feats]
    values   = [v for _, v in top_feats]
    # Reverse for bottom-to-top display
    features = features[::-1]
    values   = values[::-1]

    bar_colors = ["#1D9E75" if v >= 0 else "#E24B4A" for v in values]

    fig, ax = plt.subplots(figsize=(9, max(4, len(features) * 0.42)))
    bars    = ax.barh(features, values, color=bar_colors,
                      edgecolor="white", linewidth=0.6, height=0.65)
    for bar, v in zip(bars, values):
        xoff = 0.0005 if v >= 0 else -0.0005
        ha   = "left" if v >= 0 else "right"
        ax.text(bar.get_width() + xoff, bar.get_y() + bar.get_height() / 2,
                f"{v:+.4f}", va="center", ha=ha, fontsize=8)

    ax.axvline(0, color="#888", linewidth=0.8, linestyle="--")
    ax.set_title(
        f"LIME Explanation — {cat_label}\n"
        f"P(Legitimate)={lime_result.prediction_proba:.4f}  "
        f"R²={lime_result.local_r2:.4f}  "
        f"({'native' if lime_result.is_native_lime else 'fallback'})",
        fontsize=10, fontweight="700",
    )
    ax.set_xlabel("LIME contribution\n(positive → Legitimate | negative → Phishing)")
    ax.text(0.01, -0.11,
            "← Phishing                    Legitimate →",
            transform=ax.transAxes, fontsize=8, color="#888")
    sns.despine(ax=ax)
    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{category}_{sample_num:02d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ── JSON explanation report ───────────────────────────────────────────────────

def save_lime_explanation_json(
    lime_result : LIMEResult,
    category    : str,
    out_dir     : Path,
    sample_num  : int = 0,
    y_true_val  : Optional[int] = None,
) -> Path:
    """Save a per-sample LIME explanation as JSON."""
    record = {
        "sample_index"     : lime_result.sample_idx,
        "sample_number"    : sample_num,
        "category"         : category,
        "category_label"   : CATEGORY_LABELS.get(category, category),
        "prediction_proba" : lime_result.prediction_proba,
        "prediction_class" : lime_result.prediction_class,
        "true_label"       : int(y_true_val) if y_true_val is not None else None,
        "local_r2"         : lime_result.local_r2,
        "intercept"        : lime_result.intercept,
        "is_native_lime"   : lime_result.is_native_lime,
        "top_contributions": [
            {
                "rank"      : i + 1,
                "feature"   : feat,
                "contribution": round(float(contrib), 8),
                "direction" : "→ Legitimate" if contrib > 0 else "→ Phishing",
            }
            for i, (feat, contrib) in enumerate(lime_result.top_features[:20])
        ],
    }
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{category}_{sample_num:02d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return out_path


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_lime_local(
    explainer    : Any,
    predict_fn   : Callable,
    X_test       : np.ndarray,
    y_test       : np.ndarray,
    feature_names: list[str],
    model        : Any,
    n_per_class  : int = 10,
    n_lime_feats : int = 10,
    n_lime_samp  : int = 2000,
    plots_dir    : str | Path = "outputs/plots/lime",
    reports_dir  : str | Path = "outputs/reports/lime_local_explanations",
    random_state : int = 42,
) -> dict:
    """
    Generate LIME explanations for all selected TP/TN/FP/FN samples.

    Returns
    -------
    dict  keys: selected_indices, lime_results, plot_paths, json_paths,
               category_counts, all_lime_results_flat
    """
    plots_dir   = Path(plots_dir)
    reports_dir = Path(reports_dir)

    logger.info("=" * 55)
    logger.info("M8.1 — LOCAL LIME ANALYSIS")
    logger.info("=" * 55)

    selected = select_samples(model, X_test, y_test, n_per_class, random_state)

    lime_results : dict[str, list[LIMEResult]] = {}
    plot_paths   : dict[str, list[Path]] = {}
    json_paths   : dict[str, list[Path]] = {}

    for cat, indices in selected.items():
        lime_results[cat] = []
        plot_paths[cat]   = []
        json_paths[cat]   = []

        cat_plot_dir  = plots_dir / cat
        cat_plot_dir.mkdir(parents=True, exist_ok=True)

        for num, idx in enumerate(indices):
            sample  = X_test[idx]
            y_true  = int(y_test[idx])

            lr = explain_sample(
                explainer, predict_fn, sample, feature_names,
                n_features=n_lime_feats, n_samples=n_lime_samp,
                sample_idx=int(idx),
            )
            lime_results[cat].append(lr)

            pp = plot_lime_contribution(lr, cat, cat_plot_dir, sample_num=num)
            plot_paths[cat].append(pp)

            jp = save_lime_explanation_json(
                lr, cat, reports_dir, sample_num=num, y_true_val=y_true
            )
            json_paths[cat].append(jp)

        logger.info(
            f"  {cat.upper()}: {len(indices)} samples → "
            f"{len(lime_results[cat])} explanations"
        )

    # Flatten all results
    all_flat = [r for results in lime_results.values() for r in results]

    total = sum(len(v) for v in plot_paths.values())
    logger.info(f"Local LIME complete: {total} explanations generated")

    return {
        "selected_indices"    : selected,
        "lime_results"        : lime_results,
        "plot_paths"          : plot_paths,
        "json_paths"          : json_paths,
        "category_counts"     : {c: int(len(v)) for c, v in lime_results.items()},
        "all_lime_results_flat": all_flat,
    }
