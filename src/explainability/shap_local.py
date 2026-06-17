"""
src/explainability/shap_local.py
----------------------------------
Local SHAP explainability: per-sample waterfall and force plots for
automatically selected True Positives, True Negatives, False Positives,
and False Negatives.

Public API
----------
    select_samples(result, y_true, n_per_class)     -> dict[str, np.ndarray]
    plot_waterfall(result, sample_idx, label, out_dir, top_k) -> Path
    plot_force(result, sample_idx, label, out_dir)            -> Path
    save_local_explanation_json(result, sample_idx, label, out_dir) -> Path
    run_local_analysis(result, y_true, n_per_class, plots_dir,
                       reports_dir)                 -> dict
"""

import json
import sys
from pathlib import Path
from typing  import Optional

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

CATEGORY_COLORS = {
    "tp": "#1D9E75",   # green  — correctly identified legitimate
    "tn": "#185FA5",   # blue   — correctly identified phishing
    "fp": "#EF9F27",   # orange — legitimate flagged as phishing
    "fn": "#E24B4A",   # red    — phishing missed as legitimate
}
CATEGORY_LABELS = {
    "tp": "True Positive (correct: Legitimate)",
    "tn": "True Negative (correct: Phishing)",
    "fp": "False Positive (error: Legitimate → Phishing)",
    "fn": "False Negative (error: Phishing → Legitimate)",
}


def _setup():
    sns.set_theme(style="white", font_scale=1.0)
    plt.rcParams.update({"figure.facecolor": "white", "savefig.dpi": 150})


# ── Sample selection ──────────────────────────────────────────────────────────

def select_samples(
    result      : SHAPResult,
    y_true      : np.ndarray | pd.Series,
    n_per_class : int = 10,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """
    Identify indices within result.X_explained for TP, TN, FP, FN samples.

    Parameters
    ----------
    result       : SHAPResult (shap_values already computed on X_explained)
    y_true       : true labels aligned with result.X_explained
    n_per_class  : number of samples per category
    random_state : reproducibility seed

    Returns
    -------
    dict  keys: 'tp', 'tn', 'fp', 'fn'  → np.ndarray of indices
    """
    y_true = np.asarray(y_true)
    y_pred = result.y_pred

    rng     = np.random.default_rng(random_state)
    cats    = {
        "tp": np.where((y_pred == 1) & (y_true == 1))[0],
        "tn": np.where((y_pred == 0) & (y_true == 0))[0],
        "fp": np.where((y_pred == 0) & (y_true == 1))[0],
        "fn": np.where((y_pred == 1) & (y_true == 0))[0],
    }

    selected = {}
    for cat, indices in cats.items():
        n       = min(n_per_class, len(indices))
        chosen  = rng.choice(indices, size=n, replace=False) if n > 0 else np.array([], dtype=int)
        selected[cat] = chosen
        logger.info(f"  {cat.upper()}: {len(indices):,} available → {n} selected")

    total = sum(len(v) for v in selected.values())
    logger.info(f"Local analysis: {total} samples selected (TP/TN/FP/FN)")
    return selected


# ── Waterfall plot ────────────────────────────────────────────────────────────

def plot_waterfall(
    result    : SHAPResult,
    sample_idx: int,
    category  : str,
    out_dir   : Path,
    sample_num: int  = 0,
    top_k     : int  = 15,
) -> Path:
    """
    Waterfall plot for a single sample: shows how each feature contribution
    pushes the prediction from base_value to the final output.

    Parameters
    ----------
    result     : SHAPResult
    sample_idx : row index within result.shap_values
    category   : 'tp', 'tn', 'fp', 'fn' (for naming / colour)
    out_dir    : destination directory
    sample_num : sequential number for filename
    top_k      : features to display (rest collapsed into "other features")
    """
    _setup()
    sv         = result.shap_values[sample_idx]
    fnames     = result.feature_names
    fvals      = result.X_explained.iloc[sample_idx]
    base_val   = result.expected_value
    pred_val   = float(result.y_proba[sample_idx])
    color      = CATEGORY_COLORS.get(category, "#185FA5")

    # Sort by |SHAP| descending, keep top_k
    order      = np.argsort(np.abs(sv))[::-1]
    top_order  = order[:top_k]
    rest_sv    = sv[order[top_k:]].sum()
    has_rest   = (len(order) > top_k) and abs(rest_sv) > 1e-9

    labels  = [fnames[i] for i in top_order]
    values  = [sv[i]     for i in top_order]
    f_vals  = [fvals.iloc[i] for i in top_order]

    if has_rest:
        labels.append(f"other {len(order)-top_k} features")
        values.append(rest_sv)
        f_vals.append(np.nan)

    # Reverse for bottom-to-top waterfall
    labels = labels[::-1]
    values = values[::-1]
    f_vals = f_vals[::-1]

    n      = len(labels)
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.38)))

    bar_colors = ["#E24B4A" if v < 0 else "#1D9E75" for v in values]
    bars = ax.barh(range(n), values, color=bar_colors,
                   edgecolor="white", linewidth=0.5, height=0.65)

    # Add feature = value annotations
    for i, (bar, fv) in enumerate(zip(bars, f_vals)):
        fv_str = f" = {fv:.2f}" if not np.isnan(float(fv if fv is not None else np.nan)) else ""
        xoff   = 0.0005 if bar.get_width() >= 0 else -0.0005
        ha     = "left"  if bar.get_width() >= 0 else "right"
        ax.text(bar.get_width() + xoff, i,
                f"{bar.get_width():+.4f}{fv_str}",
                va="center", ha=ha, fontsize=8)

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="#888", linewidth=0.8, linestyle="--")

    cat_label   = CATEGORY_LABELS.get(category, category.upper())
    ax.set_title(
        f"SHAP Waterfall — {cat_label}\n"
        f"Base={base_val:.4f}  →  Prediction={pred_val:.4f}  "
        f"({'Legitimate' if result.y_pred[sample_idx]==1 else 'Phishing'})",
        fontsize=11, fontweight="700",
    )
    ax.set_xlabel("SHAP value contribution")
    sns.despine(ax=ax)
    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname    = out_dir / f"{category}_{sample_num:02d}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fname


# ── Force plot ────────────────────────────────────────────────────────────────

def plot_force(
    result    : SHAPResult,
    sample_idx: int,
    category  : str,
    out_dir   : Path,
    sample_num: int = 0,
    top_k     : int = 10,
) -> Path:
    """
    Force plot: horizontal stacked bar showing which features push
    prediction toward Legitimate (right) or Phishing (left).
    """
    _setup()
    sv        = result.shap_values[sample_idx]
    fnames    = result.feature_names
    base_val  = result.expected_value
    pred_val  = float(result.y_proba[sample_idx])

    # Sort: positives (→ Legitimate) and negatives (→ Phishing)
    order     = np.argsort(np.abs(sv))[::-1][:top_k]
    sv_top    = sv[order]
    fn_top    = [fnames[i] for i in order]

    pos_mask  = sv_top > 0
    neg_mask  = sv_top < 0

    fig, ax   = plt.subplots(figsize=(12, 3.5))

    # Stacked positive bars from base_val rightward
    x_pos = base_val
    for val, name in zip(sv_top[pos_mask], np.array(fn_top)[pos_mask]):
        ax.barh(0, val, left=x_pos, height=0.5,
                color="#1D9E75", edgecolor="white", linewidth=0.5)
        if abs(val) > 0.005:
            ax.text(x_pos + val / 2, 0, name,
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="600", clip_on=True)
        x_pos += val

    # Stacked negative bars from base_val leftward
    x_neg = base_val
    for val, name in zip(sv_top[neg_mask], np.array(fn_top)[neg_mask]):
        ax.barh(0, val, left=x_neg, height=0.5,
                color="#E24B4A", edgecolor="white", linewidth=0.5)
        if abs(val) > 0.005:
            ax.text(x_neg + val / 2, 0, name,
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="600", clip_on=True)
        x_neg += val

    ax.axvline(base_val, color="#888", linewidth=1.5, linestyle="--",
               label=f"Base={base_val:.4f}")
    ax.axvline(pred_val, color="#333", linewidth=2.0,
               label=f"Pred={pred_val:.4f}")

    cat_label = CATEGORY_LABELS.get(category, category.upper())
    ax.set_title(f"SHAP Force Plot — {cat_label}", fontsize=11, fontweight="700")
    ax.set_xlabel("P(Legitimate)")
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    ax.text(0.02, -0.35, "← Phishing", transform=ax.transAxes,
            fontsize=9, color="#E24B4A")
    ax.text(0.75, -0.35, "Legitimate →", transform=ax.transAxes,
            fontsize=9, color="#1D9E75")
    sns.despine(ax=ax, left=True)
    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname   = out_dir / f"{category}_{sample_num:02d}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fname


# ── Local explanation JSON ────────────────────────────────────────────────────

def save_local_explanation_json(
    result    : SHAPResult,
    sample_idx: int,
    category  : str,
    out_dir   : Path,
    sample_num: int = 0,
    y_true_val: Optional[int] = None,
) -> Path:
    """
    Save a per-sample explanation as a JSON report.
    """
    sv       = result.shap_values[sample_idx]
    fnames   = result.feature_names
    fvals    = result.X_explained.iloc[sample_idx]
    order    = np.argsort(np.abs(sv))[::-1]

    top_contributions = [
        {
            "rank"       : int(rank + 1),
            "feature"    : str(fnames[i]),
            "shap_value" : round(float(sv[i]), 8),
            "feature_val": round(float(fvals.iloc[i]), 6),
            "direction"  : "→ Legitimate" if sv[i] > 0 else "→ Phishing",
        }
        for rank, i in enumerate(order[:20])
    ]

    record = {
        "sample_index"     : int(sample_idx),
        "category"         : category,
        "category_label"   : CATEGORY_LABELS.get(category, category),
        "sample_number"    : int(sample_num),
        "prediction_proba" : round(float(result.y_proba[sample_idx]), 6),
        "prediction_class" : int(result.y_pred[sample_idx]),
        "true_label"       : int(y_true_val) if y_true_val is not None else None,
        "base_value"       : round(float(result.expected_value), 6),
        "top_contributions": top_contributions,
        "model_class"      : result.model_class,
        "is_native_shap"   : result.is_native_shap,
    }

    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{category}_{sample_num:02d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return out_path


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_local_analysis(
    result       : SHAPResult,
    y_true       : np.ndarray | pd.Series,
    n_per_class  : int = 10,
    plots_dir    : str | Path = "outputs/plots/shap",
    reports_dir  : str | Path = "outputs/reports/shap_local_explanations",
) -> dict:
    """
    Full local analysis: select TP/TN/FP/FN, generate waterfall, force,
    and JSON reports for each.

    Returns
    -------
    dict  keys: selected_indices, waterfall_paths, force_paths,
               json_paths, category_counts
    """
    plots_dir   = Path(plots_dir)
    reports_dir = Path(reports_dir)

    logger.info("=" * 55)
    logger.info("M7.1 — LOCAL SHAP ANALYSIS")
    logger.info("=" * 55)

    y_true_arr = np.asarray(y_true)[: result.n_samples]
    selected   = select_samples(result, y_true_arr, n_per_class)

    waterfall_paths : dict[str, list[Path]] = {}
    force_paths     : dict[str, list[Path]] = {}
    json_paths      : dict[str, list[Path]] = {}

    wf_dir  = plots_dir / "waterfall"
    fo_dir  = plots_dir / "force"
    wf_dir.mkdir(parents=True, exist_ok=True)
    fo_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for cat, indices in selected.items():
        waterfall_paths[cat] = []
        force_paths[cat]     = []
        json_paths[cat]      = []

        for num, idx in enumerate(indices):
            yt = int(y_true_arr[idx]) if idx < len(y_true_arr) else None

            wf = plot_waterfall(result, idx, cat, wf_dir, sample_num=num)
            waterfall_paths[cat].append(wf)

            fo = plot_force(result, idx, cat, fo_dir, sample_num=num)
            force_paths[cat].append(fo)

            jp = save_local_explanation_json(
                result, idx, cat, reports_dir, sample_num=num, y_true_val=yt
            )
            json_paths[cat].append(jp)

        logger.info(
            f"  {cat.upper()}: {len(indices)} samples → "
            f"{len(waterfall_paths[cat])} waterfall + force + JSON"
        )

    total = sum(len(v) for v in waterfall_paths.values())
    logger.info(f"Local analysis complete: {total} waterfall + {total} force plots")

    return {
        "selected_indices": selected,
        "waterfall_paths" : waterfall_paths,
        "force_paths"     : force_paths,
        "json_paths"      : json_paths,
        "category_counts" : {c: int(len(v)) for c, v in selected.items()},
    }
