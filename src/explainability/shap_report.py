"""
src/explainability/shap_report.py
-----------------------------------
HTML report generation and Track A vs Track B SHAP comparison for M7.1.

The track comparison quantifies how much URLSimilarityIndex dominates
the Track A SHAP rankings and how its absence reshuffles the importance
hierarchy in Track B.

Public API
----------
    run_track_comparison(model_A, model_B, X_train_A, X_train_B,
                         X_test_A, X_test_B, feature_names_A, feature_names_B)
                                                              -> dict
    generate_shap_report(global_r, local_r, interaction_r,
                         track_comp, shap_result, output_path) -> str
"""

import sys
from datetime import datetime
from pathlib  import Path
from typing   import Any, Optional

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
from src.explainability.shap_explainer import compute_shap_values, SHAPResult
from src.explainability.shap_global    import plot_importance_bar

logger = get_logger(__name__)


# ── Track comparison ──────────────────────────────────────────────────────────

def run_track_comparison(
    model_A      : Any,
    model_B      : Any,
    X_train_A    : pd.DataFrame,
    X_train_B    : pd.DataFrame,
    X_test_A     : pd.DataFrame,
    X_test_B     : pd.DataFrame,
    feature_names_A: list[str],
    feature_names_B: list[str],
    plots_dir    : str | Path = "outputs/plots/shap",
    sample_n     : int = 2000,
) -> dict:
    """
    Compute SHAP values for both track winners and compare importance ranks.

    Returns
    -------
    dict  keys:
        result_A, result_B        : SHAPResult objects
        ranking_A, ranking_B      : pd.DataFrame feature rankings
        importance_shift_df       : merged comparison
        usi_contribution_pct      : float — URLSimilarityIndex share of Track A
        comparison_plot_path      : Path
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Computing SHAP values for Track A …")
    result_A = compute_shap_values(
        model_A, X_train_A, X_test_A, feature_names_A, sample_n=sample_n
    )
    logger.info("Computing SHAP values for Track B …")
    result_B = compute_shap_values(
        model_B, X_train_B, X_test_B, feature_names_B, sample_n=sample_n
    )

    ranking_A = result_A.get_feature_ranking()
    ranking_B = result_B.get_feature_ranking()

    # URLSimilarityIndex contribution in Track A
    usi_row    = ranking_A[ranking_A["feature"] == "URLSimilarityIndex"]
    usi_pct    = float(usi_row["relative_importance"].values[0]) * 100 \
        if len(usi_row) > 0 else 0.0

    # Merge rankings — features common to both tracks (Track B feature set)
    common_feats = set(feature_names_B)
    r_A_common   = ranking_A[ranking_A["feature"].isin(common_feats)].copy()
    r_A_common   = r_A_common.rename(columns={
        "rank": "rank_A", "mean_abs_shap": "mean_shap_A",
        "relative_importance": "rel_imp_A",
    })

    r_B_common   = ranking_B.copy()
    r_B_common   = r_B_common.rename(columns={
        "rank": "rank_B", "mean_abs_shap": "mean_shap_B",
        "relative_importance": "rel_imp_B",
    })

    shift_df = pd.merge(r_B_common, r_A_common, on="feature", how="left")
    shift_df["rank_shift"] = shift_df["rank_A"].fillna(99) - shift_df["rank_B"]
    shift_df = shift_df.sort_values("rank_B").reset_index(drop=True)

    # Comparison bar chart: Track A (common) vs Track B importances
    _setup_plot()
    top_n   = 20
    top_b   = ranking_B.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    x       = np.arange(top_n)
    w       = 0.38

    b_imp = top_b["relative_importance"].values
    a_imp = []
    for feat in top_b["feature"]:
        row = r_A_common[r_A_common["feature"] == feat]
        a_imp.append(float(row["rel_imp_A"].values[0]) if len(row) > 0 else 0)

    ax.bar(x - w / 2, b_imp, width=w, label="Track B (leakage-aware)",
           color="#1D9E75", edgecolor="white")
    ax.bar(x + w / 2, a_imp, width=w, label="Track A (with URLSimilarityIndex)",
           color="#E24B4A", edgecolor="white", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(top_b["feature"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Relative importance (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax.set_title(
        f"Feature Importance Shift: Track B vs Track A\n"
        f"(URLSimilarityIndex = {usi_pct:.1f}% of Track A total — excluded from Track B)",
        fontsize=12, fontweight="700",
    )
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()

    comp_plot = plots_dir / "track_comparison_importance.png"
    fig.savefig(comp_plot, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {comp_plot.name}")
    logger.info(
        f"Track comparison: URLSimilarityIndex = {usi_pct:.2f}% of Track A importance"
    )

    return {
        "result_A"             : result_A,
        "result_B"             : result_B,
        "ranking_A"            : ranking_A,
        "ranking_B"            : ranking_B,
        "importance_shift_df"  : shift_df,
        "usi_contribution_pct" : usi_pct,
        "comparison_plot_path" : comp_plot,
    }


def _setup_plot():
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({"figure.facecolor": "white", "savefig.dpi": 150})


# ── HTML report CSS ───────────────────────────────────────────────────────────

_CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,sans-serif;font-size:14px;color:#1a1a1a;
     max-width:1120px;margin:40px auto;padding:0 26px 80px}
h1{font-size:22px;font-weight:700;border-bottom:3px solid #185FA5;
   padding-bottom:12px;margin-bottom:6px}
h2{font-size:16px;font-weight:600;color:#185FA5;margin:32px 0 10px;
   border-left:4px solid #185FA5;padding-left:10px}
h3{font-size:14px;font-weight:600;color:#333;margin:16px 0 8px}
p{line-height:1.65;color:#444;margin-bottom:10px}
p.meta{font-size:12px;color:#888;margin-bottom:22px}
table{width:100%;border-collapse:collapse;font-size:12.5px;margin:10px 0 20px}
th{background:#E6F1FB;padding:8px 12px;text-align:left;
   border:1px solid #B5D4F4;font-weight:600;white-space:nowrap}
td{padding:7px 12px;border:1px solid #ddd}
tr:nth-child(even) td{background:#F8FBFF}
code{font-family:monospace;font-size:12px;background:#F1F3F5;
     padding:2px 5px;border-radius:3px}
.card{border:1px solid #B5D4F4;border-radius:8px;padding:13px 17px;
      background:#F0F7FF;margin:8px 0}
.card-v{font-size:24px;font-weight:700;color:#185FA5}
.card-l{font-size:12px;color:#666;margin-top:3px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:12px 0 20px}
.alert{border-left:4px solid #E24B4A;background:#FCEBEB;
       padding:10px 14px;border-radius:0 6px 6px 0;margin:10px 0}
.info{border-left:4px solid #378ADD;background:#E6F1FB;
      padding:10px 14px;border-radius:0 6px 6px 0;margin:10px 0}
img{max-width:100%;border:1px solid #ddd;border-radius:6px;
    margin:6px 0;display:block}
.img2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0}
.footer{font-size:12px;color:#aaa;margin-top:56px;
        border-top:1px solid #eee;padding-top:16px}
"""


def _rel(path: Path) -> str:
    try:    return "../../" + str(path.relative_to(ROOT))
    except: return str(path)


def _img(path: Path, caption: str = "") -> str:
    if not path.exists():
        return ""
    r   = _rel(path)
    cap = (f"<figcaption style='font-size:11px;color:#888;margin-top:4px'>"
           f"{caption}</figcaption>") if caption else ""
    return f"<figure><img src='{r}' alt='{caption}'>{cap}</figure>"


# ── HTML section builders ─────────────────────────────────────────────────────

def _exec_summary(shap_result: SHAPResult, top_df: pd.DataFrame,
                  usi_pct: float) -> str:
    top5 = ", ".join(top_df.head(5)["feature"].tolist())
    return f"""
<h2>1. Executive Summary</h2>
<div class="grid3">
  <div class="card">
    <div class="card-v">{shap_result.n_features}</div>
    <div class="card-l">Features analysed</div>
  </div>
  <div class="card">
    <div class="card-v">{shap_result.n_samples:,}</div>
    <div class="card-l">Test samples explained</div>
  </div>
  <div class="card">
    <div class="card-v">{'Native' if shap_result.is_native_shap else 'Fallback'}</div>
    <div class="card-l">SHAP backend</div>
  </div>
</div>
<div class="info">
  <strong>Top-5 features (Track B):</strong> {top5}
</div>
<div class="{'alert' if usi_pct > 10 else 'info'}">
  <strong>URLSimilarityIndex</strong> contributed
  <strong>{usi_pct:.1f}%</strong> of total Track A feature importance.
  It is excluded from Track B — the deployment model.
</div>"""


def _top_features_section(top_df: pd.DataFrame, plots_dir: Path) -> str:
    rows = "".join(
        f"<tr><td>{r['rank']}</td>"
        f"<td><code>{r['feature']}</code></td>"
        f"<td>{r['mean_abs_shap']:.6f}</td>"
        f"<td>{r['relative_importance']*100:.2f}%</td></tr>"
        for _, r in top_df.head(20).iterrows()
    )
    return f"""
<h2>2. Top Features — Track B</h2>
<div class="img2">
  {_img(plots_dir / 'summary_beeswarm.png',  'SHAP beeswarm summary')}
  {_img(plots_dir / 'global_importance.png', 'Mean |SHAP| importance bar')}
</div>
<table>
  <tr><th>Rank</th><th>Feature</th><th>Mean |SHAP|</th><th>Relative importance</th></tr>
  {rows}
</table>"""


def _interaction_section(top_pairs: pd.DataFrame, plots_dir: Path) -> str:
    rows = "".join(
        f"<tr><td>{r['rank']}</td>"
        f"<td><code>{r['feature_A']}</code></td>"
        f"<td><code>{r['feature_B']}</code></td>"
        f"<td>{r['interaction_strength']:.6f}</td></tr>"
        for _, r in top_pairs.head(10).iterrows()
    )
    int_dir = plots_dir / "interactions"
    return f"""
<h2>3. Feature Interactions</h2>
<div class="img2">
  {_img(int_dir / 'interaction_heatmap.png',   'Interaction heatmap')}
  {_img(int_dir / 'top_interaction_pairs.png', 'Top interaction pairs')}
</div>
<table>
  <tr><th>Rank</th><th>Feature A</th><th>Feature B</th><th>Strength</th></tr>
  {rows}
</table>"""


def _track_comparison_section(track_comp: dict) -> str:
    shift_df  = track_comp.get("importance_shift_df", pd.DataFrame())
    usi_pct   = track_comp.get("usi_contribution_pct", 0)
    comp_plot = track_comp.get("comparison_plot_path")

    if shift_df.empty:
        return "<h2>4. Track Comparison</h2><p>No comparison data available.</p>"

    rows = "".join(
        f"<tr><td><code>{r['feature']}</code></td>"
        f"<td>{int(r['rank_B'])}</td>"
        f"<td>{int(r['rank_A']) if not pd.isna(r['rank_A']) else '—'}</td>"
        f"<td style='color:{'#E24B4A' if r['rank_shift']<-2 else '#0F6E56' if r['rank_shift']>2 else '#333'}'>"
        f"{r['rank_shift']:+.0f}</td></tr>"
        for _, r in shift_df.head(15).iterrows()
    )
    img = _img(comp_plot, "Importance shift Track A vs B") if comp_plot else ""
    return f"""
<h2>4. Track A vs Track B — Leakage Impact</h2>
{img}
<p><code>URLSimilarityIndex</code> dominates Track A with
<strong>{usi_pct:.1f}%</strong> of total importance.
Removing it (Track B) reshuffles all remaining features.</p>
<table>
  <tr><th>Feature</th><th>Rank B</th><th>Rank A</th><th>Shift</th></tr>
  {rows}
</table>"""


def _local_section(local_r: dict, plots_dir: Path) -> str:
    cats    = ["tp", "tn", "fp", "fn"]
    labels  = {"tp":"True Positive","tn":"True Negative",
                "fp":"False Positive","fn":"False Negative"}
    content = ""
    for cat in cats:
        count = local_r.get("category_counts", {}).get(cat, 0)
        wf_p  = local_r.get("waterfall_paths", {}).get(cat, [])
        fo_p  = local_r.get("force_paths", {}).get(cat, [])
        imgs  = ""
        for i, (wp, fp) in enumerate(zip(wf_p[:2], fo_p[:2])):
            imgs += f"<div class='img2'>{_img(wp,'Waterfall')}{_img(fp,'Force')}</div>"
        content += f"""
<h3>{labels[cat]} ({count} samples)</h3>
{imgs if imgs else '<p>No samples in this category.</p>'}"""

    return f"<h2>5. Local Explanations (sample selection)</h2>{content}"


# ── Main report function ──────────────────────────────────────────────────────

def generate_shap_report(
    global_r    : dict,
    local_r     : dict,
    interaction_r: dict,
    track_comp  : dict,
    shap_result : SHAPResult,
    output_path : str | Path,
    plots_dir   : str | Path = "outputs/plots/shap",
) -> str:
    """
    Assemble and write the complete SHAP analysis HTML report.

    Returns
    -------
    str — absolute path to written HTML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir   = Path(plots_dir)
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    top_df    = global_r.get("feature_ranking_df", pd.DataFrame())
    top_pairs = interaction_r.get("top_pairs_df",  pd.DataFrame())
    usi_pct   = track_comp.get("usi_contribution_pct", 0.0)

    body = (
        _exec_summary(shap_result, top_df, usi_pct)
        + _top_features_section(top_df, plots_dir)
        + _interaction_section(top_pairs, plots_dir)
        + _track_comparison_section(track_comp)
        + _local_section(local_r, plots_dir)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>M7.1 SHAP Analysis — Phishing Detection</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Module M7.1 — SHAP Explainability Analysis</h1>
<p class="meta">
  Project: Explainable and Bias-Aware ML for Phishing Website Detection
  &nbsp;|&nbsp; Generated: {timestamp}
  &nbsp;|&nbsp; Model: {shap_result.model_class}
  &nbsp;|&nbsp; Track B
  &nbsp;|&nbsp; SHAP backend: {'native' if shap_result.is_native_shap else 'fallback'}
</p>
{body}
<p class="footer">M7.1 complete. Next: M8.1 — LIME Explainability.</p>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info(f"SHAP report saved: {output_path}")
    return str(output_path)
