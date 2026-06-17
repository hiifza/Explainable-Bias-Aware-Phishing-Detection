"""
src/evaluation/evaluation_report.py
-------------------------------------
Generates the comprehensive HTML evaluation report for Module M6.1.

Output: outputs/reports/model_evaluation_report.html

Public API
----------
    generate_html_report(eval_results, metrics_df, selection, output_path,
                         plots_dir) -> str
"""

import sys
from datetime import datetime
from pathlib  import Path

import numpy  as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     font-size:14px;color:#1a1a1a;background:#fff;
     max-width:1140px;margin:40px auto;padding:0 28px 80px}
h1{font-size:22px;font-weight:700;border-bottom:3px solid #185FA5;
   padding-bottom:12px;margin-bottom:6px}
h2{font-size:16px;font-weight:600;color:#185FA5;margin:34px 0 10px;
   border-left:4px solid #185FA5;padding-left:10px}
h3{font-size:14px;font-weight:600;color:#333;margin:18px 0 8px}
p{line-height:1.65;color:#444;margin-bottom:10px}
p.meta{font-size:12px;color:#888;margin-bottom:22px}
table{width:100%;border-collapse:collapse;font-size:12.5px;margin:10px 0 20px}
th{background:#E6F1FB;padding:8px 11px;text-align:left;
   border:1px solid #B5D4F4;font-weight:600;white-space:nowrap}
td{padding:7px 11px;border:1px solid #ddd;vertical-align:top}
tr:nth-child(even) td{background:#F8FBFF}
.best{background:#E1F5EE!important;font-weight:600;color:#0F6E56}
.warn{background:#FAEEDA!important}
code{font-family:"SF Mono",Menlo,monospace;font-size:12px;
     background:#F1F3F5;padding:2px 5px;border-radius:3px}
.card{border:1px solid #B5D4F4;border-radius:8px;padding:14px 18px;
      background:#F0F7FF;margin:8px 0}
.card-v{font-size:24px;font-weight:700;color:#185FA5}
.card-l{font-size:12px;color:#666;margin-top:3px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:12px 0 20px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:12px 0}
.alert{border-left:4px solid #E24B4A;background:#FCEBEB;
       padding:10px 14px;border-radius:0 6px 6px 0;margin:10px 0}
.info{border-left:4px solid #378ADD;background:#E6F1FB;
      padding:10px 14px;border-radius:0 6px 6px 0;margin:10px 0}
.deploy{border-left:4px solid #1D9E75;background:#E1F5EE;
        padding:12px 16px;border-radius:0 6px 6px 0;margin:12px 0}
img{max-width:100%;border:1px solid #ddd;border-radius:6px;margin:6px 0;display:block}
.img-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0}
.footer{font-size:12px;color:#aaa;margin-top:56px;
        border-top:1px solid #eee;padding-top:16px}
.badge-a{background:#FCEBEB;color:#A32D2D;padding:2px 7px;
         border-radius:4px;font-size:11px;font-weight:600}
.badge-b{background:#E1F5EE;color:#0F6E56;padding:2px 7px;
         border-radius:4px;font-size:11px;font-weight:600}
"""

# ── HTML helpers ──────────────────────────────────────────────────────────────

def _rel(path: Path) -> str:
    try:    return "../../" + str(path.relative_to(ROOT))
    except: return str(path)


def _img(plots_dir: Path, name: str, caption: str = "") -> str:
    p   = plots_dir / name
    rel = _rel(p)
    cap = (f"<figcaption style='font-size:11px;color:#888;margin-top:4px'>"
           f"{caption}</figcaption>") if caption else ""
    return f"<figure><img src='{rel}' alt='{caption}'>{cap}</figure>"


def _metric_table(metrics_df: pd.DataFrame) -> str:
    """Render the full metrics comparison table as HTML."""
    show_cols = [
        "model","track","accuracy","precision","recall","f1",
        "roc_auc","pr_auc","mcc","balanced_accuracy","brier_score",
        "fpr","fnr","training_time",
    ]
    cols = [c for c in show_cols if c in metrics_df.columns]

    # Best per track per metric
    best_auc_A = metrics_df[metrics_df["track"]=="A"]["roc_auc"].max() \
        if "track" in metrics_df.columns else -1
    best_auc_B = metrics_df[metrics_df["track"]=="B"]["roc_auc"].max() \
        if "track" in metrics_df.columns else -1

    thead = "".join(f"<th>{c}</th>" for c in cols)
    tbody = ""
    for _, row in metrics_df.iterrows():
        is_best = (
            (row["track"] == "A" and abs(row.get("roc_auc",0) - best_auc_A) < 1e-6)
            or
            (row["track"] == "B" and abs(row.get("roc_auc",0) - best_auc_B) < 1e-6)
        )
        css = " class='best'" if is_best else ""
        cells = ""
        for c in cols:
            v = row.get(c, "—")
            if c == "track":
                badge = "badge-a" if v=="A" else "badge-b"
                cells += f"<td><span class='{badge}'>{v}</span></td>"
            elif isinstance(v, float):
                cells += f"<td>{v:.6f}</td>"
            else:
                cells += f"<td>{v}</td>"
        tbody += f"<tr{css}>{cells}</tr>\n"

    return (f"<table><thead><tr>{thead}</tr></thead>"
            f"<tbody>{tbody}</tbody></table>")


# ── Main report function ──────────────────────────────────────────────────────

def generate_html_report(
    eval_results: list[dict],
    metrics_df  : pd.DataFrame,
    selection   : dict,
    output_path : str | Path,
    plots_dir   : str | Path = "outputs/plots/evaluation",
) -> str:
    """
    Write the full Model Evaluation HTML report.

    Parameters
    ----------
    eval_results : from evaluator.run_full_evaluation()
    metrics_df   : from build_metrics_dataframe()
    selection    : from model_selector.select_final_deployment_model()
    output_path  : destination HTML path
    plots_dir    : base directory for plot images

    Returns
    -------
    str — absolute path to written HTML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir   = Path(plots_dir)
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    best_A = selection["best_A"]
    best_B = selection["best_B"]

    cm_dir  = plots_dir / "confusion_matrices"
    roc_dir = plots_dir / "roc"
    pr_dir  = plots_dir / "pr_curves"
    cal_dir = plots_dir / "calibration"

    # ── Executive summary cards ───────────────────────────────────────────────
    n_test = int(len(eval_results[0]["y_true"])) if eval_results else 0
    summary_cards = f"""
<div class="grid3">
  <div class="card">
    <div class="card-v">{len(eval_results)}</div>
    <div class="card-l">Models evaluated</div>
  </div>
  <div class="card">
    <div class="card-v">{n_test:,}</div>
    <div class="card-l">Test set rows</div>
  </div>
  <div class="card">
    <div class="card-v">{best_B['primary_score']:.4f}</div>
    <div class="card-l">Best Track B ROC AUC</div>
  </div>
</div>"""

    # ── Final deployment highlight ────────────────────────────────────────────
    deploy_box = f"""
<div class="deploy">
  <strong>FINAL_DEPLOYMENT_MODEL</strong>
  &nbsp;=&nbsp;
  <strong style="font-size:16px">{best_B['model_name']}</strong>
  &nbsp;(Track B)&nbsp;&nbsp;
  ROC AUC: <strong>{best_B['primary_score']:.6f}</strong>
  &nbsp;|&nbsp;
  F1: <strong>{best_B['secondary_score']:.6f}</strong>
  &nbsp;|&nbsp;
  Calibration: <strong>{best_B['tertiary_score']:.4f}</strong>
  <br>
  <em style="font-size:12px">{best_B['rationale']}</em>
</div>"""

    # ── Leakage impact ────────────────────────────────────────────────────────
    leak_delta = selection.get("leakage_impact_auc", 0)
    if abs(leak_delta) > 0.005:
        leakage_box = f"""
<div class="alert">
  <strong>Leakage impact (Track A − Track B best AUC):</strong>
  {leak_delta:+.6f}.
  Track A benefits from <code>URLSimilarityIndex</code> leakage.
  Track B is the honest deployment estimate.
</div>"""
    else:
        leakage_box = f"""
<div class="info">
  Leakage impact (Track A − Track B best AUC): {leak_delta:+.6f}.
  Minimal benefit from <code>URLSimilarityIndex</code>;
  both tracks perform similarly.
</div>"""

    # ── Metrics table ─────────────────────────────────────────────────────────
    metrics_table = _metric_table(metrics_df)

    # ── Plot sections ─────────────────────────────────────────────────────────
    def _safe_img(d, name, cap):
        return _img(d, name, cap) if (d / name).exists() else ""

    plots_html = f"""
<h2>4. ROC Curves</h2>
<div class="img-grid">
  {_safe_img(roc_dir, 'roc_trackA.png', 'ROC — Track A')}
  {_safe_img(roc_dir, 'roc_trackB.png', 'ROC — Track B')}
</div>
{_safe_img(roc_dir, 'roc_all_tracks.png', 'ROC — All models')}
{_safe_img(roc_dir, 'auc_ranking.png', 'AUC ranking chart')}

<h2>5. Precision-Recall Curves</h2>
<div class="img-grid">
  {_safe_img(pr_dir, 'pr_trackA.png', 'PR — Track A')}
  {_safe_img(pr_dir, 'pr_trackB.png', 'PR — Track B')}
</div>

<h2>6. Confusion Matrices</h2>
<div class="img-grid">
  {_safe_img(cm_dir, 'cm_grid_trackA.png', 'CM grid — Track A')}
  {_safe_img(cm_dir, 'cm_grid_trackB.png', 'CM grid — Track B')}
</div>

<h2>7. Calibration Analysis</h2>
{_safe_img(cal_dir, 'calibration_comparison.png', 'Calibration quality ranking')}
<div class="img-grid">
  {_safe_img(cal_dir, 'calibration_trackA.png', 'Reliability diagrams — Track A')}
  {_safe_img(cal_dir, 'calibration_trackB.png', 'Reliability diagrams — Track B')}
</div>
<div class="img-grid">
  {_safe_img(cal_dir, 'prob_dist_trackA.png', 'Prob distributions — Track A')}
  {_safe_img(cal_dir, 'prob_dist_trackB.png', 'Prob distributions — Track B')}
</div>
"""

    # ── Downstream interface table ────────────────────────────────────────────
    interface_table = f"""
<table>
  <tr><th>Module</th><th>Variable</th><th>Type / Shape</th><th>Description</th></tr>
  <tr><td>M7.1 SHAP</td><td><code>SHAP_model</code></td>
      <td>{type(best_B['fitted_model']).__name__}</td>
      <td>Track B winner — TreeExplainer or LinearExplainer</td></tr>
  <tr><td>M7.1 SHAP</td><td><code>SHAP_X_train</code></td>
      <td>DataFrame</td><td>Track B training data (SHAP background)</td></tr>
  <tr><td>M7.1 SHAP</td><td><code>SHAP_X_test</code></td>
      <td>DataFrame</td><td>Track B test data (SHAP values)</td></tr>
  <tr><td>M8.1 LIME</td><td><code>LIME_predict_fn</code></td>
      <td>callable</td><td><code>best_model_B.predict_proba</code></td></tr>
  <tr><td>M8.1 LIME</td><td><code>LIME_X_train_np</code></td>
      <td>np.ndarray</td><td>Track B training data (numpy)</td></tr>
  <tr><td>M9 Bias</td><td><code>BIAS_X_test, BIAS_y_test, BIAS_y_pred</code></td>
      <td>DataFrame / Series</td><td>Track B test set + predictions</td></tr>
  <tr><td>M10 Blindspot</td><td><code>BLINDSPOT_y_pred, BLINDSPOT_y_proba</code></td>
      <td>np.ndarray</td><td>Track B predictions for FP/FN analysis</td></tr>
  <tr><td>All</td><td><code>feature_names_B</code></td>
      <td>list[str] ({metrics_df[metrics_df['track']=='B'].iloc[0].get('model','')})</td>
      <td>56 Track B feature names</td></tr>
</table>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>M6.1 Model Evaluation — Phishing Detection</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Module M6.1 — Comprehensive Model Evaluation &amp; Final Model Selection</h1>
<p class="meta">
  Project: Explainable and Bias-Aware ML for Phishing Website Detection
  &nbsp;|&nbsp; Generated: {timestamp}
  &nbsp;|&nbsp; 8 models &times; 2 tracks
</p>

<h2>1. Executive Summary</h2>
{summary_cards}
{deploy_box}
{leakage_box}

<h2>2. Full Metrics Comparison Table</h2>
<p>Highlighted rows = winners per track. Columns sorted by evaluation priority.
   Brier Score: lower is better. All others: higher is better.</p>
{metrics_table}

<h2>3. Selection Rationale</h2>
<table>
  <tr><th>Track</th><th>Winner</th><th>ROC AUC</th>
      <th>F1</th><th>Calibration</th><th>Rationale</th></tr>
  <tr>
    <td><span class="badge-a">A</span></td>
    <td><strong>{best_A['model_name']}</strong></td>
    <td>{best_A['primary_score']:.6f}</td>
    <td>{best_A['secondary_score']:.6f}</td>
    <td>{best_A['tertiary_score']:.4f}</td>
    <td>{best_A['rationale']}</td>
  </tr>
  <tr class="best">
    <td><span class="badge-b">B</span></td>
    <td><strong>{best_B['model_name']}</strong></td>
    <td>{best_B['primary_score']:.6f}</td>
    <td>{best_B['secondary_score']:.6f}</td>
    <td>{best_B['tertiary_score']:.4f}</td>
    <td>{best_B['rationale']}</td>
  </tr>
</table>

{plots_html}

<h2>8. Downstream Module Interface</h2>
{interface_table}

<p class="footer">
  M6.1 complete.
  FINAL_DEPLOYMENT_MODEL = <strong>{best_B['model_name']}</strong> (Track B).
  Next: M7.1 — SHAP Explainability.
</p>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Evaluation report saved: {output_path}")
    return str(output_path)
