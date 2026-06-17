"""
src/explainability/lime_report.py
-----------------------------------
Generates the comprehensive HTML report for Module M8.1.

Output: outputs/reports/lime_analysis_report.html

Public API
----------
    generate_lime_report(lime_local_r, comparison_r, global_metrics,
                         consistency, lime_backend, output_path,
                         plots_dir)  -> str
"""

import sys
from datetime import datetime
from pathlib  import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

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
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:12px 0 20px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:12px 0}
.card{border:1px solid #B5D4F4;border-radius:8px;padding:13px 17px;background:#F0F7FF}
.card-v{font-size:24px;font-weight:700;color:#185FA5}
.card-l{font-size:12px;color:#666;margin-top:3px}
.pass{background:#E1F5EE;color:#0F6E56;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600}
.warn{background:#FAEEDA;color:#854F0B;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600}
.fail{background:#FCEBEB;color:#A32D2D;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600}
.alert{border-left:4px solid #E24B4A;background:#FCEBEB;padding:10px 14px;
       border-radius:0 6px 6px 0;margin:10px 0}
.info{border-left:4px solid #378ADD;background:#E6F1FB;padding:10px 14px;
      border-radius:0 6px 6px 0;margin:10px 0}
.ok{border-left:4px solid #1D9E75;background:#E1F5EE;padding:10px 14px;
    border-radius:0 6px 6px 0;margin:10px 0}
img{max-width:100%;border:1px solid #ddd;border-radius:6px;margin:6px 0;display:block}
.img2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0}
.footer{font-size:12px;color:#aaa;margin-top:56px;border-top:1px solid #eee;padding-top:16px}
"""


def _rel(path: Path) -> str:
    try:    return "../../" + str(path.relative_to(ROOT))
    except: return str(path)


def _img(path: Path, caption: str = "") -> str:
    if not isinstance(path, Path) or not path.exists():
        return ""
    r   = _rel(path)
    cap = (f"<figcaption style='font-size:11px;color:#888;margin-top:4px'>"
           f"{caption}</figcaption>") if caption else ""
    return f"<figure><img src='{r}' alt='{caption}'>{cap}</figure>"


def _agreement_badge(score: float) -> str:
    if   score >= 0.8: return f"<span class='pass'>✓ {score:.2f}</span>"
    elif score >= 0.4: return f"<span class='warn'>~ {score:.2f}</span>"
    else:              return f"<span class='fail'>✗ {score:.2f}</span>"


def _executive_summary(
    global_metrics: dict,
    consistency   : dict,
    n_explained   : int,
    lime_backend  : str,
) -> str:
    mean_a    = global_metrics.get("mean", 0)
    n_high_d  = global_metrics.get("n_high_disagree", 0)
    con_score = consistency.get("consistency_score", 0)

    agree_badge = _agreement_badge(mean_a)

    return f"""
<h2>1. Executive Summary</h2>
<div class="grid3">
  <div class="card"><div class="card-v">{n_explained}</div>
    <div class="card-l">Samples explained</div></div>
  <div class="card"><div class="card-v">{mean_a:.4f}</div>
    <div class="card-l">Mean SHAP-LIME agreement</div></div>
  <div class="card"><div class="card-v">{con_score:.4f}</div>
    <div class="card-l">Feature consistency score</div></div>
</div>
<div class="{'ok' if mean_a >= 0.5 else 'warn' if mean_a >= 0.3 else 'alert'}">
  <strong>Mean agreement: {agree_badge}</strong>
  &nbsp;|&nbsp; High-disagreement samples (&lt;0.4): <strong>{n_high_d}</strong>
  &nbsp;|&nbsp; LIME backend: <code>{lime_backend}</code>
</div>"""


def _agreement_section(
    agreement_df: pd.DataFrame,
    global_metrics: dict,
    plots_dir: Path,
) -> str:
    by_cat = global_metrics.get("by_category", {})
    cat_rows = "".join(
        f"<tr><td><strong>{c.upper()}</strong></td>"
        f"<td>{v.get('mean', 0):.4f}</td>"
        f"<td>{v.get('median', 0):.4f}</td>"
        f"<td>{v.get('n', 0)}</td></tr>"
        for c, v in by_cat.items()
    )

    # Top 10 agreement table
    top10 = agreement_df.sort_values("agreement_score", ascending=False).head(10)
    agg_rows = "".join(
        f"<tr><td>{r['sample_id']}</td>"
        f"<td>{r['category'].upper()}</td>"
        f"<td style='font-size:11px'>{r['shap_features']}</td>"
        f"<td style='font-size:11px'>{r['lime_features']}</td>"
        f"<td>{r['overlap_count']}</td>"
        f"<td>{_agreement_badge(r['agreement_score'])}</td></tr>"
        for _, r in top10.iterrows()
    )
    agree_plot = plots_dir / "shap_lime_agreement.png"

    return f"""
<h2>2. SHAP vs LIME Agreement Analysis</h2>
{_img(agree_plot, 'Agreement distribution, category comparison, R² scatter')}
<table>
  <tr><th>Category</th><th>Mean agreement</th><th>Median</th><th>N</th></tr>
  {cat_rows}
</table>
<h3>Top-10 Best Agreement Samples</h3>
<table>
  <tr><th>Sample ID</th><th>Cat</th><th>SHAP top-5</th><th>LIME top-5</th>
      <th>Overlap</th><th>Score</th></tr>
  {agg_rows}
</table>"""


def _high_disagree_section(high_dis: pd.DataFrame) -> str:
    if high_dis.empty:
        return """
<h2>3. High-Disagreement Cases (agreement &lt; 0.40)</h2>
<div class="ok">No high-disagreement cases found — SHAP and LIME are consistent.</div>"""

    rows = "".join(
        f"<tr><td>{r['sample_id']}</td>"
        f"<td>{r['category'].upper()}</td>"
        f"<td style='font-size:11px'>{r['shap_features']}</td>"
        f"<td style='font-size:11px'>{r['lime_features']}</td>"
        f"<td>{r['overlap_count']}</td>"
        f"<td>{_agreement_badge(r['agreement_score'])}</td>"
        f"<td>{r.get('local_r2', 0):.4f}</td></tr>"
        for _, r in high_dis.iterrows()
    )
    return f"""
<h2>3. High-Disagreement Cases (agreement &lt; 0.40)</h2>
<div class="alert">
  <strong>{len(high_dis)} samples</strong> with agreement &lt; 0.40.
  Possible causes: low LIME local R² (poor local fit), highly non-linear
  decision boundary in sample neighbourhood, or sparse feature region.
</div>
<table>
  <tr><th>Sample ID</th><th>Cat</th><th>SHAP top-5</th><th>LIME top-5</th>
      <th>Overlap</th><th>Score</th><th>LIME R²</th></tr>
  {rows}
</table>"""


def _consistency_section(consistency: dict, plots_dir: Path) -> str:
    shared    = consistency.get("shared", [])
    shap_only = consistency.get("shap_only", [])
    lime_only = consistency.get("lime_only", [])
    score     = consistency.get("consistency_score", 0)
    lime_r    = consistency.get("lime_global_ranking", pd.DataFrame())

    lime_rows = ""
    if not lime_r.empty:
        lime_rows = "".join(
            f"<tr><td>{r['rank']}</td>"
            f"<td><code>{r['feature']}</code></td>"
            f"<td>{r['total_abs_contribution']:.6f}</td></tr>"
            for _, r in lime_r.head(10).iterrows()
        )
    consist_p = plots_dir / "feature_consistency.png"

    return f"""
<h2>4. Feature Consistency Analysis</h2>
{_img(consist_p, 'Feature overlap and LIME global ranking')}
<div class="grid3">
  <div class="card">
    <div class="card-v">{len(shared)}</div>
    <div class="card-l">Shared features (SHAP ∩ LIME top-20)</div>
  </div>
  <div class="card">
    <div class="card-v">{len(shap_only)}</div>
    <div class="card-l">SHAP-only features</div>
  </div>
  <div class="card">
    <div class="card-v">{len(lime_only)}</div>
    <div class="card-l">LIME-only features</div>
  </div>
</div>
<p><strong>Consistency score: {score:.4f}</strong>
   ({len(shared)} / 20 features agree in both top-20 lists)</p>
<h3>Shared features</h3>
<p>{', '.join(f'<code>{f}</code>' for f in sorted(shared)) or '—'}</p>
<h3>SHAP-only (top-20)</h3>
<p>{', '.join(f'<code>{f}</code>' for f in sorted(shap_only)) or '—'}</p>
<h3>LIME-only (top-20)</h3>
<p>{', '.join(f'<code>{f}</code>' for f in sorted(lime_only)) or '—'}</p>
<h3>LIME Global Ranking (top-10)</h3>
<table>
  <tr><th>Rank</th><th>Feature</th><th>Total |contribution|</th></tr>
  {lime_rows}
</table>"""


def _local_section(lime_local_r: dict, plots_dir: Path) -> str:
    cats   = ["tp", "tn", "fp", "fn"]
    labels = {"tp": "True Positive", "tn": "True Negative",
              "fp": "False Positive", "fn": "False Negative"}
    content = ""
    for cat in cats:
        cnt   = lime_local_r.get("category_counts", {}).get(cat, 0)
        paths = lime_local_r.get("plot_paths", {}).get(cat, [])
        imgs  = "".join(_img(p, f"{cat.upper()} sample {i}")
                        for i, p in enumerate(paths[:2]))
        content += f"""
<h3>{labels[cat]} ({cnt} samples)</h3>
{imgs if imgs else '<p>No samples.</p>'}"""

    return f"<h2>5. Local LIME Explanations</h2>{content}"


def generate_lime_report(
    lime_local_r  : dict,
    comparison_r  : dict,
    global_metrics: dict,
    consistency   : dict,
    lime_backend  : str,
    output_path   : str | Path,
    plots_dir     : str | Path = "outputs/plots/lime",
) -> str:
    """
    Assemble and write the M8.1 LIME analysis HTML report.

    Returns
    -------
    str — absolute path to written HTML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir   = Path(plots_dir)
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    agreement_df = comparison_r.get("agreement_df", pd.DataFrame())
    high_dis     = comparison_r.get("high_disagree_df", pd.DataFrame())
    n_explained  = int(lime_local_r.get(
        "category_counts",
        {c: 0 for c in ["tp","tn","fp","fn"]}
    ).get("tp", 0) + lime_local_r.get("category_counts", {}).get("tn", 0) +
        lime_local_r.get("category_counts", {}).get("fp", 0) +
        lime_local_r.get("category_counts", {}).get("fn", 0))

    body = (
        _executive_summary(global_metrics, consistency, n_explained, lime_backend)
        + _agreement_section(agreement_df, global_metrics, plots_dir)
        + _high_disagree_section(high_dis)
        + _consistency_section(consistency, plots_dir)
        + _local_section(lime_local_r, plots_dir)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>M8.1 LIME Analysis — Phishing Detection</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Module M8.1 — LIME Explainability &amp; SHAP Agreement Validation</h1>
<p class="meta">
  Project: Explainable and Bias-Aware ML for Phishing Website Detection
  &nbsp;|&nbsp; Generated: {timestamp}
  &nbsp;|&nbsp; LIME backend: {lime_backend}
</p>
{body}
<p class="footer">M8.1 complete. Next: M9 — Bias Analysis.</p>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info(f"LIME report saved: {output_path}")
    return str(output_path)
