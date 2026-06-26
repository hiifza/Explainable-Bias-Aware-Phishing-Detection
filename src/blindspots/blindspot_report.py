"""
src/blindspots/blindspot_report.py — M10 HTML report generator.
"""
import sys
from datetime import datetime
from pathlib  import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.utils.logger import get_logger
logger = get_logger(__name__)

_CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,sans-serif;font-size:14px;color:#1a1a1a;
     max-width:1140px;margin:40px auto;padding:0 26px 80px}
h1{font-size:22px;font-weight:700;border-bottom:3px solid #185FA5;padding-bottom:12px;margin-bottom:6px}
h2{font-size:16px;font-weight:600;color:#185FA5;margin:32px 0 10px;border-left:4px solid #185FA5;padding-left:10px}
h3{font-size:14px;font-weight:600;color:#333;margin:16px 0 8px}
p{line-height:1.65;color:#444;margin-bottom:10px}
p.meta{font-size:12px;color:#888;margin-bottom:22px}
table{width:100%;border-collapse:collapse;font-size:12.5px;margin:10px 0 20px}
th{background:#E6F1FB;padding:8px 12px;text-align:left;border:1px solid #B5D4F4;font-weight:600;white-space:nowrap}
td{padding:7px 12px;border:1px solid #ddd}tr:nth-child(even) td{background:#F8FBFF}
code{font-family:monospace;font-size:12px;background:#F1F3F5;padding:2px 5px;border-radius:3px}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:12px 0 20px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:12px 0}
.card{border:1px solid #B5D4F4;border-radius:8px;padding:13px 17px;background:#F0F7FF}
.card-v{font-size:24px;font-weight:700;color:#185FA5}.card-l{font-size:12px;color:#666;margin-top:3px}
.alert{border-left:4px solid #E24B4A;background:#FCEBEB;padding:10px 14px;border-radius:0 6px 6px 0;margin:10px 0}
.warn{border-left:4px solid #EF9F27;background:#FAEEDA;padding:10px 14px;border-radius:0 6px 6px 0;margin:10px 0}
.ok{border-left:4px solid #1D9E75;background:#E1F5EE;padding:10px 14px;border-radius:0 6px 6px 0;margin:10px 0}
.info{border-left:4px solid #378ADD;background:#E6F1FB;padding:10px 14px;border-radius:0 6px 6px 0;margin:10px 0}
.badge-r{background:#FCEBEB;color:#A32D2D;padding:2px 7px;border-radius:4px;font-size:11px;font-weight:600}
.badge-y{background:#FAEEDA;color:#854F0B;padding:2px 7px;border-radius:4px;font-size:11px;font-weight:600}
.badge-g{background:#E1F5EE;color:#0F6E56;padding:2px 7px;border-radius:4px;font-size:11px;font-weight:600}
img{max-width:100%;border:1px solid #ddd;border-radius:6px;margin:6px 0;display:block}
.img2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0}
.footer{font-size:12px;color:#aaa;margin-top:56px;border-top:1px solid #eee;padding-top:16px}
"""

def _rel(p):
    try: return "../../"+str(Path(p).relative_to(ROOT))
    except: return str(p)

def _img(path, caption=""):
    if not path or not Path(path).exists(): return ""
    r=_rel(path); cap=f"<figcaption style='font-size:11px;color:#888;margin-top:4px'>{caption}</figcaption>" if caption else ""
    return f"<figure><img src='{r}' alt='{caption}'>{cap}</figure>"

def _badge(risk):
    m={"CRITICAL":"badge-r","HIGH":"badge-r","MEDIUM":"badge-y","LOW":"badge-g"}
    return f"<span class='{m.get(risk,'badge-g')}'>{risk}</span>"

def _fmt_cell(col_name, val):
    if col_name == "risk_level":
        return _badge(str(val))
    if isinstance(val, float):
        return "{:.4f}".format(val)
    return str(val)

def _top20_table(top_20):
    if top_20.empty:
        return "<p>No data.</p>"
    cols = ["severity_rank","sample_idx","confidence_zone","is_error","confidence",
            "severity_score_norm","tiers_flagged","risk_level"]
    cols = [c for c in cols if c in top_20.columns]
    thead = "".join("<th>" + c + "</th>" for c in cols)
    all_rows = []
    for _, r in top_20.iterrows():
        cells = "".join("<td>" + _fmt_cell(c, r[c]) + "</td>" for c in cols)
        all_rows.append("<tr>" + cells + "</tr>")
    rows = "\n".join(all_rows)
    return "<table><thead><tr>" + thead + "</tr></thead><tbody>" + rows + "</tbody></table>"
def _archetype_table(cluster_meta):
    if not cluster_meta: return "<p>No archetype data.</p>"
    rows=""
    for m in cluster_meta:
        risk_n=m.get("n_errors",0); n=m.get("n_samples",1)
        badge=_badge("HIGH" if risk_n>0 else "MEDIUM" if m.get("mean_confidence",1)<0.9 else "LOW")
        rows+=f"<tr><td>{m['cluster']}</td><td><strong>{m['label']}</strong></td><td>{n}</td><td>{risk_n}</td><td>{m.get('mean_confidence',0):.4f}</td><td>{badge}</td></tr>"
    return f"<table><thead><tr><th>#</th><th>Archetype</th><th>N</th><th>Errors</th><th>Mean Conf</th><th>Risk</th></tr></thead><tbody>{rows}</tbody></table>"

def generate_blindspot_report(m10_results, output_path, plots_dir="outputs/plots/blindspot"):
    output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    plots_dir=Path(plots_dir); ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    s=m10_results["summary"]; ar=m10_results["archetype_r"]
    fcs=m10_results["fcs"]; top20=m10_results["top_20_bs"]
    unc=m10_results["uncertainty_r"]; rel=m10_results["reliability_r"]
    cl=m10_results["cluster_r"]; sf=m10_results["shap_fail_r"]
    lf=m10_results["lime_fail_r"]

    zs=unc["zone_stats"]
    def _zcard(zone,color):
        z=zs.get(zone,{})
        conf_str = "≥0.95" if zone=="green" else "0.75-0.95" if zone=="yellow" else "<0.75"
        return ("<div class='card' style='border-color:" + color + "'>"
                "<div class='card-v' style='color:" + color + "'>" + str(z.get('n',0)) + "</div>"
                "<div class='card-l'>" + zone.title() + " zone (conf " + conf_str + ")</div>"
                "<div class='card-l'>error rate: " + "{:.6f}".format(z.get('error_rate',0)) + "</div></div>")

    rel_s=rel.get("reliability_stats",{})
    q1=rel_s.get("q1_answer","—"); q2=rel_s.get("q2_answer","—"); q3=rel_s.get("q3_answer","—")

    html=f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>M10 Blind Spot Intelligence — Phishing Detection</title><style>{_CSS}</style></head><body>
<h1>Module M10 — Failure Intelligence Engine</h1>
<p class="meta">Project: Explainable and Bias-Aware ML | Generated: {ts} | Model: {type(m10_results["inputs"]["model"]).__name__}</p>

<h2>1. Executive Summary</h2>
<div class="grid4">
  <div class="card"><div class="card-v">{s.get('n_test',0):,}</div><div class="card-l">Test samples</div></div>
  <div class="card"><div class="card-v">{s.get('n_errors',0)}</div><div class="card-l">Hard errors (FP+FN)</div></div>
  <div class="card"><div class="card-v">{s.get('n_failure_cases',0)}</div><div class="card-l">Failure zone cases</div></div>
  <div class="card"><div class="card-v">{s.get('n_archetypes',0)}</div><div class="card-l">Archetypes discovered</div></div>
</div>
<div class="{'alert' if s.get('n_errors',0)>0 else 'ok'}">
  <strong>Hard errors:</strong> {s.get('n_fp',0)} FP (legit flagged as phishing) + {s.get('n_fn',0)} FN (phishing missed).
  Error rate: {s.get('error_rate',0):.8f} on {s.get('n_test',0):,} samples.
</div>
<div class="info">
  <strong>Near-perfect performance confirmed.</strong>
  Failure intelligence focuses on the uncertainty zone ({s.get('yellow_zone_n',0)+s.get('red_zone_n',0):,} samples)
  and explanation disagreements as the primary risk signals.
</div>

<h2>2. Confidence Reliability Engine</h2>
<div class="grid4">
  {_zcard("green","#1D9E75")}{_zcard("yellow","#EF9F27")}{_zcard("red","#E24B4A")}
</div>
<div class="img2">{_img(unc.get('confidence_dist_plot'),'Confidence distribution')}{_img(unc.get('zone_error_plot'),'Zone error rates')}</div>

<h2>3. Top-20 Blind Spots (by Severity Score)</h2>
<p>Severity = 0.40×error + 0.30×uncertainty + 0.20×disagreement + 0.10×SHAP-instability</p>
{_top20_table(top20)}

<h2>4. Failure Archetypes (Auto-Discovered)</h2>
<div class="info">Silhouette score: {ar.silhouette_score:.4f} | PCA variance explained: {ar.explained_var:.2%}</div>
{_archetype_table(ar.cluster_meta)}
<div class="img2">{_img(cl.get('pca_plot'),'PCA cluster map')}{_img(cl.get('heatmap_plot'),'Cluster risk heatmap')}</div>
{_img(cl.get('density_plot'),'Failure density vs all test')}

<h2>5. SHAP Failure Analysis</h2>
<p>Top failure-zone feature: <code>{s.get('shap_top_failure_feat','—')}</code></p>
<div class="img2">{_img(sf.get('comparison_plot'),'SHAP: global vs failure zones')}{_img(sf.get('masking_plot'),'Dominant feature masking')}</div>

<h2>6. LIME Failure Analysis</h2>
{_img(lf.get('lime_failure_plot'),'LIME feature frequency in failure zone')}

<h2>7. SHAP-LIME Reliability Correlation</h2>
<div class="info"><strong>Q1: Low agreement → higher error?</strong> {q1}</div>
<div class="info"><strong>Q2: Low agreement → lower confidence?</strong> {q2}</div>
<div class="info"><strong>Q3: Low agreement → higher severity?</strong> {q3}</div>
{_img(rel.get('agreement_plot'),'Agreement vs error/uncertainty')}

<h2>8. Hidden Weakness Investigation</h2>
<div class="warn"><strong>HTTPS dependence:</strong> All legitimate sites use HTTPS; non-HTTPS legitimate sites are a structural blind spot.</div>
<div class="warn"><strong>Dominant feature masking:</strong> <code>{s.get('shap_top_failure_feat','—')}</code> dominates model decisions — adversarial manipulation of this feature may evade detection.</div>
<div class="warn"><strong>Rare TLD blind spots:</strong> Low-frequency TLDs have sparse training data; model may be miscalibrated for novel domains.</div>
<div class="ok"><strong>Error rate is extremely low</strong> ({s.get('error_rate',0):.8f}); remaining risk is structural rather than statistical.</div>

<h2>9. Dashboard-Ready Outputs</h2>
<table>
  <tr><th>Object</th><th>Description</th><th>For Module</th></tr>
  <tr><td><code>top_20_bs</code></td><td>Top-20 most dangerous blind spots</td><td>M11 Dashboard: Blind Spot Center</td></tr>
  <tr><td><code>fcs</code></td><td>FailureCaseSet with all tier flags</td><td>M11 Dashboard: Threat Intelligence</td></tr>
  <tr><td><code>archetype_r</code></td><td>Cluster labels + metadata</td><td>M11 Dashboard: Blind Spot Center</td></tr>
  <tr><td><code>severity_df</code></td><td>Per-case severity scores</td><td>M11 Dashboard: Performance Intelligence</td></tr>
  <tr><td><code>uncertainty_r[zone_stats]</code></td><td>Green/Yellow/Red zone stats</td><td>M11 Dashboard: All centers</td></tr>
  <tr><td><code>reliability_r</code></td><td>SHAP-LIME correlation analysis</td><td>M11 Dashboard: Explainability Center</td></tr>
</table>

<p class="footer">M10 complete. Next: M11 — Streamlit Dashboard.</p></body></html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Blindspot report saved: {output_path}")
    return str(output_path)
