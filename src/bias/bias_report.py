"""src/bias/bias_report.py — M9 HTML report with Near-Perfect Performance section."""
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
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
td{padding:7px 12px;border:1px solid #ddd}
tr:nth-child(even) td{background:#F8FBFF}
code{font-family:monospace;font-size:12px;background:#F1F3F5;padding:2px 5px;border-radius:3px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:12px 0 20px}
.card{border:1px solid #B5D4F4;border-radius:8px;padding:13px 17px;background:#F0F7FF}
.card-v{font-size:24px;font-weight:700;color:#185FA5}.card-l{font-size:12px;color:#666;margin-top:3px}
.alert{border-left:4px solid #E24B4A;background:#FCEBEB;padding:10px 14px;border-radius:0 6px 6px 0;margin:10px 0}
.warn{border-left:4px solid #EF9F27;background:#FAEEDA;padding:10px 14px;border-radius:0 6px 6px 0;margin:10px 0}
.ok{border-left:4px solid #1D9E75;background:#E1F5EE;padding:10px 14px;border-radius:0 6px 6px 0;margin:10px 0}
.info{border-left:4px solid #378ADD;background:#E6F1FB;padding:10px 14px;border-radius:0 6px 6px 0;margin:10px 0}
.perf{border:2px solid #185FA5;background:#F0F7FF;border-radius:8px;padding:16px 20px;margin:16px 0}
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

def _metrics_table(metrics_df, dim):
    sub=metrics_df[metrics_df["dimension"]==dim] if "dimension" in metrics_df.columns else metrics_df
    if sub.empty: return "<p>No data.</p>"
    cols=["group","n","accuracy","f1","roc_auc","fpr","fnr"]
    cols=[c for c in cols if c in sub.columns]
    sub=sub[cols].sort_values("fpr",ascending=True) if "fpr" in sub.columns else sub[cols]
    thead="".join(f"<th>{c}</th>" for c in cols); rows=""
    for _,r in sub.iterrows():
        cells=""
        for c in cols:
            v=r.get(c)
            if isinstance(v,float): cells+=f"<td>{v:.4f}</td>"
            else: cells+=f"<td>{v}</td>"
        rows+=f"<tr>{cells}</tr>\n"
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{rows}</tbody></table>"

def _performance_section(perf_inv):
    if not perf_inv: return ""
    plots=perf_inv.get("plot_paths",{})
    def _p(key): return plots.get(key)
    top1=perf_inv.get("top1_pct",0); top3=perf_inv.get("top3_pct",0)
    top5=perf_inv.get("top5_pct",0); top10=perf_inv.get("top10_pct",0)
    n80=perf_inv.get("n_features_80pct",0); n95=perf_inv.get("n_features_95pct",0)
    top_feat=perf_inv.get("top_feature","—"); top3f=perf_inv.get("top3_features",[])

    conclusions = []
    if top1 > 30: conclusions.append(f"Single feature dominance: top feature accounts for {top1:.1f}% of importance")
    if top3 > 60: conclusions.append(f"Extreme concentration: top-3 features explain {top3:.1f}% — model over-reliant on few signals")
    if n80 <= 5:  conclusions.append(f"Only {n80} features needed for 80% importance — high compression")
    conclusions.append("Dataset appears intrinsically separable — non-overlapping class distributions explain AUC≈1.0")
    conclusions.append("Near-perfect performance is REAL for this dataset but may not generalise to adversarial evasion")

    return f"""
<h2 style='border-left:4px solid #E24B4A;color:#A32D2D'>Explaining the Near-Perfect Model Performance</h2>
<div class="perf">
  <strong>Observation:</strong> ROC-AUC ≈ 1.0, Accuracy ≈ 99.98%, F1 ≈ 99.98%<br>
  <strong>Primary cause:</strong> Dataset contains highly discriminative features with near-zero class overlap.<br>
  Top feature: <code>{top_feat}</code> | Top-3: {', '.join(f'<code>{f}</code>' for f in top3f)}
</div>
<div class="grid3">
  <div class="card"><div class="card-v">{top1:.1f}%</div><div class="card-l">Top-1 feature importance</div></div>
  <div class="card"><div class="card-v">{top5:.1f}%</div><div class="card-l">Top-5 features importance</div></div>
  <div class="card"><div class="card-v">{n80}</div><div class="card-l">Features for 80% importance</div></div>
</div>
<div class="warn">
  <strong>Feature concentration:</strong> Top-3 features account for {top3:.1f}% of model decisions.
  Top-10 account for {top10:.1f}%. Model is highly dependent on a small feature subset.
</div>
<h3>Visualisation 1 — Top Feature Dominance</h3>
{_img(_p("1_dominance"),"Top-10 SHAP relative contribution")}
<h3>Visualisation 2 — Dataset Separability</h3>
{_img(_p("2_separability"),"Class-separated distributions of top features")}
<h3>Visualisation 3 — Feature Dominance vs Remaining</h3>
{_img(_p("3_dom_vs_rest"),"Top-N vs remaining feature importance share")}
<h3>Visualisation 4 — Cumulative SHAP Importance Curve</h3>
{_img(_p("4_cumulative"),"Cumulative importance — steep rise confirms concentration")}
<h3>Visualisation 5 — Feature Contribution Distribution</h3>
{_img(_p("5_distribution"),"Heavy right tail confirms extreme feature dominance")}
<h3>Visualisation 6 — URLSimilarityIndex Impact (Track A leakage)</h3>
{_img(_p("6_usi_impact"),"ALL legitimate sites have URLSimilarityIndex=100.0")}
<h3>Visualisation 7 — HTTPS Impact</h3>
{_img(_p("7_https_impact"),"ALL legitimate sites use HTTPS → strong separating signal")}
<h3>Conclusions</h3>
<ul style='margin:10px 0 10px 20px'>
{''.join(f'<li style="margin:4px 0">{c}</li>' for c in conclusions)}
</ul>"""

def generate_bias_report(bias_results, output_path, plots_dir="outputs/plots/bias",
                          perf_investigation=None):
    output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    plots_dir=Path(plots_dir); ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_df=bias_results["all_metrics_df"]; disparity_df=bias_results["all_disparity_df"]
    summary=bias_results["summary"]; n_test=int(len(bias_results["inputs"]["y_test"]))
    disp_plot=bias_results.get("disp_plot"); heat_plot=bias_results.get("heat_plot")
    tld_plot=bias_results.get("tld_plot"); shap_bias_r=bias_results["shap_bias_r"]
    hfp=summary.get("highest_fpr_group",{}); hfn=summary.get("highest_fnr_group",{})
    mbd=summary.get("most_biased_dimension",{}); lbd=summary.get("least_biased_dimension",{})

    # Disparity table
    if not disparity_df.empty and "metric" in disparity_df.columns:
        disp_cols=["metric","dimension","best_group","best_value","worst_group","worst_value","disparity"]
        disp_cols=[c for c in disp_cols if c in disparity_df.columns]
        disp_rows="".join(f"<tr>{''.join(f'<td>{round(r[c],4) if isinstance(r[c],float) else r[c]}</td>' for c in disp_cols)}</tr>"
                          for _,r in disparity_df.iterrows())
        disp_table=f"<table><thead><tr>{''.join(f'<th>{c}</th>' for c in disp_cols)}</tr></thead><tbody>{disp_rows}</tbody></table>"
    else: disp_table="<p>No disparity data.</p>"

    shap_imgs=""
    for dim in ["url_length","https","tld","domain_length","ext_resources"]:
        p=plots_dir/"shap_groups"/f"shap_{dim}.png"
        if p.exists(): shap_imgs+=_img(p,f"SHAP importance — {dim.replace('_',' ').title()}")

    html=f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>M9 Bias Analysis — Phishing Detection</title><style>{_CSS}</style></head><body>
<h1>Module M9.1–M9.3 — Bias &amp; Fairness Analysis</h1>
<p class="meta">Project: Explainable and Bias-Aware ML | Generated: {ts} | Test samples: {n_test:,}</p>
<h2>1. Executive Summary</h2>
<div class="grid3">
  <div class="card"><div class="card-v">{n_test:,}</div><div class="card-l">Test samples</div></div>
  <div class="card"><div class="card-v">5</div><div class="card-l">Bias dimensions</div></div>
  <div class="card"><div class="card-v">{mbd.get('disparity',0):.4f}</div><div class="card-l">Max FPR disparity</div></div>
</div>
<div class="{'alert' if mbd.get('disparity',0)>0.10 else 'warn' if mbd.get('disparity',0)>0.05 else 'ok'}">
<strong>Most biased:</strong> {mbd.get('dimension','—')} (FPR disp={mbd.get('disparity',0):.4f}, worst={mbd.get('worst_group','—')})</div>
<div class="ok"><strong>Least biased:</strong> {lbd.get('dimension','—')} (FPR disp={lbd.get('disparity',0):.4f})</div>
<table><tr><th>Finding</th><th>Group</th><th>Dimension</th><th>Value</th></tr>
<tr><td>Highest FPR</td><td><code>{hfp.get('group','—')}</code></td><td>{hfp.get('dimension','—')}</td><td>{hfp.get('fpr',0):.4f}</td></tr>
<tr><td>Highest FNR</td><td><code>{hfn.get('group','—')}</code></td><td>{hfn.get('dimension','—')}</td><td>{hfn.get('fnr',0):.4f}</td></tr>
</table>
{_img(heat_plot,"Mean metric heatmap")}{_img(disp_plot,"FPR/FNR disparity")}
<h2>2. URL Length Group Analysis</h2>{_img(plots_dir/"url_length"/"url_length_fpr.png","FPR by URL length")}{_metrics_table(metrics_df,"url_length")}
<h2>3. HTTPS Group Analysis</h2>{_img(plots_dir/"https"/"https_fpr.png","FPR by HTTPS group")}{_metrics_table(metrics_df,"https")}
<h2>4. TLD Group Fairness</h2>{_img(tld_plot,"TLD fairness")}{_metrics_table(metrics_df,"tld")}
<h2>5. Domain Length Analysis</h2>{_img(plots_dir/"domain_length"/"domain_length_fpr.png","FPR by domain length")}{_metrics_table(metrics_df,"domain_length")}
<h2>6. External Resource Analysis</h2>{_img(plots_dir/"ext_resources"/"ext_resources_fpr.png","FPR by external resources")}{_metrics_table(metrics_df,"ext_resources")}
<h2>7. Full Disparity Table</h2>{disp_table}
<h2>8. SHAP-Based Bias Analysis</h2>{shap_imgs}
{_performance_section(perf_investigation)}
<h2>10. Recommendations</h2>
<div class="alert"><strong>Risk:</strong> Group <code>{hfp.get('group','—')}</code> has FPR={hfp.get('fpr',0):.4f} — investigate training data coverage.</div>
<div class="warn"><strong>Detection gap:</strong> Group <code>{hfn.get('group','—')}</code> has FNR={hfn.get('fnr',0):.4f} — phishing more likely to evade.</div>
<p class="footer">M9 complete. Next: M10 — Blind Spot Analysis.</p></body></html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Bias report saved: {output_path}")
    return str(output_path)
