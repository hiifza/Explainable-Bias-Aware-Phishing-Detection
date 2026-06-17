"""
src/eda/visualization_manager.py
---------------------------------
Centralised manager for all EDA plot saving, style configuration, and
HTML report assembly for Module M2.1.

Responsibilities
----------------
- Apply a consistent matplotlib/seaborn theme project-wide.
- Provide a single `save_figure()` entry point that every other EDA
  module calls — guarantees uniform DPI, bbox, and naming.
- Maintain a registry of every saved figure so the HTML report can
  enumerate them automatically.
- Assemble `outputs/reports/m2_1_eda_report.html` from all results
  produced by the other four analyzers.

Public API
----------
    setup_plot_style()
    save_figure(fig, path, dpi, close)        -> Path
    get_saved_figures()                        -> list[Path]
    clear_registry()
    generate_eda_report(results, output_path) -> str
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Module-level figure registry ─────────────────────────────────────────────
_SAVED_FIGURES: list[Path] = []


# ── Style ─────────────────────────────────────────────────────────────────────

def setup_plot_style() -> None:
    """
    Apply the project-wide matplotlib / seaborn theme.
    Call once at the start of any script or notebook.
    """
    import seaborn as sns
    sns.set_theme(
        style   = "whitegrid",
        palette = "muted",
        font    = "sans-serif",
        font_scale = 1.05,
        rc = {
            "axes.spines.top"   : False,
            "axes.spines.right" : False,
            "figure.facecolor"  : "white",
            "axes.facecolor"    : "white",
            "grid.color"        : "#EBEBEB",
            "grid.linewidth"    : 0.7,
        },
    )
    plt.rcParams.update({
        "figure.dpi"    : 130,
        "savefig.dpi"   : 150,
        "font.size"     : 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    })
    logger.debug("Plot style configured")


# ── Save helper ───────────────────────────────────────────────────────────────

def save_figure(
    fig   : plt.Figure,
    path  : str | Path,
    dpi   : int  = 150,
    close : bool = True,
) -> Path:
    """
    Save *fig* to *path* and register it in the module registry.

    Parameters
    ----------
    fig   : matplotlib Figure
    path  : destination file path (PNG)
    dpi   : output resolution
    close : whether to call plt.close(fig) after saving

    Returns
    -------
    Path — the saved file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    _SAVED_FIGURES.append(path)
    if close:
        plt.close(fig)
    try:
        display_path = path.relative_to(ROOT)
    except ValueError:
        display_path = path
    logger.debug(f"Saved plot → {display_path}")
    return path


def get_saved_figures() -> list[Path]:
    """Return all figures saved in this session."""
    return list(_SAVED_FIGURES)


def clear_registry() -> None:
    """Clear the figure registry (useful between test runs)."""
    global _SAVED_FIGURES
    _SAVED_FIGURES = []


# ── HTML report ───────────────────────────────────────────────────────────────

_CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     font-size:14px;color:#1a1a1a;background:#fff;
     max-width:1140px;margin:40px auto;padding:0 28px 80px}
h1{font-size:22px;font-weight:700;border-bottom:3px solid #185FA5;
   padding-bottom:12px;margin-bottom:6px}
h2{font-size:16px;font-weight:600;color:#185FA5;margin:36px 0 10px;
   border-left:4px solid #185FA5;padding-left:10px}
h3{font-size:14px;font-weight:600;color:#333;margin:18px 0 8px}
p{line-height:1.65;color:#444;margin-bottom:10px}
p.meta{font-size:12px;color:#888;margin-bottom:22px}
table{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0 20px}
th{background:#E6F1FB;padding:8px 13px;text-align:left;
   border:1px solid #B5D4F4;font-weight:600;white-space:nowrap}
td{padding:7px 13px;border:1px solid #ddd;vertical-align:top}
tr:nth-child(even) td{background:#F8FBFF}
code{font-family:"SF Mono",Menlo,monospace;font-size:12px;
     background:#F1F3F5;padding:2px 5px;border-radius:3px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:14px 0 22px}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0 22px}
.grid2{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin:14px 0 22px}
.card{border:1px solid #B5D4F4;border-radius:8px;padding:14px 18px;background:#F0F7FF}
.card-v{font-size:26px;font-weight:700;color:#185FA5}
.card-l{font-size:12px;color:#666;margin-top:3px}
.pass{background:#E1F5EE;color:#0F6E56;padding:3px 8px;border-radius:4px;
      font-size:12px;font-weight:600}
.fail{background:#FCEBEB;color:#A32D2D;padding:3px 8px;border-radius:4px;
      font-size:12px;font-weight:600}
.warn{background:#FAEEDA;color:#854F0B;padding:3px 8px;border-radius:4px;
      font-size:12px;font-weight:600}
.info{background:#E6F1FB;color:#185FA5;padding:3px 8px;border-radius:4px;
      font-size:12px;font-weight:600}
.alert{border-left:4px solid #E24B4A;background:#FCEBEB;
       padding:12px 16px;border-radius:0 6px 6px 0;margin:10px 0}
.alert-warn{border-left:4px solid #EF9F27;background:#FAEEDA;
            padding:12px 16px;border-radius:0 6px 6px 0;margin:10px 0}
.alert-info{border-left:4px solid #378ADD;background:#E6F1FB;
            padding:12px 16px;border-radius:0 6px 6px 0;margin:10px 0}
img{max-width:100%;border:1px solid #ddd;border-radius:6px;
    margin:6px 0;display:block}
.img-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:10px 0}
.footer{font-size:12px;color:#aaa;margin-top:56px;
        border-top:1px solid #eee;padding-top:16px}
.toc a{display:block;padding:3px 0;color:#185FA5;text-decoration:none;font-size:13px}
.toc a:hover{text-decoration:underline}
.finding{background:#F8FBFF;border:1px solid #ddd;border-radius:6px;
         padding:10px 14px;margin:8px 0;font-size:13px}
"""


def _rel(path: Path) -> str:
    """Return a relative path string for embedding in HTML."""
    try:
        return "../../" + str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _img(path: Path, caption: str = "") -> str:
    """Return an HTML <figure> block for a plot."""
    r = _rel(path)
    cap = f"<figcaption style='font-size:11px;color:#888;margin-top:4px'>{caption}</figcaption>" if caption else ""
    return f"<figure><img src='{r}' alt='{caption}' loading='lazy'>{cap}</figure>"


def _overview_section(ov: dict) -> str:
    cd = ov.get("class_distribution", {})
    return f"""
<h2 id='s1'>1. Dataset Overview</h2>
<div class='grid4'>
  <div class='card'><div class='card-v'>{ov.get('n_rows',0):,}</div>
    <div class='card-l'>Rows (post-dedup)</div></div>
  <div class='card'><div class='card-v'>{ov.get('n_features',0)}</div>
    <div class='card-l'>Features (Track B)</div></div>
  <div class='card'><div class='card-v'>{cd.get('phishing_count',0):,}</div>
    <div class='card-l'>Phishing (label=0)</div></div>
  <div class='card'><div class='card-v'>{cd.get('legitimate_count',0):,}</div>
    <div class='card-l'>Legitimate (label=1)</div></div>
</div>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Missing values</td><td>{ov.get('total_missing',0):,}</td></tr>
  <tr><td>Memory usage</td><td>{ov.get('memory_mb',0):.1f} MB</td></tr>
  <tr><td>Numeric features</td><td>{ov.get('n_numeric',0)}</td></tr>
  <tr><td>Categorical features</td><td>{ov.get('n_categorical',0)}</td></tr>
  <tr><td>Binary features</td><td>{ov.get('n_binary',0)}</td></tr>
  <tr><td>Phishing %</td><td>{cd.get('phishing_pct',0):.2f}%</td></tr>
  <tr><td>Legitimate %</td><td>{cd.get('legitimate_pct',0):.2f}%</td></tr>
  <tr><td>Class imbalance ratio</td><td>{cd.get('imbalance_ratio',0):.4f}</td></tr>
</table>
"""


def _leakage_section(lk: dict, plots_dir: Path) -> str:
    usi = lk.get("urlsimilarity", {})
    https = lk.get("https", {})

    usi_plot  = plots_dir / "leakage_urlsimilarity.png"
    http_plot = plots_dir / "leakage_https.png"

    usi_img  = _img(usi_plot,  "URLSimilarityIndex by class") if usi_plot.exists()  else ""
    http_img = _img(http_plot, "IsHTTPS by class")            if http_plot.exists() else ""

    return f"""
<h2 id='s2'>2. Leakage Analysis</h2>

<h3>2a. URLSimilarityIndex — Critical Leakage</h3>
<div class='alert'>
  <strong>CRITICAL:</strong> All legitimate records (label=1) have
  <code>URLSimilarityIndex = 100.0</code> exactly (std=0.0).
  Confirmed: {usi.get('all_legit_100', False)}.
  AUC as separator: <strong>{usi.get('auroc', 0):.6f}</strong>.
  This feature is excluded from Track B entirely.
</div>
{usi_img}
<table>
  <tr><th>Class</th><th>Min</th><th>Mean</th><th>Max</th><th>Std</th></tr>
  <tr><td>Phishing (0)</td>
      <td>{usi.get('phishing_min',0):.2f}</td>
      <td>{usi.get('phishing_mean',0):.2f}</td>
      <td>{usi.get('phishing_max',0):.2f}</td>
      <td>{usi.get('phishing_std',0):.2f}</td></tr>
  <tr><td>Legitimate (1)</td>
      <td>{usi.get('legit_min',0):.2f}</td>
      <td>{usi.get('legit_mean',0):.2f}</td>
      <td>{usi.get('legit_max',0):.2f}</td>
      <td>{usi.get('legit_std',0):.2f}</td></tr>
</table>

<h3>2b. IsHTTPS — Advisory Leakage</h3>
<div class='alert-warn'>
  <strong>ADVISORY:</strong> All legitimate records have IsHTTPS=1.
  {https.get('phishing_https_pct',0):.1f}% of phishing sites also use HTTPS.
  AUC as separator: <strong>{https.get('auroc',0):.6f}</strong>.
  Retained in both tracks; flagged for bias analysis.
</div>
{http_img}
"""


def _correlation_section(corr: dict, plots_dir: Path) -> str:
    top_pos = corr.get("top_positive", pd.DataFrame())
    top_neg = corr.get("top_negative", pd.DataFrame())

    def pair_rows(df: pd.DataFrame) -> str:
        if df is None or len(df) == 0:
            return "<tr><td colspan='3'>—</td></tr>"
        rows = ""
        for _, r in df.head(10).iterrows():
            rows += (f"<tr><td><code>{r.get('feat_A','')}</code></td>"
                     f"<td><code>{r.get('feat_B','')}</code></td>"
                     f"<td>{r.get('pearson_r', r.get('abs_r',0)):.4f}</td></tr>")
        return rows

    heat_p = plots_dir / "correlation_heatmap_pearson.png"
    heat_s = plots_dir / "correlation_heatmap_spearman.png"
    img_p  = _img(heat_p, "Pearson correlation heatmap")   if heat_p.exists()  else ""
    img_s  = _img(heat_s, "Spearman correlation heatmap")  if heat_s.exists()  else ""

    return f"""
<h2 id='s3'>3. Correlation Analysis</h2>
<div class='img-grid'>{img_p}{img_s}</div>

<h3>Top Positive Correlations (Pearson)</h3>
<table>
  <tr><th>Feature A</th><th>Feature B</th><th>r</th></tr>
  {pair_rows(top_pos)}
</table>

<h3>Top Negative Correlations (Pearson)</h3>
<table>
  <tr><th>Feature A</th><th>Feature B</th><th>r</th></tr>
  {pair_rows(top_neg)}
</table>
"""


def _tld_section(tld: dict, plots_dir: Path) -> str:
    tld_plot = plots_dir / "tld_phishing_rate.png"
    img      = _img(tld_plot, "Phishing rate by TLD (top 30)") if tld_plot.exists() else ""
    top_tlds = tld.get("top_tlds", pd.DataFrame())

    def tld_rows(df: pd.DataFrame) -> str:
        if df is None or len(df) == 0:
            return "<tr><td colspan='4'>—</td></tr>"
        rows = ""
        for _, r in df.head(20).iterrows():
            rows += (f"<tr><td><code>{r.get('TLD','')}</code></td>"
                     f"<td>{r.get('count',0):,}</td>"
                     f"<td>{r.get('phishing_count',0):,}</td>"
                     f"<td>{r.get('phishing_rate',0)*100:.1f}%</td></tr>")
        return rows

    return f"""
<h2 id='s4'>4. TLD Analysis</h2>
<p>Total unique TLDs: <strong>{tld.get('n_unique_tlds', 0)}</strong></p>
{img}
<table>
  <tr><th>TLD</th><th>Total</th><th>Phishing</th><th>Phishing Rate</th></tr>
  {tld_rows(top_tlds)}
</table>
"""


def _outlier_section(out: dict, plots_dir: Path) -> str:
    out_df   = out.get("outlier_stats", pd.DataFrame())
    out_plot = plots_dir / "outlier_boxplots.png"
    img      = _img(out_plot, "Outlier boxplots for count features") if out_plot.exists() else ""

    def out_rows(df: pd.DataFrame) -> str:
        if df is None or len(df) == 0:
            return "<tr><td colspan='6'>—</td></tr>"
        rows = ""
        for _, r in df.iterrows():
            rows += (f"<tr><td><code>{r.get('feature','')}</code></td>"
                     f"<td>{r.get('q1',0):.1f}</td>"
                     f"<td>{r.get('median',0):.1f}</td>"
                     f"<td>{r.get('q3',0):.1f}</td>"
                     f"<td>{r.get('p99',0):.1f}</td>"
                     f"<td>{r.get('max',0):.1f}</td></tr>")
        return rows

    return f"""
<h2 id='s5'>5. Outlier Analysis</h2>
{img}
<table>
  <tr><th>Feature</th><th>Q1</th><th>Median</th><th>Q3</th><th>P99</th><th>Max</th></tr>
  {out_rows(out_df)}
</table>
"""


def _prescreening_section(ps: dict, plots_dir: Path) -> str:
    mi_df   = ps.get("mutual_information", pd.DataFrame())
    mi_plot = plots_dir / "prescreening_mutual_info.png"
    img     = _img(mi_plot, "Mutual information scores (Track B)") if mi_plot.exists() else ""

    def ps_rows(df: pd.DataFrame) -> str:
        if df is None or len(df) == 0:
            return "<tr><td colspan='4'>—</td></tr>"
        rows = ""
        for _, r in df.head(20).iterrows():
            rows += (f"<tr><td><code>{r.get('feature','')}</code></td>"
                     f"<td>{r.get('mi_score',0):.6f}</td>"
                     f"<td>{r.get('anova_f',float('nan')):.2f}</td>"
                     f"<td>{r.get('mi_rank',0)}</td></tr>")
        return rows

    return f"""
<h2 id='s6'>6. Feature Importance Pre-Screening</h2>
{img}
<table>
  <tr><th>Feature</th><th>MI Score</th><th>ANOVA F</th><th>MI Rank</th></tr>
  {ps_rows(mi_df)}
</table>
"""


def _findings_section(results: dict) -> str:
    lk    = results.get("leakage", {})
    corr  = results.get("correlation", {})
    tld   = results.get("tld", {})
    out   = results.get("outliers", {})

    usi_auroc   = lk.get("urlsimilarity", {}).get("auroc", 0)
    https_auroc = lk.get("https", {}).get("auroc", 0)
    n_high_corr = corr.get("n_high_pairs", 0)
    n_unique_tld= tld.get("n_unique_tlds", 0)
    n_outlier_f = out.get("n_outlier_features", 0)

    return f"""
<h2 id='s0'>Executive Summary — Key Findings</h2>
<div class='finding'>
  <strong>🔴 Critical Leakage:</strong> <code>URLSimilarityIndex</code> achieves
  AUROC={usi_auroc:.4f} as a solo predictor. All legitimate records = 100.0 exactly.
  Excluded from Track B.
</div>
<div class='finding'>
  <strong>🟡 Advisory Leakage:</strong> <code>IsHTTPS</code> achieves
  AUROC={https_auroc:.4f}. All legitimate records have IsHTTPS=1. Retained but
  flagged for bias analysis.
</div>
<div class='finding'>
  <strong>🔵 Correlation:</strong> {n_high_corr} feature pairs with |r| ≥ 0.75
  in the retained Track B feature set. All retained (tree models are robust).
</div>
<div class='finding'>
  <strong>🌐 TLD Diversity:</strong> {n_unique_tld} unique TLDs with extreme
  phishing-rate variation by TLD group — bias analysis required in M9.1.
</div>
<div class='finding'>
  <strong>📊 Outliers:</strong> {n_outlier_f} count features have extreme right-skew
  (max &gt; 100× median). Log1p transform required in preprocessing (M3.1).
</div>
"""


def generate_eda_report(
    results    : dict,
    output_path: str | Path = "outputs/reports/m2_1_eda_report.html",
    plots_dir  : str | Path = "outputs/plots/eda",
) -> str:
    """
    Assemble the complete M2.1 EDA HTML report from all analyzer results.

    Parameters
    ----------
    results     : dict with keys: overview, leakage, correlation, tld, outliers, prescreening
    output_path : destination HTML file
    plots_dir   : directory where EDA plots are saved

    Returns
    -------
    str — absolute path to the written HTML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir   = Path(plots_dir)
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ov   = results.get("overview",      {})
    lk   = results.get("leakage",       {})
    corr = results.get("correlation",   {})
    tld  = results.get("tld",           {})
    out  = results.get("outliers",      {})
    ps   = results.get("prescreening",  {})

    toc = """
<nav class='toc' style='background:#F8FBFF;border:1px solid #ddd;
     border-radius:6px;padding:14px 18px;margin:20px 0;max-width:400px'>
  <strong style='font-size:13px'>Table of Contents</strong>
  <a href='#s0'>Executive Summary</a>
  <a href='#s1'>1. Dataset Overview</a>
  <a href='#s2'>2. Leakage Analysis</a>
  <a href='#s3'>3. Correlation Analysis</a>
  <a href='#s4'>4. TLD Analysis</a>
  <a href='#s5'>5. Outlier Analysis</a>
  <a href='#s6'>6. Feature Pre-Screening</a>
</nav>"""

    body = (
        toc
        + _findings_section(results)
        + _overview_section(ov)
        + _leakage_section(lk, plots_dir)
        + _correlation_section(corr, plots_dir)
        + _tld_section(tld, plots_dir)
        + _outlier_section(out, plots_dir)
        + _prescreening_section(ps, plots_dir)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>M2.1 EDA Report — Phishing Detection</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Module M2.1 — Exploratory Data Analysis</h1>
<p class='meta'>
  Project: Explainable and Bias-Aware ML for Phishing Website Detection
  &nbsp;|&nbsp; Generated: {timestamp}
  &nbsp;|&nbsp; Dataset: PhiUSIIL &nbsp;|&nbsp;
  Rows: {ov.get('n_rows',0):,} &nbsp;|&nbsp;
  Track B: {ov.get('n_features',0)} features
</p>
{body}
<p class='footer'>M2.1 complete. Next: M3.1 — Preprocessing Pipeline.</p>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info(f"M2.1 EDA report saved: {output_path}")
    return str(output_path)
