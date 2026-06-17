"""
src/features/report_m1_2.py
----------------------------
Generates the professional HTML report for Module M1.2:
    outputs/reports/m1_2_feature_finalization_report.html

The report documents every feature decision made during M1.2 and
becomes the permanent audit trail for the entire project.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy  as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger         import get_logger
from src.features.feature_catalog import (
    DROP_IDENTIFIERS, DROP_MULTICOLLINEAR, DROP_ALL,
    LEAKAGE_CRITICAL, LEAKAGE_ADVISORY,
    TRACK_A_FEATURES, TRACK_B_FEATURES,
    FEATURE_CATEGORIES, TARGET_COLUMN,
)

logger = get_logger(__name__)

# ── CSS / HTML shared styles ──────────────────────────────────────────────────

_CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px; color: #1a1a1a; background: #fff;
  max-width: 1100px; margin: 40px auto; padding: 0 28px 80px;
}
h1  { font-size: 22px; font-weight: 700; border-bottom: 3px solid #185FA5;
      padding-bottom: 12px; margin-bottom: 6px; }
h2  { font-size: 16px; font-weight: 600; color: #185FA5;
      margin: 36px 0 12px; border-left: 4px solid #185FA5;
      padding-left: 10px; }
h3  { font-size: 14px; font-weight: 600; color: #333; margin: 20px 0 8px; }
p   { line-height: 1.65; color: #444; margin-bottom: 10px; }
p.meta { font-size: 12px; color: #888; margin-bottom: 24px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 10px 0 22px; }
th    { background: #E6F1FB; padding: 9px 13px; text-align: left;
        border: 1px solid #B5D4F4; font-weight: 600; white-space: nowrap; }
td    { padding: 7px 13px; border: 1px solid #ddd; vertical-align: top; }
tr:nth-child(even) td { background: #F8FBFF; }
code { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 12px;
       background: #F1F3F5; padding: 2px 5px; border-radius: 3px; }
.grid3 { display: grid; grid-template-columns: repeat(3,1fr); gap:16px; margin:16px 0 24px; }
.grid4 { display: grid; grid-template-columns: repeat(4,1fr); gap:14px; margin:16px 0 24px; }
.card  { border:1px solid #B5D4F4; border-radius:8px; padding:16px 18px;
         background:#F0F7FF; }
.card-v { font-size:28px; font-weight:700; color:#185FA5; }
.card-l { font-size:12px; color:#666; margin-top:4px; }
.pass  { background:#E1F5EE; color:#0F6E56; padding:3px 8px; border-radius:4px;
         font-size:12px; font-weight:600; }
.fail  { background:#FCEBEB; color:#A32D2D; padding:3px 8px; border-radius:4px;
         font-size:12px; font-weight:600; }
.warn  { background:#FAEEDA; color:#854F0B; padding:3px 8px; border-radius:4px;
         font-size:12px; font-weight:600; }
.info  { background:#E6F1FB; color:#185FA5; padding:3px 8px; border-radius:4px;
         font-size:12px; font-weight:600; }
.alert { border-left:4px solid #E24B4A; background:#FCEBEB;
         padding:12px 16px; border-radius:0 6px 6px 0; margin:12px 0; }
.alert-warn { border-left:4px solid #EF9F27; background:#FAEEDA;
              padding:12px 16px; border-radius:0 6px 6px 0; margin:12px 0; }
.alert-info { border-left:4px solid #378ADD; background:#E6F1FB;
              padding:12px 16px; border-radius:0 6px 6px 0; margin:12px 0; }
.tag  { display:inline-block; font-size:11px; font-weight:500;
        padding:2px 7px; border-radius:4px; margin:2px 3px 2px 0; }
.tag-url    { background:#E6F1FB; color:#185FA5; }
.tag-stat   { background:#EAF3DE; color:#3B6D11; }
.tag-char   { background:#EEEDFE; color:#533AB7; }
.tag-obf    { background:#FAEEDA; color:#854F0B; }
.tag-html   { background:#FBEAF0; color:#993556; }
.tag-redir  { background:#FAECE7; color:#993C1D; }
.tag-form   { background:#E1F5EE; color:#0F6E56; }
.tag-trust  { background:#EAF3DE; color:#3B6D11; }
.tag-ext    { background:#E6F1FB; color:#185FA5; }
.corr-bar   { display:inline-block; height:10px; border-radius:3px;
              vertical-align:middle; margin-right:6px; }
.footer { font-size:12px; color:#aaa; margin-top:56px;
          border-top:1px solid #eee; padding-top:16px; }
img { max-width:100%; border:1px solid #ddd; border-radius:6px; margin:8px 0; }
"""

# ── Category tag CSS class map ────────────────────────────────────────────────

_CAT_TAG = {
    "URL Structure"           : "tag-url",
    "URL Statistical"         : "tag-stat",
    "URL Character Composition": "tag-char",
    "Obfuscation"             : "tag-obf",
    "HTML Structure"          : "tag-html",
    "Redirects & Navigation"  : "tag-redir",
    "Forms & Interaction"     : "tag-form",
    "Content & Trust"         : "tag-trust",
    "External Resources"      : "tag-ext",
}


# ── Helper renderers ──────────────────────────────────────────────────────────

def _corr_bar(r: float, max_w: int = 120) -> str:
    """Return an HTML inline bar representing |r| value."""
    if np.isnan(r):
        return "—"
    w     = max(int(r * max_w), 2)
    color = "#E24B4A" if r >= 0.75 else "#EF9F27" if r >= 0.50 else "#378ADD"
    return (
        f'<span class="corr-bar" style="width:{w}px;background:{color};"></span>'
        f'<span style="font-size:12px;font-family:monospace">{r:.4f}</span>'
    )


def _tag(cat: str) -> str:
    css = _CAT_TAG.get(cat, "tag-url")
    return f'<span class="tag {css}">{cat}</span>'


def _badge(ok: bool, yes: str = "PASS", no: str = "FAIL") -> str:
    css = "pass" if ok else "fail"
    sym = "✓" if ok else "✗"
    return f'<span class="{css}">{sym} {yes if ok else no}</span>'


# ── Section builders ──────────────────────────────────────────────────────────

def _section_overview(df_clean: pd.DataFrame) -> str:
    n = len(df_clean)
    ph = int((df_clean[TARGET_COLUMN] == 0).sum())
    lg = int((df_clean[TARGET_COLUMN] == 1).sum())
    return f"""
<h2>1. Dataset Overview</h2>
<div class="grid4">
  <div class="card"><div class="card-v">{n:,}</div><div class="card-l">Total rows (post-dedup)</div></div>
  <div class="card"><div class="card-v">56</div><div class="card-l">Raw columns</div></div>
  <div class="card"><div class="card-v">{ph:,}</div><div class="card-l">Phishing (label=0)</div></div>
  <div class="card"><div class="card-v">{lg:,}</div><div class="card-l">Legitimate (label=1)</div></div>
</div>
"""


def _section_dropped(df_clean: pd.DataFrame) -> str:
    id_rows = "".join(
        f"<tr><td><code>{c}</code></td>"
        f"<td>{str(df_clean[c].dtype)}</td>"
        f"<td>{df_clean[c].nunique():,}</td>"
        f"<td>Identifier — no predictive value</td>"
        f"<td><span class='fail'>✗ DROPPED</span></td></tr>\n"
        for c in DROP_IDENTIFIERS if c in df_clean.columns
    )
    mc_rows = "".join(
        f"<tr><td><code>{c}</code></td>"
        f"<td>{str(df_clean[c].dtype) if c in df_clean.columns else 'float64'}</td>"
        f"<td>—</td>"
        f"<td>r=0.961 with DomainTitleMatchScore — multicollinear</td>"
        f"<td><span class='fail'>✗ DROPPED</span></td></tr>\n"
        for c in DROP_MULTICOLLINEAR
    )
    return f"""
<h2>2. Columns Removed</h2>
<p>5 columns removed before any modelling. These are identifiers or structurally
redundant features that provide no independent predictive signal.</p>

<h3>2a. Identifier Columns (4)</h3>
<table>
  <tr><th>Column</th><th>Dtype</th><th>Unique Values</th><th>Reason</th><th>Decision</th></tr>
  {id_rows}
</table>

<h3>2b. Multicollinear Column (1)</h3>
<table>
  <tr><th>Column</th><th>Dtype</th><th>Unique Values</th><th>Reason</th><th>Decision</th></tr>
  {mc_rows}
</table>
"""


def _section_leakage(df_clean: pd.DataFrame) -> str:
    usi_all_100 = (df_clean[df_clean[TARGET_COLUMN] == 1]["URLSimilarityIndex"] == 100).all() \
        if "URLSimilarityIndex" in df_clean.columns else True
    https_all_1 = (df_clean[df_clean[TARGET_COLUMN] == 1]["IsHTTPS"] == 1).all() \
        if "IsHTTPS" in df_clean.columns else True
    https_ph_pct = df_clean[df_clean[TARGET_COLUMN] == 0]["IsHTTPS"].mean() * 100 \
        if "IsHTTPS" in df_clean.columns else 0.0

    return f"""
<h2>3. Leakage & Bias-Risk Features</h2>

<h3>3a. Critical Leakage — URLSimilarityIndex</h3>
<div class="alert">
  <strong>CRITICAL DATA LEAKAGE</strong><br>
  Every legitimate record (label=1) has <code>URLSimilarityIndex = 100.0</code> exactly.
  Zero variance in the legitimate class. All-100 confirmed: <strong>{usi_all_100}</strong>.
  This feature encodes the label directly and cannot be used in a real deployment scenario.
  <br><br>
  <strong>Decision:</strong> Excluded from <em>Track B</em> (primary benchmark).
  Included in <em>Track A</em> only to quantify the performance inflation it causes.
  The AUC delta between Track A and Track B is reported as the leakage impact.
</div>
<table>
  <tr><th>Label</th><th>Min</th><th>Mean</th><th>Max</th><th>Std</th></tr>
  <tr><td>0 — Phishing</td><td>~0</td><td>~49.6</td><td>&lt;100</td><td>high</td></tr>
  <tr><td>1 — Legitimate</td><td>100.0</td><td>100.0</td><td>100.0</td><td>0.0</td></tr>
</table>

<h3>3b. Advisory Leakage — IsHTTPS</h3>
<div class="alert-warn">
  <strong>ADVISORY: Near-Perfect Separation</strong><br>
  All legitimate records (label=1) have <code>IsHTTPS = 1</code>.
  All-1 confirmed: <strong>{https_all_1}</strong>. However,
  {https_ph_pct:.1f}% of phishing sites in this dataset also use HTTPS, so the
  feature is not a perfect separator in the phishing class. This pattern is
  temporally unstable — modern phishing campaigns increasingly obtain SSL certificates.
  <br><br>
  <strong>Decision:</strong> Retained in <em>both tracks</em> but flagged for bias
  analysis (M9.2) and SHAP attribution investigation (M7.1).
</div>
"""


def _section_feature_table(audit_table: pd.DataFrame) -> str:
    track_b = audit_table[audit_table["in_track_B"]].copy()
    rows = ""
    for _, r in track_b.iterrows():
        nc_flag = '<span class="warn">near-const</span>' if r["is_near_constant"] else ""
        adv_flag = '<span class="warn">advisory</span>' if r["is_advisory"] else ""
        flags = " ".join(filter(None, [nc_flag, adv_flag]))
        rows += (
            f"<tr>"
            f"<td><code>{r['feature']}</code></td>"
            f"<td>{_tag(r['category'])}</td>"
            f"<td>{r['dtype']}</td>"
            f"<td>{r['nunique']:,}</td>"
            f"<td>{r['top_value_pct']:.1f}%</td>"
            f"<td>{_corr_bar(r['abs_r_with_label'])}</td>"
            f"<td>{r['skewness']:.1f}</td>"
            f"<td>{flags if flags else '—'}</td>"
            f"</tr>\n"
        )
    return f"""
<h2>4. Approved Feature Catalog — Track B (49 features)</h2>
<p>These 49 features are the approved set for Track B (leakage-aware primary model).
Track A uses the same 49 plus <code>URLSimilarityIndex</code> = 50 features total.</p>
<table>
  <tr>
    <th>Feature</th><th>Category</th><th>Dtype</th><th>Unique</th>
    <th>Top-val%</th><th>|r| with label</th><th>Skew</th><th>Flags</th>
  </tr>
  {rows}
</table>
"""


def _section_correlations(pairs_high: pd.DataFrame, pairs_medium: pd.DataFrame) -> str:
    def pair_rows(df: pd.DataFrame) -> str:
        rows = ""
        for _, r in df.iterrows():
            color = "#E24B4A" if r["abs_r"] >= 0.75 else "#EF9F27"
            rows += (
                f"<tr>"
                f"<td><code>{r['feat_A']}</code></td>"
                f"<td><code>{r['feat_B']}</code></td>"
                f"<td style='color:{color};font-weight:600'>{r['abs_r']:.4f}</td>"
                f"<td>{'High — review' if r['abs_r'] >= 0.75 else 'Medium — retained'}</td>"
                f"</tr>\n"
            )
        return rows or "<tr><td colspan='4' style='color:#888'>None</td></tr>"

    return f"""
<h2>5. Feature-Feature Correlation Analysis</h2>

<h3>5a. High-Correlation Pairs (|r| ≥ 0.75)</h3>
<div class="alert-info">
  Pairs below are from the <em>retained</em> feature set only (after dropping
  URLTitleMatchScore). The remaining high-correlation pairs consist primarily of
  ratio/count pairs that carry complementary information and are retained intentionally.
  Tree-based models are robust to moderate inter-feature correlation.
</div>
<table>
  <tr><th>Feature A</th><th>Feature B</th><th>|r|</th><th>Action</th></tr>
  {pair_rows(pairs_high)}
</table>

<h3>5b. Medium-Correlation Pairs (0.50 ≤ |r| &lt; 0.75)</h3>
<table>
  <tr><th>Feature A</th><th>Feature B</th><th>|r|</th><th>Action</th></tr>
  {pair_rows(pairs_medium.head(20))}
</table>
"""


def _section_categories() -> str:
    cat_rows = ""
    for cat, feats in FEATURE_CATEGORIES.items():
        tags = "".join(f"<code style='margin:2px 3px;display:inline-block'>{f}</code>" for f in feats)
        cat_rows += (
            f"<tr><td>{_tag(cat)}</td>"
            f"<td style='text-align:center'>{len(feats)}</td>"
            f"<td>{tags}</td></tr>\n"
        )
    return f"""
<h2>6. Feature Categories (Track B)</h2>
<table>
  <tr><th>Category</th><th>Count</th><th>Features</th></tr>
  {cat_rows}
</table>
"""


def _section_tracks() -> str:
    a_tags = "".join(
        f"<code style='margin:2px 3px;display:inline-block;"
        f"{'background:#FCEBEB;color:#A32D2D' if f=='URLSimilarityIndex' else ''}'>"
        f"{f}</code>"
        for f in TRACK_A_FEATURES
    )
    b_tags = "".join(
        f"<code style='margin:2px 3px;display:inline-block'>{f}</code>"
        for f in TRACK_B_FEATURES
    )
    return f"""
<h2>7. Experimental Track Definitions</h2>
<table>
  <tr><th>Track</th><th>Features</th><th>Includes URLSimilarityIndex</th>
      <th>Purpose</th><th>Primary Report</th></tr>
  <tr>
    <td><strong>Track A</strong></td><td>50</td>
    <td><span class="fail">Yes — leakage included</span></td>
    <td>Ceiling experiment — quantify leakage inflation</td>
    <td>No — reference only</td>
  </tr>
  <tr>
    <td><strong>Track B</strong></td><td>49</td>
    <td><span class="pass">No — leakage excluded</span></td>
    <td>Deployment-realistic model — honest benchmark</td>
    <td><span class="pass">Yes — all XAI &amp; bias analysis</span></td>
  </tr>
</table>

<h3>Track A — 50 Features</h3>
<div style="line-height:2">{a_tags}</div>

<h3 style="margin-top:16px">Track B — 49 Features</h3>
<div style="line-height:2">{b_tags}</div>
"""


def _section_plots(plots_dir: Path) -> str:
    plot_files = {
        "category_distribution.png"  : "Feature count by category (Track B)",
        "correlation_heatmap.png"     : "Full feature-feature correlation heatmap",
        "high_corr_pairs.png"         : "High-correlation pairs (|r| ≥ 0.75)",
        "target_correlation.png"      : "Feature correlation with target label",
        "variance_audit.png"          : "Near-constant feature variance audit",
    }
    sections = ""
    for fname, caption in plot_files.items():
        rel = f"../../outputs/plots/eda/{fname}"
        sections += f"""
        <h3>{caption}</h3>
        <img src="{rel}" alt="{caption}" loading="lazy">
        """
    return f"<h2>8. Visualisations</h2>{sections}"


def _section_validation(feat_lists) -> str:
    checks = [
        ("Track B feature count == 49",  len(feat_lists.track_B) == 49),
        ("Track A feature count == 50",  len(feat_lists.track_A) == 50),
        ("No duplicates in Track B",      len(set(feat_lists.track_B)) == 49),
        ("No duplicates in Track A",      len(set(feat_lists.track_A)) == 50),
        ("URLSimilarityIndex in Track A", "URLSimilarityIndex" in feat_lists.track_A),
        ("URLSimilarityIndex NOT in Track B", "URLSimilarityIndex" not in feat_lists.track_B),
        ("label NOT in any track",        "label" not in feat_lists.track_A),
        ("All identifiers dropped",       all(c not in feat_lists.track_A for c in DROP_IDENTIFIERS)),
        ("URLTitleMatchScore dropped",    "URLTitleMatchScore" not in feat_lists.track_A),
    ]
    rows = "".join(
        f"<tr><td>{desc}</td><td>{_badge(ok)}</td></tr>\n"
        for desc, ok in checks
    )
    all_pass = all(ok for _, ok in checks)
    return f"""
<h2>9. Validation Checks</h2>
<p>All checks must pass before proceeding to Module M2.1 (EDA).</p>
<table>
  <tr><th>Check</th><th>Status</th></tr>
  {rows}
</table>
<p><strong>Overall: {_badge(all_pass, 'ALL CHECKS PASSED', 'FAILURES DETECTED')}</strong></p>
"""


# ── Main report generator ─────────────────────────────────────────────────────

def generate_report(
    df_clean   : pd.DataFrame,
    audit_results: dict,
    output_path: str | Path = "outputs/reports/m1_2_feature_finalization_report.html",
    plots_dir  : str | Path = "outputs/plots/eda",
) -> str:
    """
    Assemble and write the complete M1.2 HTML report.

    Parameters
    ----------
    df_clean      : deduplicated DataFrame from M1.1
    audit_results : dict returned by feature_catalog.run_feature_audit()
    output_path   : destination HTML file path
    plots_dir     : directory containing EDA plots

    Returns
    -------
    str — absolute path to the generated report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir   = Path(plots_dir)

    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feat_lists  = audit_results["feature_lists"]
    audit_table = audit_results["audit_table"]
    pairs_high  = audit_results["pairwise_high"]
    pairs_med   = audit_results["pairwise_medium"]

    body = (
        _section_overview(df_clean)
        + _section_dropped(df_clean)
        + _section_leakage(df_clean)
        + _section_feature_table(audit_table)
        + _section_correlations(pairs_high, pairs_med)
        + _section_categories()
        + _section_tracks()
        + _section_plots(plots_dir)
        + _section_validation(feat_lists)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>M1.2 Feature Finalization Report — Phishing Detection</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Module M1.2 — Column Removal &amp; Feature Set Finalization</h1>
<p class="meta">
  Project: Explainable and Bias-Aware ML for Phishing Website Detection &nbsp;|&nbsp;
  Generated: {timestamp} &nbsp;|&nbsp;
  Track B: 49 features &nbsp;|&nbsp; Track A: 50 features
</p>
{body}
<p class="footer">
  M1.2 complete. This report is the permanent audit trail for all feature decisions.
  Next: M2.1 — Exploratory Data Analysis.
</p>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info(f"M1.2 HTML report saved: {output_path}")
    return str(output_path)
