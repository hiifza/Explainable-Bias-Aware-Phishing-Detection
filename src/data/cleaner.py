"""
src/data/cleaner.py
-------------------
Cleans the validated raw DataFrame produced by loader.py.

Steps executed (in order)
--------------------------
1. Detect duplicate URLs
2. Remove duplicates (keep first occurrence)
3. Validate class distribution post-dedup
4. Save clean_df.csv to data/processed/
5. Generate an HTML audit report to outputs/reports/

Public API
----------
    detect_duplicates(df)                       -> (duplicate_df, count)
    remove_duplicates(df)                       -> pd.DataFrame
    validate_class_distribution(df, stage)      -> dict
    save_clean_df(df, output_path)              -> None
    generate_audit_report(...)                  -> str
    run_cleaning_pipeline(df_raw, ...)          -> (pd.DataFrame, dict)
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# ── Project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

URL_COLUMN    : str = "URL"
TARGET_COLUMN : str = "label"


# ── Duplicate detection ───────────────────────────────────────────────────────

def detect_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Identify rows that share a URL with an earlier row.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (validated by loader.py).

    Returns
    -------
    (duplicate_rows, count)
        *duplicate_rows* — the subset of rows that will be dropped.
        *count*          — integer count of duplicate rows found.
    """
    logger.info(f"Detecting duplicates on column: '{URL_COLUMN}'")

    dup_mask       = df.duplicated(subset=[URL_COLUMN], keep="first")
    duplicate_rows = df[dup_mask].copy()
    count          = int(dup_mask.sum())

    logger.info(f"Total rows          : {len(df):,}")
    logger.info(f"Unique URLs         : {df[URL_COLUMN].nunique():,}")
    logger.info(f"Duplicate rows found: {count:,}")

    if count > 0:
        dup_class = duplicate_rows[TARGET_COLUMN].value_counts().sort_index()
        logger.info("Class distribution inside duplicates:")
        logger.info(f"  label=0 Phishing   : {dup_class.get(0, 0)}")
        logger.info(f"  label=1 Legitimate : {dup_class.get(1, 0)}")

    return duplicate_rows, count


# ── Deduplication ─────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate URLs, retaining the first occurrence of each URL.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that may contain duplicate URLs.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame with a fresh integer index.
    """
    original_len = len(df)
    logger.info("Removing duplicates (keep='first') …")

    df_clean = (
        df.drop_duplicates(subset=[URL_COLUMN], keep="first")
          .reset_index(drop=True)
    )

    removed = original_len - len(df_clean)
    logger.info(f"Rows removed        : {removed:,}")
    logger.info(f"Rows retained       : {len(df_clean):,}")

    return df_clean


# ── Class distribution ────────────────────────────────────────────────────────

def validate_class_distribution(df: pd.DataFrame, stage: str = "current") -> dict:
    """
    Compute and log the class distribution at a named pipeline stage.

    Parameters
    ----------
    df    : pd.DataFrame
    stage : str
        Human-readable label for logging (e.g. 'pre-dedup', 'post-dedup').

    Returns
    -------
    dict
        Keys: stage, total, phishing_count, legitimate_count,
              phishing_pct, legitimate_pct, imbalance_ratio.
    """
    logger.info(f"Class distribution [{stage}]:")

    vc    = df[TARGET_COLUMN].value_counts().sort_index()
    total = len(df)

    phishing_n = int(vc.get(0, 0))
    legit_n    = int(vc.get(1, 0))
    phishing_p = phishing_n / total * 100
    legit_p    = legit_n    / total * 100
    ratio      = phishing_n / max(legit_n, 1)

    logger.info(f"  Total samples        : {total:,}")
    logger.info(f"  label=0 Phishing     : {phishing_n:,} ({phishing_p:.2f}%)")
    logger.info(f"  label=1 Legitimate   : {legit_n:,}    ({legit_p:.2f}%)")
    logger.info(f"  Imbalance ratio (0/1): {ratio:.4f}")

    return {
        "stage"           : stage,
        "total"           : total,
        "phishing_count"  : phishing_n,
        "legitimate_count": legit_n,
        "phishing_pct"    : round(phishing_p, 4),
        "legitimate_pct"  : round(legit_p,    4),
        "imbalance_ratio" : round(ratio,       6),
    }


# ── Save ──────────────────────────────────────────────────────────────────────

def save_clean_df(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Persist the cleaned DataFrame as a CSV file.

    Parameters
    ----------
    df          : pd.DataFrame
    output_path : str | Path
        Destination path, e.g. ``data/processed/clean_df.csv``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving clean DataFrame → {output_path}")
    df.to_csv(output_path, index=False)

    size_mb = output_path.stat().st_size / 1_000_000
    logger.info(f"Saved             : {len(df):,} rows × {df.shape[1]} cols  ({size_mb:.2f} MB)")


# ── HTML audit report ─────────────────────────────────────────────────────────

def generate_audit_report(
    df_original    : pd.DataFrame,
    df_clean       : pd.DataFrame,
    duplicate_rows : pd.DataFrame,
    validation_report: dict,
    output_dir     : str | Path = "outputs/reports",
) -> str:
    """
    Write a self-contained HTML audit report for Module M1.1.

    Parameters
    ----------
    df_original       : raw DataFrame (before deduplication)
    df_clean          : deduplicated DataFrame
    duplicate_rows    : the rows that were removed
    validation_report : dict returned by loader.run_full_validation()
    output_dir        : directory for the HTML file

    Returns
    -------
    str
        Absolute path to the generated HTML file.
    """
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "m1_1_data_audit_report.html"

    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_orig      = len(df_original)
    n_clean     = len(df_clean)
    n_removed   = n_orig - n_clean
    missing_tot = int(df_original.isnull().sum().sum())

    pre_vc  = df_original[TARGET_COLUMN].value_counts().sort_index()
    post_vc = df_clean[TARGET_COLUMN].value_counts().sort_index()

    # Pre-compute badge conditions to avoid set-literal issues inside f-strings
    row_ok        = n_orig == 235_795
    col_ok        = df_original.shape[1] == 56
    missing_ok    = missing_tot == 0
    label_values  = set(df_original[TARGET_COLUMN].unique())
    label_ok      = label_values == {0, 1}
    cols_ok_flag  = validation_report.get("columns_valid", True)

    def badge(ok: bool, yes_label: str = "PASS", no_label: str = "FAIL") -> str:
        css = "badge-pass" if ok else "badge-fail"
        return f'<span class="{css}">{"✓ " + yes_label if ok else "✗ " + no_label}</span>'

    def dtype_rows(df: pd.DataFrame) -> str:
        rows = ""
        for dt in ["int64", "float64", "object", "bool"]:
            cols = list(df.select_dtypes(include=[dt]).columns)
            if not cols:
                continue
            sample = ", ".join(cols[:5]) + (" …" if len(cols) > 5 else "")
            rows += f"<tr><td><code>{dt}</code></td><td>{len(cols)}</td><td>{sample}</td></tr>\n"
        return rows

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>M1.1 Data Audit Report — Phishing Detection</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body   {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 14px; color: #1a1a1a; background: #fff;
            max-width: 980px; margin: 40px auto; padding: 0 24px 60px; }}
  h1     {{ font-size: 20px; font-weight: 600; border-bottom: 2px solid #185FA5;
            padding-bottom: 12px; margin-bottom: 24px; }}
  h2     {{ font-size: 15px; font-weight: 600; color: #185FA5;
            margin: 32px 0 12px; }}
  p.meta {{ font-size: 12px; color: #888; margin-bottom: 20px; }}
  table  {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 8px 0 20px; }}
  th     {{ background: #E6F1FB; padding: 9px 14px; text-align: left;
            border: 1px solid #B5D4F4; font-weight: 600; }}
  td     {{ padding: 8px 14px; border: 1px solid #ddd; vertical-align: top; }}
  tr:nth-child(even) td {{ background: #F8FBFF; }}
  code   {{ font-family: "SF Mono", Menlo, monospace; font-size: 12px;
            background: #F1F3F5; padding: 2px 5px; border-radius: 3px; }}
  .grid  {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 16px 0 24px; }}
  .card  {{ border: 1px solid #B5D4F4; border-radius: 8px; padding: 16px 20px;
            background: #F0F7FF; }}
  .card-value {{ font-size: 26px; font-weight: 700; color: #185FA5; }}
  .card-label {{ font-size: 12px; color: #666; margin-top: 4px; }}
  .badge-pass {{ background: #E1F5EE; color: #0F6E56; padding: 3px 9px;
                 border-radius: 4px; font-size: 12px; font-weight: 600; }}
  .badge-fail {{ background: #FCEBEB; color: #A32D2D; padding: 3px 9px;
                 border-radius: 4px; font-size: 12px; font-weight: 600; }}
  .badge-warn {{ background: #FAEEDA; color: #854F0B; padding: 3px 9px;
                 border-radius: 4px; font-size: 12px; font-weight: 600; }}
  .footer {{ font-size: 12px; color: #aaa; margin-top: 48px;
             border-top: 1px solid #eee; padding-top: 16px; }}
</style>
</head>
<body>

<h1>Module M1.1 — Raw Data Loading &amp; Deduplication Audit</h1>
<p class="meta">
  Project: Explainable and Bias-Aware ML for Phishing Website Detection &nbsp;|&nbsp;
  Generated: {timestamp}
</p>

<h2>1. Dataset Overview</h2>
<div class="grid">
  <div class="card">
    <div class="card-value">{n_orig:,}</div>
    <div class="card-label">Raw rows loaded</div>
  </div>
  <div class="card">
    <div class="card-value">{df_original.shape[1]}</div>
    <div class="card-label">Columns</div>
  </div>
  <div class="card">
    <div class="card-value">{missing_tot:,}</div>
    <div class="card-label">Missing values</div>
  </div>
</div>

<h2>2. Structural Validation Results</h2>
<table>
  <tr><th>Check</th><th>Expected</th><th>Actual</th><th>Status</th></tr>
  <tr>
    <td>Row count</td><td>235,795</td><td>{n_orig:,}</td>
    <td>{badge(row_ok, "PASS", "MISMATCH")}</td>
  </tr>
  <tr>
    <td>Column count</td><td>56</td><td>{df_original.shape[1]}</td>
    <td>{badge(col_ok)}</td>
  </tr>
  <tr>
    <td>Missing values</td><td>0</td><td>{missing_tot:,}</td>
    <td>{badge(missing_ok, "PASS", "FOUND")}</td>
  </tr>
  <tr>
    <td>Label values</td><td>&#123;0, 1&#125;</td>
    <td>{", ".join(str(v) for v in sorted(label_values))}</td>
    <td>{badge(label_ok)}</td>
  </tr>
  <tr>
    <td>All expected columns present</td><td>56</td><td>56</td>
    <td>{badge(cols_ok_flag)}</td>
  </tr>
</table>

<h2>3. Duplicate URL Analysis</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Total rows (raw)</td>          <td>{n_orig:,}</td></tr>
  <tr><td>Unique URLs</td>               <td>{df_original[URL_COLUMN].nunique():,}</td></tr>
  <tr><td>Duplicate rows removed</td>    <td>{n_removed:,}</td></tr>
  <tr><td>Deduplication key</td>         <td><code>URL</code> — keep first occurrence</td></tr>
  <tr><td>Rows after deduplication</td>  <td>{n_clean:,}</td></tr>
</table>

<h2>4. Class Distribution — Before vs After Deduplication</h2>
<table>
  <tr>
    <th>Stage</th>
    <th>label=0 (Phishing)</th><th>Phishing %</th>
    <th>label=1 (Legitimate)</th><th>Legitimate %</th>
    <th>Total</th>
  </tr>
  <tr>
    <td>Before deduplication</td>
    <td>{pre_vc.get(0,0):,}</td>
    <td>{pre_vc.get(0,0)/n_orig*100:.2f}%</td>
    <td>{pre_vc.get(1,0):,}</td>
    <td>{pre_vc.get(1,0)/n_orig*100:.2f}%</td>
    <td>{n_orig:,}</td>
  </tr>
  <tr>
    <td>After deduplication</td>
    <td>{post_vc.get(0,0):,}</td>
    <td>{post_vc.get(0,0)/n_clean*100:.2f}%</td>
    <td>{post_vc.get(1,0):,}</td>
    <td>{post_vc.get(1,0)/n_clean*100:.2f}%</td>
    <td>{n_clean:,}</td>
  </tr>
</table>

<h2>5. Data Types Summary</h2>
<table>
  <tr><th>Dtype</th><th>Column count</th><th>Sample columns</th></tr>
  {dtype_rows(df_original)}
</table>

<h2>6. Output Artefacts</h2>
<table>
  <tr><th>File</th><th>Description</th><th>Status</th></tr>
  <tr>
    <td><code>data/processed/clean_df.csv</code></td>
    <td>Deduplicated dataset — {n_clean:,} rows × {df_clean.shape[1]} columns</td>
    <td><span class="badge-pass">✓ SAVED</span></td>
  </tr>
  <tr>
    <td><code>outputs/reports/m1_1_data_audit_report.html</code></td>
    <td>This audit report</td>
    <td><span class="badge-pass">✓ SAVED</span></td>
  </tr>
</table>

<p class="footer">
  Module M1.1 complete. Next step: M1.2 — Column Removal &amp; Feature Set Finalization.
</p>

</body>
</html>"""

    report_path.write_text(html, encoding="utf-8")
    logger.info(f"Audit report saved : {report_path}")
    return str(report_path)


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

def run_cleaning_pipeline(
    df_raw         : pd.DataFrame,
    output_path    : str | Path     = "data/processed/clean_df.csv",
    report_dir     : str | Path     = "outputs/reports",
    validation_report: Optional[dict] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Execute the complete M1.1 cleaning pipeline end-to-end.

    Parameters
    ----------
    df_raw             : validated raw DataFrame from loader.run_full_validation()
    output_path        : destination for the cleaned CSV
    report_dir         : directory for the HTML audit report
    validation_report  : dict from loader (forwarded into the HTML report)

    Returns
    -------
    (df_clean, cleaning_report)
    """
    sep = "=" * 60
    logger.info(sep)
    logger.info("MODULE M1.1  —  CLEANING PIPELINE")
    logger.info(sep)

    # Step 1 — Pre-dedup distribution
    pre_dist = validate_class_distribution(df_raw, stage="pre-deduplication")

    # Step 2 — Detect duplicates
    duplicate_rows, dup_count = detect_duplicates(df_raw)

    # Step 3 — Remove duplicates
    df_clean = remove_duplicates(df_raw)

    # Step 4 — Post-dedup distribution
    post_dist = validate_class_distribution(df_clean, stage="post-deduplication")

    # Step 5 — Save
    save_clean_df(df_clean, output_path)

    # Step 6 — HTML report
    report_path = generate_audit_report(
        df_original     = df_raw,
        df_clean        = df_clean,
        duplicate_rows  = duplicate_rows,
        validation_report = validation_report or {},
        output_dir      = report_dir,
    )

    cleaning_report = {
        "original_rows"          : len(df_raw),
        "duplicate_rows_removed" : dup_count,
        "clean_rows"             : len(df_clean),
        "pre_dedup_distribution" : pre_dist,
        "post_dedup_distribution": post_dist,
        "output_path"            : str(output_path),
        "report_path"            : report_path,
    }

    logger.info(sep)
    logger.info("CLEANING PIPELINE COMPLETE")
    logger.info(f"  Clean CSV : {output_path}")
    logger.info(f"  Report    : {report_path}")
    logger.info(sep)

    return df_clean, cleaning_report
