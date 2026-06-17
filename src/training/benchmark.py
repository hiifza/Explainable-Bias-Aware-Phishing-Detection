"""
src/training/benchmark.py
--------------------------
Assembles the unified benchmark table, selects best models, generates
visualisations, and writes the training_summary.html report.

Public API
----------
    create_benchmark_table(results_A, results_B)  -> pd.DataFrame
    identify_best_model(df, track)                -> (model_id, row)
    save_benchmark(df, reports_dir)
    generate_visualizations(df, plots_dir)
    generate_training_report(df, best_A, best_B, output_path) -> str
"""

import sys
from datetime import datetime
from pathlib  import Path
from typing   import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy  as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Metrics shown in the benchmark table (ordered)
BENCHMARK_COLS = [
    "model", "track",
    "accuracy", "precision", "recall", "f1", "roc_auc",
    "cv_mean_roc_auc", "cv_std_roc_auc",
    "training_time_s",
]

PRIMARY_METRIC    = "roc_auc"
TIEBREAKER_METRIC = "f1"

TRACK_COLORS = {"A": "#E24B4A", "B": "#1D9E75"}
MODEL_COLORS = {
    "Logistic Regression": "#378ADD",
    "Random Forest"      : "#854F0B",
    "XGBoost"            : "#E24B4A",
    "LightGBM"           : "#0F6E56",
}


# ── Benchmark table ───────────────────────────────────────────────────────────

def create_benchmark_table(
    results_A: list[dict],
    results_B: list[dict],
) -> pd.DataFrame:
    """
    Combine Track A and Track B results into a single benchmark DataFrame.

    Parameters
    ----------
    results_A : list of result dicts from trainer.run_track_training('A')
    results_B : list of result dicts from trainer.run_track_training('B')

    Returns
    -------
    pd.DataFrame  — 8 rows × benchmark columns
    """
    rows = []
    for result in results_A + results_B:
        row = {col: result.get(col) for col in BENCHMARK_COLS
               if col in result}
        rows.append(row)

    df = pd.DataFrame(rows)

    # Round display metrics
    for col in ["accuracy","precision","recall","f1","roc_auc",
                "cv_mean_roc_auc","cv_std_roc_auc"]:
        if col in df.columns:
            df[col] = df[col].round(6)

    logger.info(f"Benchmark table created: {df.shape}")
    return df.reset_index(drop=True)


# ── Model selection ───────────────────────────────────────────────────────────

def identify_best_model(
    df   : pd.DataFrame,
    track: str,
) -> tuple[str, pd.Series]:
    """
    Select the best model for a given track.

    Selection rule:
      1. Highest ROC AUC on test set.
      2. Tie-break: highest F1.

    Parameters
    ----------
    df    : benchmark DataFrame
    track : "A" or "B"

    Returns
    -------
    (model_id, best_row_Series)
    """
    sub = df[df["track"] == track.upper()].copy()
    if sub.empty:
        raise ValueError(f"No results for Track {track}")

    sub = sub.sort_values(
        [PRIMARY_METRIC, TIEBREAKER_METRIC],
        ascending=False,
    )
    best_row   = sub.iloc[0]
    model_name = best_row["model"]

    # Map display name → model_id
    from src.training.model_registry import MODEL_DISPLAY_NAMES
    id_map    = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}
    model_id  = id_map.get(model_name, model_name.lower().replace(" ", "_"))

    logger.info(
        f"Best model Track {track.upper()}: {model_name}  "
        f"ROC AUC={best_row[PRIMARY_METRIC]:.6f}  "
        f"F1={best_row[TIEBREAKER_METRIC]:.6f}"
    )
    return model_id, best_row


# ── CSV outputs ───────────────────────────────────────────────────────────────

def save_benchmark(
    df          : pd.DataFrame,
    reports_dir : str | Path,
) -> dict[str, Path]:
    """
    Save benchmark table and model ranking CSVs.

    Returns
    -------
    dict with 'benchmark_path' and 'ranking_path'
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    bench_path = reports_dir / "model_benchmark.csv"
    df.to_csv(bench_path, index=False)
    logger.info(f"Saved: model_benchmark.csv  ({len(df)} rows)")

    # Ranking: sort by track then roc_auc descending
    ranking = (
        df[["model","track","roc_auc","f1","accuracy","cv_mean_roc_auc"]]
        .sort_values(["track", "roc_auc"], ascending=[True, False])
        .reset_index(drop=True)
    )
    ranking["rank_within_track"] = ranking.groupby("track")["roc_auc"] \
        .rank(ascending=False, method="min").astype(int)

    rank_path = reports_dir / "model_ranking.csv"
    ranking.to_csv(rank_path, index=False)
    logger.info(f"Saved: model_ranking.csv  ({len(ranking)} rows)")

    return {"benchmark_path": bench_path, "ranking_path": rank_path}


# ── Visualisations ────────────────────────────────────────────────────────────

def _setup():
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
    plt.rcParams.update({"figure.facecolor": "white", "savefig.dpi": 150})


def _bar_compare(
    df       : pd.DataFrame,
    metric   : str,
    title    : str,
    ylabel   : str,
    filename : str,
    plots_dir: Path,
) -> None:
    """Side-by-side grouped bar chart: metric by model × track."""
    _setup()
    models = df["model"].unique()
    x      = np.arange(len(models))
    w      = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (track, color) in enumerate([("A", "#E24B4A"), ("B", "#1D9E75")]):
        sub    = df[df["track"] == track].set_index("model")
        vals   = [sub.loc[m, metric] if m in sub.index else 0 for m in models]
        offset = (i - 0.5) * w
        bars   = ax.bar(x + offset, vals, width=w, label=f"Track {track}",
                        color=color, edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8, fontweight="500")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="700")
    ax.set_ylim(max(0, df[metric].min() - 0.05), min(1.05, df[metric].max() + 0.06))
    ax.legend()
    sns.despine(ax=ax)
    plt.tight_layout()
    out = plots_dir / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out.name}")


def generate_visualizations(
    df       : pd.DataFrame,
    plots_dir: str | Path,
) -> list[Path]:
    """
    Generate all 4 benchmark visualisation plots.

    1. Model comparison bar chart (ROC AUC)
    2. F1 Score comparison
    3. CV ROC AUC with error bars
    4. Training time comparison

    Returns list of saved paths.
    """
    _setup()
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # 1. ROC AUC comparison
    _bar_compare(df, "roc_auc", "ROC AUC — Track A vs Track B",
                 "ROC AUC", "benchmark_roc_auc.png", plots_dir)
    saved.append(plots_dir / "benchmark_roc_auc.png")

    # 2. F1 Score comparison
    _bar_compare(df, "f1", "F1 Score — Track A vs Track B",
                 "F1 Score (weighted)", "benchmark_f1.png", plots_dir)
    saved.append(plots_dir / "benchmark_f1.png")

    # 3. CV ROC AUC with error bars
    if "cv_mean_roc_auc" in df.columns:
        _setup()
        models = df["model"].unique()
        x      = np.arange(len(models))
        w      = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (track, color) in enumerate([("A", "#E24B4A"), ("B", "#1D9E75")]):
            sub    = df[df["track"] == track].set_index("model")
            means  = [sub.loc[m, "cv_mean_roc_auc"] if m in sub.index else 0
                      for m in models]
            stds   = [sub.loc[m, "cv_std_roc_auc"]  if m in sub.index else 0
                      for m in models]
            offset = (i - 0.5) * w
            ax.bar(x + offset, means, width=w, label=f"Track {track}",
                   color=color, alpha=0.85, edgecolor="white")
            ax.errorbar(x + offset, means, yerr=stds,
                        fmt="none", color="#333", capsize=4, linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel("CV Mean ROC AUC ± Std")
        ax.set_title("5-Fold CV ROC AUC — Track A vs Track B",
                     fontsize=13, fontweight="700")
        ax.legend()
        sns.despine(ax=ax)
        plt.tight_layout()
        out = plots_dir / "benchmark_cv_roc_auc.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(out)
        logger.info("Saved: benchmark_cv_roc_auc.png")

    # 4. Training time comparison
    if "training_time_s" in df.columns:
        _setup()
        fig, ax = plt.subplots(figsize=(10, 4))
        for track, color, offset in [("A", "#E24B4A", -0.2), ("B","#1D9E75", 0.2)]:
            sub = df[df["track"] == track]
            if sub.empty:
                continue
            ax.bar(np.arange(len(sub)) + offset,
                   sub["training_time_s"].values,
                   width=0.35, label=f"Track {track}",
                   color=color, edgecolor="white")
        ax.set_xticks(np.arange(len(df["model"].unique())))
        ax.set_xticklabels(df["model"].unique(), rotation=15, ha="right")
        ax.set_ylabel("Training time (seconds)")
        ax.set_title("Training Time Comparison", fontsize=13, fontweight="700")
        ax.legend()
        sns.despine(ax=ax)
        plt.tight_layout()
        out = plots_dir / "benchmark_training_time.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(out)
        logger.info("Saved: benchmark_training_time.png")

    logger.info(f"Visualisations generated: {len(saved)} plots")
    return saved


# ── HTML report ───────────────────────────────────────────────────────────────

_CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,sans-serif;font-size:14px;color:#1a1a1a;
     max-width:1100px;margin:40px auto;padding:0 24px 80px}
h1{font-size:22px;font-weight:700;border-bottom:3px solid #185FA5;
   padding-bottom:12px;margin-bottom:6px}
h2{font-size:16px;font-weight:600;color:#185FA5;margin:32px 0 10px;
   border-left:4px solid #185FA5;padding-left:10px}
h3{font-size:14px;font-weight:600;color:#333;margin:16px 0 8px}
p.meta{font-size:12px;color:#888;margin-bottom:22px}
table{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0 20px}
th{background:#E6F1FB;padding:8px 12px;text-align:left;
   border:1px solid #B5D4F4;font-weight:600;white-space:nowrap}
td{padding:7px 12px;border:1px solid #ddd}
tr:nth-child(even) td{background:#F8FBFF}
.best{background:#E1F5EE!important;font-weight:600}
code{font-family:monospace;font-size:12px;background:#F1F3F5;
     padding:2px 5px;border-radius:3px}
.card{border:1px solid #B5D4F4;border-radius:8px;padding:14px 18px;
      background:#F0F7FF;margin:8px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:12px 0}
.pass{background:#E1F5EE;color:#0F6E56;padding:2px 8px;border-radius:4px;
      font-size:12px;font-weight:600}
img{max-width:100%;border:1px solid #ddd;border-radius:6px;margin:8px 0}
.footer{font-size:12px;color:#aaa;margin-top:56px;
        border-top:1px solid #eee;padding-top:16px}
"""


def generate_training_report(
    df          : pd.DataFrame,
    best_A_row  : pd.Series,
    best_B_row  : pd.Series,
    output_path : str | Path,
    plots_dir   : str | Path = "outputs/plots/training",
) -> str:
    """
    Write the complete training_summary.html report.

    Returns
    -------
    str — absolute path to the written HTML
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir   = Path(plots_dir)
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _rel(p: Path) -> str:
        try:    return "../../" + str(p.relative_to(ROOT))
        except: return str(p)

    def _img(name: str, caption: str) -> str:
        p = plots_dir / name
        r = _rel(p)
        return (f"<figure style='margin:8px 0'>"
                f"<img src='{r}' alt='{caption}'>"
                f"<figcaption style='font-size:11px;color:#888;"
                f"margin-top:4px'>{caption}</figcaption></figure>")

    # ── Benchmark table HTML ───────────────────────────────────────────────────
    cols    = [c for c in BENCHMARK_COLS if c in df.columns]
    thead   = "".join(f"<th>{c}</th>" for c in cols)
    best_ids = {
        f"{best_A_row['track']}_{best_A_row['model']}",
        f"{best_B_row['track']}_{best_B_row['model']}",
    }
    tbody = ""
    for _, row in df.iterrows():
        rid  = f"{row['track']}_{row['model']}"
        css  = " class='best'" if rid in best_ids else ""
        cells = ""
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells += f"<td>{v:.6f}</td>"
            else:
                cells += f"<td>{v}</td>"
        tbody += f"<tr{css}>{cells}</tr>\n"

    bench_html = f"""
<table>
  <thead><tr>{thead}</tr></thead>
  <tbody>{tbody}</tbody>
</table>"""

    # ── Leakage impact table ───────────────────────────────────────────────────
    leak_rows = ""
    for _, (model_a, model_b) in enumerate(
        zip(df[df["track"] == "A"].itertuples(),
            df[df["track"] == "B"].itertuples())
    ):
        delta = model_a.roc_auc - model_b.roc_auc
        leak_rows += (
            f"<tr><td>{model_a.model}</td>"
            f"<td>{model_a.roc_auc:.4f}</td>"
            f"<td>{model_b.roc_auc:.4f}</td>"
            f"<td style='color:{'#E24B4A' if delta>0.005 else '#0F6E56'}'>"
            f"+{delta:.4f}</td></tr>\n"
        )

    from src.training.model_registry import get_library_status
    lib_status = get_library_status()
    lib_rows = "".join(
        f"<tr><td>{k}</td><td><code>{v}</code></td></tr>"
        for k, v in lib_status.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>M5 Model Training Report — Phishing Detection</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Module M5.1 / M5.2 — Model Training &amp; Benchmarking</h1>
<p class="meta">
  Project: Explainable and Bias-Aware ML for Phishing Website Detection
  &nbsp;|&nbsp; Generated: {timestamp}
  &nbsp;|&nbsp; 4 models × 2 tracks = 8 experiments
</p>

<h2>1. Best Model Selection</h2>
<div class="grid2">
  <div class="card">
    <strong>Track A Winner</strong> (with URLSimilarityIndex)<br>
    <span style="font-size:18px;font-weight:700;color:#E24B4A">
      {best_A_row['model']}</span><br>
    ROC AUC: <strong>{best_A_row['roc_auc']:.6f}</strong> &nbsp;
    F1: <strong>{best_A_row['f1']:.6f}</strong>
  </div>
  <div class="card" style="background:#E1F5EE;border-color:#1D9E75">
    <strong>Track B Winner</strong> (leakage-aware primary model)<br>
    <span style="font-size:18px;font-weight:700;color:#0F6E56">
      {best_B_row['model']}</span><br>
    ROC AUC: <strong>{best_B_row['roc_auc']:.6f}</strong> &nbsp;
    F1: <strong>{best_B_row['f1']:.6f}</strong>
  </div>
</div>
<p>
  Selection rule: primary = ROC AUC, tie-breaker = F1.
  Track B winner is the <strong>primary model</strong> used for all
  SHAP, LIME, bias, and blind-spot analysis in downstream modules.
</p>

<h2>2. Full Benchmark Table</h2>
<p>Highlighted rows = winners. Track A uses 57 features (includes URLSimilarityIndex).
   Track B uses 56 features (deployment-realistic).</p>
{bench_html}

<h2>3. Visualisations</h2>
<div class="grid2">
  {_img('benchmark_roc_auc.png',     'ROC AUC comparison')}
  {_img('benchmark_f1.png',          'F1 Score comparison')}
  {_img('benchmark_cv_roc_auc.png',  '5-fold CV ROC AUC')}
  {_img('benchmark_training_time.png','Training time (seconds)')}
</div>

<h2>4. Leakage Impact (Track A − Track B ROC AUC)</h2>
<p>
  Positive delta = URLSimilarityIndex inflated Track A performance.
  Large delta (&gt;0.01) indicates heavy reliance on the leaky feature.
</p>
<table>
  <tr><th>Model</th><th>Track A AUC</th><th>Track B AUC</th><th>Delta (leakage)</th></tr>
  {leak_rows}
</table>

<h2>5. Library Status</h2>
<table>
  <tr><th>Model</th><th>Backend library</th></tr>
  {lib_rows}
</table>

<h2>6. Downstream Interface</h2>
<table>
  <tr><th>Module</th><th>Object passed in</th><th>Description</th></tr>
  <tr><td>M6.1 Evaluation</td><td><code>all_models_A, all_models_B, benchmark_df</code></td><td>All fitted models + benchmark for comparison</td></tr>
  <tr><td>M7.1 SHAP</td><td><code>best_model_B, X_train_B, X_test_B, feature_names_B</code></td><td>Track B winner + data for TreeExplainer</td></tr>
  <tr><td>M8.1 LIME</td><td><code>best_model_B, X_train_B.values, X_test_B.values, feature_names_B</code></td><td>Track B winner + numpy arrays for LIME</td></tr>
  <tr><td>M9 Bias</td><td><code>best_model_B, X_test_B, y_test, df_test_meta</code></td><td>Track B winner + test set + metadata (TLD, IsHTTPS, URLLength)</td></tr>
  <tr><td>M10 Blindspot</td><td><code>best_model_B, X_test_B, y_test, y_pred_B</code></td><td>Track B winner + predictions for FP/FN analysis</td></tr>
</table>

<p class="footer">M5 complete. Next: M6.1 — Model Evaluation.</p>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Training report saved: {output_path}")
    return str(output_path)
