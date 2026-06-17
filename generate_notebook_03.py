"""
generate_notebook_03.py
-----------------------
Run once from the project root to emit:
    notebooks/03_eda.ipynb

Usage
-----
    python generate_notebook_03.py
"""

import json
from pathlib import Path


def md(source: list[str], cid: str = "") -> dict:
    return {"cell_type": "markdown", "id": cid or source[0][:14].strip()
            .replace(" ", "_").lower(), "metadata": {}, "source": source}


def code(source: list[str], cid: str = "") -> dict:
    return {"cell_type": "code", "execution_count": None,
            "id": cid or source[0][:14].strip().replace(" ", "_").lower(),
            "metadata": {}, "outputs": [], "source": source}


cells = [

    # ── Title ──────────────────────────────────────────────────────────────────
    md([
        "# Module M2.1 — Comprehensive Exploratory Data Analysis\n",
        "\n",
        "**Project:** Explainable and Bias-Aware ML for Phishing Website Detection  \n",
        "**Roadmap ref:** Phase 2 → Module M2.1  \n",
        "\n",
        "### Sections\n",
        "1. Environment Setup  \n",
        "2. Dataset Loading  \n",
        "3. Dataset Overview  \n",
        "4. Distribution Analysis  \n",
        "5. Correlation Analysis  \n",
        "6. Leakage Analysis  \n",
        "7. TLD Analysis  \n",
        "8. Feature Importance Pre-Screening  \n",
        "9. Outlier Analysis  \n",
        "10. Report Generation  \n",
        "11. Conclusions  \n",
    ], "title"),

    # ── 0. Setup ───────────────────────────────────────────────────────────────
    md(["## 0. Environment Setup"], "md_setup"),

    code([
        "import sys\n",
        "from pathlib import Path\n",
        "\n",
        "PROJECT_ROOT = Path().resolve().parent\n",
        "if str(PROJECT_ROOT) not in sys.path:\n",
        "    sys.path.insert(0, str(PROJECT_ROOT))\n",
        "\n",
        "print(f'Project root: {PROJECT_ROOT}')\n",
    ], "cell_root"),

    code([
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "import numpy  as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "from src.utils.logger              import get_logger\n",
        "from src.features.feature_catalog  import get_feature_lists\n",
        "from src.eda.visualization_manager import (\n",
        "    setup_plot_style, save_figure, generate_eda_report\n",
        ")\n",
        "from src.eda.eda_analyzer     import run_eda_overview\n",
        "from src.eda.correlation_analyzer import run_correlation_analysis\n",
        "from src.eda.distribution_analyzer import run_distribution_analysis\n",
        "from src.eda.leakage_analyzer import run_leakage_analysis\n",
        "\n",
        "logger = get_logger('notebook.03_eda')\n",
        "setup_plot_style()\n",
        "print('Imports OK ✓')\n",
    ], "cell_imports"),

    # ── 1. Paths ───────────────────────────────────────────────────────────────
    md(["## 1. Path Configuration"], "md_paths"),

    code([
        "CLEAN_CSV   = PROJECT_ROOT / 'data' / 'processed' / 'clean_df.csv'\n",
        "REPORT_DIR  = PROJECT_ROOT / 'outputs' / 'reports'\n",
        "PLOTS_EDA   = PROJECT_ROOT / 'outputs' / 'plots' / 'eda'\n",
        "\n",
        "assert CLEAN_CSV.exists(), f'Run M1.1 first — {CLEAN_CSV} not found'\n",
        "\n",
        "REPORT_DIR.mkdir(parents=True, exist_ok=True)\n",
        "PLOTS_EDA.mkdir(parents=True, exist_ok=True)\n",
        "(PLOTS_EDA / 'distributions').mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "print(f'Clean CSV  : {CLEAN_CSV}')\n",
        "print(f'Reports    : {REPORT_DIR}')\n",
        "print(f'Plots      : {PLOTS_EDA}')\n",
    ], "cell_paths"),

    # ── 2. Load Data ───────────────────────────────────────────────────────────
    md(["## 2. Dataset Loading"], "md_load"),

    code([
        "df = pd.read_csv(CLEAN_CSV, low_memory=False)\n",
        "fl = get_feature_lists()\n",
        "\n",
        "print(f'Shape         : {df.shape}')\n",
        "print(f'Track B feats : {len(fl.track_B)}')\n",
        "print(f'Track A feats : {len(fl.track_A)}')\n",
        "print(f'Target col    : {fl.target}')\n",
        "display(df.head(3))\n",
    ], "cell_load"),

    # ── 3. Dataset Overview ────────────────────────────────────────────────────
    md(["## 3. Dataset Overview"], "md_overview"),

    code([
        "from src.eda.eda_analyzer import (\n",
        "    compute_dataset_overview, save_overview_csv,\n",
        "    plot_class_distribution, plot_dtype_summary,\n",
        ")\n",
        "\n",
        "overview = compute_dataset_overview(df, fl.track_B)\n",
        "save_overview_csv(overview, REPORT_DIR / 'dataset_overview.csv')\n",
        "\n",
        "cd = overview['class_distribution']\n",
        "print(f\"Rows          : {overview['n_rows']:,}\")\n",
        "print(f\"Track B feat  : {overview['n_features']}\")\n",
        "print(f\"Missing values: {overview['total_missing']}\")\n",
        "print(f\"Memory        : {overview['memory_mb']:.1f} MB\")\n",
        "print(f\"Phishing (0)  : {cd['phishing_count']:,}  ({cd['phishing_pct']:.2f}%)\")\n",
        "print(f\"Legitimate (1): {cd['legitimate_count']:,}  ({cd['legitimate_pct']:.2f}%)\")\n",
        "print(f\"Imbalance     : {cd['imbalance_ratio']:.4f}\")\n",
        "print('Saved → outputs/reports/dataset_overview.csv')\n",
    ], "cell_overview"),

    code([
        "plot_class_distribution(df, PLOTS_EDA)\n",
        "plot_dtype_summary(df, fl.track_B, PLOTS_EDA)\n",
        "print('Plots saved ✓')\n",
    ], "cell_overview_plots"),

    code([
        "# Feature category summary\n",
        "cat_summary = []\n",
        "for cat, feats in fl.categories.items():\n",
        "    num_in_cat = sum(1 for f in feats\n",
        "                     if df[f].dtype in ['int64','float64'])\n",
        "    cat_summary.append({\n",
        "        'category': cat,\n",
        "        'n_features': len(feats),\n",
        "        'n_numeric': num_in_cat,\n",
        "        'n_binary_or_cat': len(feats) - num_in_cat,\n",
        "    })\n",
        "cat_df = pd.DataFrame(cat_summary)\n",
        "print(cat_df.to_string(index=False))\n",
    ], "cell_cat_summary"),

    # ── 4. Distribution Analysis ───────────────────────────────────────────────
    md(["## 4. Distribution Analysis"], "md_dist"),

    md([
        "### 4a. Individual Feature Distributions\n",
        "\n",
        "Each feature gets a 4-panel figure (histogram, KDE by class, boxplot, "
        "class-separated boxplot).\n",
        "Set `PLOT_ALL_DISTS = True` to generate all 49 plots (takes ~2 minutes).\n",
    ], "md_dist_note"),

    code([
        "# Set to True to generate all 49 per-feature distribution plots\n",
        "PLOT_ALL_DISTS = True\n",
        "\n",
        "dist_results = run_distribution_analysis(\n",
        "    df        = df,\n",
        "    features  = fl.track_B,\n",
        "    output_dir= REPORT_DIR,\n",
        "    plots_dir = PLOTS_EDA,\n",
        "    plot_all  = PLOT_ALL_DISTS,\n",
        ")\n",
        "print(f\"Distribution plots generated: {len(dist_results['distribution_paths'])}\")\n",
        "print(f\"Features with >5% outliers  : {dist_results['n_outlier_features']}\")\n",
    ], "cell_dist"),

    code([
        "# Preview outlier stats table\n",
        "outlier_df = dist_results['outlier_stats']\n",
        "print('Top 10 features by outlier %:')\n",
        "display(outlier_df[['feature','median','p99','max','outlier_pct']].head(10))\n",
    ], "cell_outlier_preview"),

    # ── 5. Correlation Analysis ────────────────────────────────────────────────
    md(["## 5. Correlation Analysis"], "md_corr"),

    code([
        "corr_results = run_correlation_analysis(\n",
        "    df        = df,\n",
        "    features  = fl.track_B,\n",
        "    output_dir= REPORT_DIR,\n",
        "    plots_dir = PLOTS_EDA,\n",
        ")\n",
        "print(f\"High-corr pairs (|r|≥0.75)   : {corr_results['n_high_pairs']}\")\n",
        "print(f\"Network edges (|r|≥0.30)      : {len(corr_results['network'])}\")\n",
    ], "cell_corr"),

    code([
        "print('Top 10 positive correlations (Pearson):')\n",
        "display(corr_results['top_positive'][['feat_A','feat_B','pearson_r']].head(10))\n",
        "\n",
        "print('\\nTop 10 negative correlations (Pearson):')\n",
        "display(corr_results['top_negative'][['feat_A','feat_B','pearson_r']].head(10))\n",
    ], "cell_corr_display"),

    code([
        "# Inspect correlation matrix snippet\n",
        "print('Pearson matrix shape:', corr_results['pearson_matrix'].shape)\n",
        "print('Spearman matrix shape:', corr_results['spearman_matrix'].shape)\n",
        "\n",
        "# Show correlation with target\n",
        "target_corr = df[fl.track_B + ['label']].select_dtypes(include=[np.number]).corr()['label']\n",
        "target_corr = target_corr.drop('label').abs().sort_values(ascending=False)\n",
        "print('\\nTop 10 features correlated with label (Track B):')\n",
        "print(target_corr.head(10).to_string())\n",
    ], "cell_corr_target"),

    # ── 6. Leakage Analysis ────────────────────────────────────────────────────
    md(["## 6. Leakage Analysis"], "md_leakage"),

    code([
        "leakage_results = run_leakage_analysis(\n",
        "    df        = df,\n",
        "    output_dir= REPORT_DIR,\n",
        "    plots_dir = PLOTS_EDA,\n",
        ")\n",
        "\n",
        "usi   = leakage_results['urlsimilarity']\n",
        "https = leakage_results['https']\n",
        "\n",
        "print('=== URLSimilarityIndex ===')\n",
        "print(f\"  All legitimate = 100.0 : {usi['all_legit_100']}\")\n",
        "print(f\"  Phishing mean          : {usi['phishing_mean']}\")\n",
        "print(f\"  Legitimate mean        : {usi['legit_mean']}\")\n",
        "print(f\"  AUROC (solo)           : {usi['auroc']:.6f}\")\n",
        "\n",
        "print('\\n=== IsHTTPS ===')\n",
        "print(f\"  All legitimate HTTPS=1 : {https['all_legit_https1']}\")\n",
        "print(f\"  Phishing HTTPS rate    : {https['phishing_https_pct']:.1f}%\")\n",
        "print(f\"  Legitimate HTTPS rate  : {https['legit_https_pct']:.1f}%\")\n",
        "print(f\"  AUROC (solo)           : {https['auroc']:.6f}\")\n",
    ], "cell_leakage"),

    code([
        "# Validation assertions\n",
        "assert usi['all_legit_100'], 'URLSimilarityIndex leakage not confirmed'\n",
        "assert usi['auroc'] > 0.95,  'URLSimilarityIndex AUROC unexpectedly low'\n",
        "assert https['all_legit_https1'], 'IsHTTPS not all-1 for legitimate'\n",
        "print('All leakage assertions PASSED ✓')\n",
    ], "cell_leakage_assert"),

    # ── 7. TLD Analysis ────────────────────────────────────────────────────────
    md(["## 7. TLD Analysis"], "md_tld"),

    code([
        "from src.eda.eda_analyzer import compute_tld_analysis, plot_tld_phishing_rate, plot_tld_frequency\n",
        "\n",
        "tld_results = compute_tld_analysis(df)\n",
        "\n",
        "print(f\"Unique TLDs      : {tld_results['n_unique_tlds']}\")\n",
        "print(f\"Long-tail TLDs   : {tld_results['long_tail_count']} (< 100 samples)\")\n",
        "\n",
        "top_tlds = tld_results['top_tlds']\n",
        "print('\\nTop 20 TLDs by sample count:')\n",
        "display(top_tlds[['TLD','count','phishing_count','phishing_rate']].head(20))\n",
    ], "cell_tld"),

    code([
        "# Most phishing-heavy TLDs (min 100 samples)\n",
        "high_phish = top_tlds[top_tlds['count'] >= 100].nlargest(10, 'phishing_rate')\n",
        "print('Most phishing-heavy TLDs (min 100 samples):')\n",
        "display(high_phish[['TLD','count','phishing_rate']].reset_index(drop=True))\n",
        "\n",
        "# Safest TLDs\n",
        "safest = top_tlds[top_tlds['count'] >= 100].nsmallest(10, 'phishing_rate')\n",
        "print('\\nSafest TLDs (min 100 samples):')\n",
        "display(safest[['TLD','count','phishing_rate']].reset_index(drop=True))\n",
    ], "cell_tld_analysis"),

    code([
        "plot_tld_phishing_rate(tld_results, PLOTS_EDA)\n",
        "plot_tld_frequency(tld_results, PLOTS_EDA)\n",
        "print('TLD plots saved ✓')\n",
    ], "cell_tld_plots"),

    # ── 8. Feature Pre-Screening ───────────────────────────────────────────────
    md(["## 8. Feature Importance Pre-Screening"], "md_ps"),

    code([
        "from src.eda.eda_analyzer import (\n",
        "    compute_feature_prescreening, plot_mutual_information, plot_anova_prescreening\n",
        ")\n",
        "\n",
        "ps_results = compute_feature_prescreening(df, fl.track_B, sample_n=50_000)\n",
        "\n",
        "mi_df = ps_results['mutual_information']\n",
        "print(f\"Pre-screening computed for {ps_results['n_numeric']} numeric features\")\n",
        "print('\\nTop 15 features by Mutual Information:')\n",
        "display(mi_df[['feature','mi_score','anova_f','mi_rank']].head(15))\n",
    ], "cell_ps"),

    code([
        "plot_mutual_information(ps_results, PLOTS_EDA)\n",
        "plot_anova_prescreening(ps_results, PLOTS_EDA)\n",
        "print('Pre-screening plots saved ✓')\n",
    ], "cell_ps_plots"),

    # ── 9. Outlier Analysis ────────────────────────────────────────────────────
    md(["## 9. Outlier Analysis"], "md_outlier"),

    code([
        "from src.eda.distribution_analyzer import (\n",
        "    OUTLIER_FEATURES, compute_outlier_stats, plot_outlier_boxplots, plot_skewness_summary\n",
        ")\n",
        "\n",
        "valid_outlier_feats = [f for f in OUTLIER_FEATURES if f in df.columns]\n",
        "outlier_df = compute_outlier_stats(df, valid_outlier_feats)\n",
        "\n",
        "print('Full Outlier Statistics:')\n",
        "display(outlier_df[['feature','q1','median','q3','p99','max','n_outliers','outlier_pct']])\n",
    ], "cell_outlier"),

    code([
        "plot_outlier_boxplots(df, valid_outlier_feats, PLOTS_EDA)\n",
        "plot_skewness_summary(df, fl.track_B, PLOTS_EDA)\n",
        "print('Outlier plots saved ✓')\n",
    ], "cell_outlier_plots"),

    code([
        "# Identify features requiring log1p transformation (|skew| > 5)\n",
        "num_feats = [c for c in fl.track_B if df[c].dtype in ['int64','float64']]\n",
        "skew_vals = df[num_feats].skew().abs().sort_values(ascending=False)\n",
        "log_candidates = skew_vals[skew_vals > 5]\n",
        "print(f'Features needing log1p transform (|skew|>5): {len(log_candidates)}')\n",
        "print(log_candidates.to_string())\n",
    ], "cell_logtransform"),

    # ── 10. Full Run & Report ──────────────────────────────────────────────────
    md(["## 10. Complete Pipeline Run & HTML Report"], "md_report"),

    code([
        "# Collect all results for the HTML report\n",
        "all_results = {\n",
        "    'overview'    : overview,\n",
        "    'leakage'     : leakage_results,\n",
        "    'correlation' : corr_results,\n",
        "    'tld'         : tld_results,\n",
        "    'outliers'    : {\n",
        "        'outlier_stats'       : outlier_df,\n",
        "        'n_outlier_features'  : int((outlier_df['outlier_pct'] > 5).sum()),\n",
        "    },\n",
        "    'prescreening': ps_results,\n",
        "}\n",
        "\n",
        "report_path = generate_eda_report(\n",
        "    results     = all_results,\n",
        "    output_path = REPORT_DIR / 'm2_1_eda_report.html',\n",
        "    plots_dir   = PLOTS_EDA,\n",
        ")\n",
        "print(f'EDA report saved: {report_path}')\n",
    ], "cell_report"),

    # ── 11. Conclusions ────────────────────────────────────────────────────────
    md(["## 11. Conclusions & Preprocessing Recommendations"], "md_conclusions"),

    code([
        "print('=' * 65)\n",
        "print('MODULE M2.1 — EDA CONCLUSIONS')\n",
        "print('=' * 65)\n",
        "\n",
        "usi   = leakage_results['urlsimilarity']\n",
        "https = leakage_results['https']\n",
        "n_log = len(df[num_feats].skew()[df[num_feats].skew().abs() > 5])\n",
        "\n",
        "print(f'  LEAKAGE  URLSimilarityIndex AUROC     : {usi[\"auroc\"]:.4f} → exclude from Track B')\n",
        "print(f'  ADVISORY IsHTTPS AUROC                : {https[\"auroc\"]:.4f} → retain, flag for bias')\n",
        "print(f'  CORR     High-corr pairs (|r|≥0.75)  : {corr_results[\"n_high_pairs\"]}')\n",
        "print(f'  TLD      Unique TLDs                  : {tld_results[\"n_unique_tlds\"]} → frequency encode')\n",
        "print(f'  OUTLIERS Features needing log1p        : {n_log}')\n",
        "print(f'  IMBALANCE Phishing %                   : {overview[\"class_distribution\"][\"phishing_pct\"]:.2f}% → mild, stratified CV')\n",
        "print()\n",
        "print('PREPROCESSING REQUIREMENTS FOR M3.1:')\n",
        "print('  1. TLD frequency encoding (top-50 + rare_tld)')\n",
        "print('  2. Log1p transform for all high-skew count features')\n",
        "print('  3. Outlier capping at P99.9 before scaling')\n",
        "print('  4. RobustScaler for continuous features')\n",
        "print('  5. No imputation required (zero missing values)')\n",
        "print('  6. Stratified 80/20 split (class balance preserved)')\n",
        "print()\n",
        "print('M2.1 COMPLETE. Next: M3.1 — Preprocessing Pipeline')\n",
    ], "cell_conclusions"),
]

# ── Assemble and write ─────────────────────────────────────────────────────────

notebook = {
    "nbformat"      : 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out = Path(__file__).resolve().parent / "notebooks" / "03_eda.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Notebook written → {out}")
