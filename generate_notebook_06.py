"""
generate_notebook_06.py
-----------------------
Run once from the project root to emit:
    notebooks/06_model_evaluation.ipynb

Usage
-----
    python generate_notebook_06.py
"""

import json
from pathlib import Path


def md(source: list[str], cid: str = "") -> dict:
    return {"cell_type": "markdown",
            "id": cid or source[0][:16].strip().replace(" ", "_").lower(),
            "metadata": {}, "source": source}


def code(source: list[str], cid: str = "") -> dict:
    return {"cell_type": "code", "execution_count": None,
            "id": cid or source[0][:16].strip().replace(" ", "_").lower(),
            "metadata": {}, "outputs": [], "source": source}


cells = [

    # ── Title ──────────────────────────────────────────────────────────────────
    md([
        "# Module M6.1 — Comprehensive Model Evaluation & Final Model Selection\n",
        "\n",
        "**Project:** Explainable and Bias-Aware ML for Phishing Website Detection  \n",
        "**Roadmap ref:** Phase 6 → Module M6.1  \n",
        "\n",
        "### Evaluation pipeline\n",
        "1. Load all 8 trained models from `outputs/models/`  \n",
        "2. Compute 9 metrics per model (Accuracy, Precision, Recall, F1, ROC AUC,  \n",
        "   PR AUC, MCC, Balanced Accuracy, Brier Score)  \n",
        "3. Generate confusion matrices, ROC curves, PR curves, calibration diagrams  \n",
        "4. Select FINAL_DEPLOYMENT_MODEL (Track B winner by ROC AUC → F1 → Calibration)  \n",
        "5. Export all downstream interface objects  \n",
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
        "from src.utils.logger                    import get_logger\n",
        "from src.evaluation.evaluator            import (\n",
        "    load_all_models_and_data, run_full_evaluation\n",
        ")\n",
        "from src.evaluation.metrics              import (\n",
        "    compute_full_metrics, build_metrics_dataframe\n",
        ")\n",
        "from src.evaluation.confusion_analysis   import (\n",
        "    plot_all_confusion_matrices, plot_combined_confusion_grid\n",
        ")\n",
        "from src.evaluation.roc_analysis         import (\n",
        "    plot_combined_roc, plot_roc_all_tracks,\n",
        "    plot_combined_pr, plot_auc_ranking\n",
        ")\n",
        "from src.evaluation.calibration_analysis import (\n",
        "    plot_calibration_curves, plot_probability_distribution,\n",
        "    plot_calibration_comparison, calibration_quality_score\n",
        ")\n",
        "from src.evaluation.model_selector       import (\n",
        "    select_best_model, select_final_deployment_model,\n",
        "    build_deployment_interface, print_selection_report\n",
        ")\n",
        "from src.evaluation.evaluation_report    import generate_html_report\n",
        "\n",
        "logger = get_logger('notebook.06_model_evaluation')\n",
        "sns.set_theme(style='whitegrid', font_scale=1.05)\n",
        "plt.rcParams['figure.dpi'] = 120\n",
        "print('Imports OK ✓')\n",
    ], "cell_imports"),

    # ── 1. Paths ───────────────────────────────────────────────────────────────
    md(["## 1. Path Configuration"], "md_paths"),

    code([
        "MODELS_DIR    = PROJECT_ROOT / 'outputs' / 'models'\n",
        "PROCESSED_DIR = PROJECT_ROOT / 'data'    / 'processed'\n",
        "REPORTS_DIR   = PROJECT_ROOT / 'outputs' / 'reports'\n",
        "PLOTS_BASE    = PROJECT_ROOT / 'outputs' / 'plots' / 'evaluation'\n",
        "\n",
        "for p in [MODELS_DIR, REPORTS_DIR, PLOTS_BASE]:\n",
        "    p.mkdir(parents=True, exist_ok=True)\n",
        "for sub in ['confusion_matrices','roc','pr_curves','calibration']:\n",
        "    (PLOTS_BASE / sub).mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "# Verify all 8 trained models exist\n",
        "print('Checking trained models ...')\n",
        "missing = []\n",
        "for track in ['A','B']:\n",
        "    for m in ['logistic_regression','random_forest','xgboost','lightgbm']:\n",
        "        p = MODELS_DIR / f'track_{track}' / f'{m}.pkl'\n",
        "        sym = '✓' if p.exists() else '✗ MISSING'\n",
        "        if not p.exists(): missing.append(str(p))\n",
        "        print(f'  {sym}  {p.relative_to(PROJECT_ROOT)}')\n",
        "if missing:\n",
        "    raise RuntimeError(f'Missing models: {missing} — run M5 notebook first')\n",
        "print('All 8 models found ✓')\n",
    ], "cell_paths"),

    # ── 2. Load Models & Data ─────────────────────────────────────────────────
    md(["## 2. Load Trained Models and Test Data"], "md_load"),

    code([
        "loaded = load_all_models_and_data(\n",
        "    models_dir    = MODELS_DIR,\n",
        "    processed_dir = PROCESSED_DIR,\n",
        ")\n",
        "\n",
        "models_A = loaded['models_A']\n",
        "models_B = loaded['models_B']\n",
        "X_test_A = loaded['X_test_A']\n",
        "X_test_B = loaded['X_test_B']\n",
        "y_test   = loaded['y_test']\n",
        "\n",
        "print(f'Track A models : {list(models_A.keys())}')\n",
        "print(f'Track B models : {list(models_B.keys())}')\n",
        "print(f'X_test_A       : {X_test_A.shape}')\n",
        "print(f'X_test_B       : {X_test_B.shape}')\n",
        "print(f'y_test         : {len(y_test):,} labels')\n",
        "print(f'Class balance  : phishing={int((y_test==0).sum()):,}  '\n",
        "      f'legitimate={int((y_test==1).sum()):,}')\n",
    ], "cell_load"),

    # ── 3. Full Evaluation ────────────────────────────────────────────────────
    md(["## 3. Full Evaluation (all 8 models × 9 metrics)"], "md_eval"),

    code([
        "# Run the complete evaluation pipeline\n",
        "# This generates all metrics, plots, and CSV reports automatically.\n",
        "print('Running full evaluation ...')\n",
        "\n",
        "eval_output = run_full_evaluation(\n",
        "    models_dir    = MODELS_DIR,\n",
        "    processed_dir = PROCESSED_DIR,\n",
        "    reports_dir   = REPORTS_DIR,\n",
        "    plots_dir     = PLOTS_BASE,\n",
        ")\n",
        "\n",
        "eval_results = eval_output['eval_results']\n",
        "metrics_df   = eval_output['metrics_df']\n",
        "error_df     = eval_output['error_df']\n",
        "\n",
        "print(f'Evaluated {len(eval_results)} models')\n",
        "print(f'Metrics shape: {metrics_df.shape}')\n",
    ], "cell_eval"),

    # ── 4. Metrics Table ──────────────────────────────────────────────────────
    md(["## 4. Metrics Comparison Table"], "md_metrics"),

    code([
        "display_cols = ['model','track','accuracy','precision','recall','f1',\n",
        "                'roc_auc','pr_auc','mcc','balanced_accuracy','brier_score']\n",
        "display(metrics_df[[c for c in display_cols if c in metrics_df.columns]])\n",
        "print(f'\\nSaved → outputs/reports/evaluation_metrics.csv')\n",
    ], "cell_metrics_table"),

    # ── 5. Confusion Matrix Analysis ──────────────────────────────────────────
    md(["## 5. Confusion Matrix Analysis"], "md_cm"),

    code([
        "from src.evaluation.metrics import compute_error_metrics\n",
        "\n",
        "print('TP / FP / FN / TN breakdown per model (Track B):')\n",
        "b_results = [r for r in eval_results if r.get('track','')=='B']\n",
        "for r in b_results:\n",
        "    em = compute_error_metrics(r['y_true'], r['y_pred'])\n",
        "    print(f\"  {r['model']:<26}  \"\n",
        "          f\"TP={em['tp']:>5,}  TN={em['tn']:>5,}  \"\n",
        "          f\"FP={em['fp']:>5,}  FN={em['fn']:>5,}  \"\n",
        "          f\"FPR={em['fpr']:.4f}  FNR={em['fnr']:.4f}\")\n",
    ], "cell_cm_table"),

    code([
        "# Display combined CM grid for Track B\n",
        "from IPython.display import Image\n",
        "cm_grid_path = PLOTS_BASE / 'confusion_matrices' / 'cm_grid_trackB.png'\n",
        "if cm_grid_path.exists():\n",
        "    display(Image(str(cm_grid_path)))\n",
        "else:\n",
        "    print('CM grid not found — run evaluation first')\n",
    ], "cell_cm_plot"),

    # ── 6. ROC Analysis ───────────────────────────────────────────────────────
    md(["## 6. ROC Analysis"], "md_roc"),

    code([
        "from IPython.display import Image\n",
        "roc_path = PLOTS_BASE / 'roc' / 'roc_all_tracks.png'\n",
        "if roc_path.exists():\n",
        "    display(Image(str(roc_path)))\n",
        "else:\n",
        "    print('ROC plot not found')\n",
        "\n",
        "# Show ROC AUC ranking\n",
        "print('\\nROC AUC ranking (Track B):')\n",
        "b_auc = metrics_df[metrics_df['track']=='B'][['model','roc_auc','pr_auc']]\n",
        "print(b_auc.sort_values('roc_auc', ascending=False).to_string(index=False))\n",
    ], "cell_roc"),

    # ── 7. Precision-Recall Analysis ──────────────────────────────────────────
    md(["## 7. Precision-Recall Analysis"], "md_pr"),

    code([
        "from IPython.display import Image\n",
        "pr_path = PLOTS_BASE / 'pr_curves' / 'pr_trackB.png'\n",
        "if pr_path.exists():\n",
        "    display(Image(str(pr_path)))\n",
        "\n",
        "print('PR AUC ranking (Track B):')\n",
        "print(b_auc.sort_values('pr_auc', ascending=False).to_string(index=False))\n",
    ], "cell_pr"),

    # ── 8. Calibration Analysis ───────────────────────────────────────────────
    md(["## 8. Calibration Analysis"], "md_cal"),

    code([
        "from IPython.display import Image\n",
        "cal_path = PLOTS_BASE / 'calibration' / 'calibration_comparison.png'\n",
        "if cal_path.exists():\n",
        "    display(Image(str(cal_path)))\n",
        "\n",
        "print('Calibration quality scores (Track B):')\n",
        "if 'calibration_quality' in metrics_df.columns:\n",
        "    cal_df = metrics_df[metrics_df['track']=='B'][['model','calibration_quality']]\n",
        "    print(cal_df.sort_values('calibration_quality', ascending=False).to_string(index=False))\n",
    ], "cell_cal"),

    # ── 9. Error Analysis ─────────────────────────────────────────────────────
    md(["## 9. Error Analysis (FPR, FNR, Specificity, Sensitivity)"], "md_err"),

    code([
        "err_cols = ['model','track','fpr','fnr','specificity','sensitivity']\n",
        "err_cols_present = [c for c in err_cols if c in error_df.columns]\n",
        "display(error_df[err_cols_present])\n",
        "print('\\nSaved → outputs/reports/error_analysis.csv')\n",
    ], "cell_err"),

    code([
        "# FNR insight: phishing sites missed (highest risk for security)\n",
        "print('FNR comparison (lower = fewer missed phishing sites):')\n",
        "if 'fnr' in metrics_df.columns:\n",
        "    fnr_df = metrics_df[metrics_df['track']=='B'][['model','fnr','fpr']]\n",
        "    fnr_df = fnr_df.sort_values('fnr')\n",
        "    for _, row in fnr_df.iterrows():\n",
        "        print(f\"  {row['model']:<26}  FNR={row['fnr']:.4f}  FPR={row['fpr']:.4f}\")\n",
    ], "cell_fnr"),

    # ── 10. Model Ranking ────────────────────────────────────────────────────
    md(["## 10. Model Ranking"], "md_rank"),

    code([
        "print('=== COMPLETE RANKING TABLE (by ROC AUC) ===')\n",
        "rank_df = metrics_df.sort_values(['track','roc_auc'], ascending=[True,False])\n",
        "rank_cols = ['model','track','roc_auc','f1','mcc','brier_score']\n",
        "rank_cols_p = [c for c in rank_cols if c in rank_df.columns]\n",
        "print(rank_df[rank_cols_p].to_string(index=False))\n",
    ], "cell_rank"),

    # ── 11. Final Model Selection ─────────────────────────────────────────────
    md(["## 11. Final Model Selection"], "md_select"),

    code([
        "selection = select_final_deployment_model(eval_results)\n",
        "print_selection_report(selection)\n",
        "\n",
        "FINAL_DEPLOYMENT_MODEL = selection['FINAL_DEPLOYMENT_MODEL']\n",
        "print(f'\\nFINAL_DEPLOYMENT_MODEL type : {type(FINAL_DEPLOYMENT_MODEL).__name__}')\n",
        "print(f'deployment_model_name       : {selection[\"deployment_model_name\"]}')\n",
        "print(f'deployment_model_id         : {selection[\"deployment_model_id\"]}')\n",
    ], "cell_select"),

    # ── 12. Deployment Interface ──────────────────────────────────────────────
    md(["## 12. Build Downstream Deployment Interface"], "md_interface"),

    code([
        "interface = build_deployment_interface(selection, loaded)\n",
        "\n",
        "print('=== DOWNSTREAM MODULE INTERFACE ===')\n",
        "print()\n",
        "print('M7.1 SHAP:')\n",
        "print(f'  SHAP_model         : {type(interface[\"SHAP_model\"]).__name__}')\n",
        "print(f'  SHAP_X_train       : {interface[\"SHAP_X_train\"].shape}')\n",
        "print(f'  SHAP_X_test        : {interface[\"SHAP_X_test\"].shape}')\n",
        "print(f'  SHAP_feature_names : {len(interface[\"SHAP_feature_names\"])} features')\n",
        "print()\n",
        "print('M8.1 LIME:')\n",
        "print(f'  LIME_predict_fn    : {interface[\"LIME_predict_fn\"]}')\n",
        "print(f'  LIME_X_train_np    : {interface[\"LIME_X_train_np\"].shape}')\n",
        "print(f'  LIME_X_test_np     : {interface[\"LIME_X_test_np\"].shape}')\n",
        "print()\n",
        "print('M9 Bias / M10 Blindspot:')\n",
        "print(f'  BIAS_X_test        : {interface[\"BIAS_X_test\"].shape}')\n",
        "print(f'  BIAS_y_test        : {len(interface[\"BIAS_y_test\"]):,} labels')\n",
        "print(f'  BIAS_y_pred shape  : {interface[\"BIAS_y_pred\"].shape}')\n",
        "print(f'  BIAS_y_proba shape : {interface[\"BIAS_y_proba\"].shape}')\n",
        "print()\n",
        "print(f'feature_names_B    : first 5 = {interface[\"feature_names_B\"][:5]}')\n",
    ], "cell_interface"),

    # ── 13. Export HTML Report ────────────────────────────────────────────────
    md(["## 13. Generate Evaluation Report"], "md_report"),

    code([
        "report_path = generate_html_report(\n",
        "    eval_results = eval_results,\n",
        "    metrics_df   = metrics_df,\n",
        "    selection    = selection,\n",
        "    output_path  = REPORTS_DIR / 'model_evaluation_report.html',\n",
        "    plots_dir    = PLOTS_BASE,\n",
        ")\n",
        "print(f'Report saved: {report_path}')\n",
    ], "cell_report"),

    # ── 14. Artifact Verification ─────────────────────────────────────────────
    md(["## 14. Artifact Verification"], "md_verify"),

    code([
        "print('=== M6.1 OUTPUT ARTIFACTS ===')\n",
        "artifacts = [\n",
        "    'outputs/reports/evaluation_metrics.csv',\n",
        "    'outputs/reports/error_analysis.csv',\n",
        "    'outputs/reports/model_evaluation_report.html',\n",
        "    'outputs/plots/evaluation/confusion_matrices/cm_grid_trackB.png',\n",
        "    'outputs/plots/evaluation/roc/roc_all_tracks.png',\n",
        "    'outputs/plots/evaluation/roc/auc_ranking.png',\n",
        "    'outputs/plots/evaluation/pr_curves/pr_trackB.png',\n",
        "    'outputs/plots/evaluation/calibration/calibration_comparison.png',\n",
        "]\n",
        "import pathlib\n",
        "for rel in artifacts:\n",
        "    p = PROJECT_ROOT / rel\n",
        "    print(f\"  {'✓' if p.exists() else '✗'}  {rel}\")\n",
    ], "cell_verify"),

    # ── 15. Conclusions ───────────────────────────────────────────────────────
    md(["## 15. Conclusions & Downstream Handoff"], "md_conclusions"),

    code([
        "print('=' * 65)\n",
        "print('MODULE M6.1 COMPLETE')\n",
        "print('=' * 65)\n",
        "print()\n",
        "print(f\"FINAL_DEPLOYMENT_MODEL    : {selection['deployment_model_name']}\")\n",
        "print(f\"  Track B ROC AUC         : {selection['best_B']['primary_score']:.6f}\")\n",
        "print(f\"  Track B F1              : {selection['best_B']['secondary_score']:.6f}\")\n",
        "print(f\"  Calibration quality     : {selection['best_B']['tertiary_score']:.4f}\")\n",
        "print(f\"  Leakage impact (A-B)    : {selection.get('leakage_impact_auc',0):+.6f}\")\n",
        "print()\n",
        "print('Ready for downstream modules:')\n",
        "print('  M7.1 SHAP    — SHAP_model, SHAP_X_train, SHAP_X_test, SHAP_feature_names')\n",
        "print('  M8.1 LIME    — LIME_predict_fn, LIME_X_train_np, LIME_X_test_np, LIME_feature_names')\n",
        "print('  M9  Bias     — BIAS_model, BIAS_X_test, BIAS_y_test, BIAS_y_pred, BIAS_y_proba')\n",
        "print('  M10 Blindspot — same as Bias + BLINDSPOT_y_proba')\n",
        "print()\n",
        "print('Next: M7.1 — SHAP Explainability Analysis')\n",
    ], "cell_conclusions"),
]


notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out = Path(__file__).resolve().parent / "notebooks" / "06_model_evaluation.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Notebook written → {out}")
