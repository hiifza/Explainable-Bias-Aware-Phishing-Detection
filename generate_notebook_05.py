"""
generate_notebook_05.py
-----------------------
Run once from the project root to emit:
    notebooks/05_model_training.ipynb

Usage
-----
    python generate_notebook_05.py
"""

import json
from pathlib import Path


def md(source: list[str], cid: str = "") -> dict:
    return {"cell_type": "markdown", "id": cid or source[0][:16].strip()
            .replace(" ", "_").lower(), "metadata": {}, "source": source}


def code(source: list[str], cid: str = "") -> dict:
    return {"cell_type": "code", "execution_count": None,
            "id": cid or source[0][:16].strip().replace(" ", "_").lower(),
            "metadata": {}, "outputs": [], "source": source}


cells = [

    # ── Title ──────────────────────────────────────────────────────────────────
    md([
        "# Module M5.1 / M5.2 — Model Training & Benchmarking\n",
        "\n",
        "**Project:** Explainable and Bias-Aware ML for Phishing Website Detection  \n",
        "**Roadmap ref:** Phase 5 → Modules M5.1 and M5.2  \n",
        "\n",
        "### Models trained\n",
        "- Logistic Regression  \n",
        "- Random Forest  \n",
        "- XGBoost (or HistGBM fallback if not installed)  \n",
        "- LightGBM (or HistGBM fallback if not installed)  \n",
        "\n",
        "### Tracks\n",
        "- **Track A**: 57 features (includes URLSimilarityIndex — ceiling experiment)  \n",
        "- **Track B**: 56 features (leakage-aware — primary deployment model)  \n",
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
        "import joblib\n",
        "\n",
        "from src.utils.logger             import get_logger\n",
        "from src.utils.metrics_logger     import log_benchmark_table\n",
        "from src.training.model_registry  import get_all_models, get_library_status, MODEL_DISPLAY_NAMES\n",
        "from src.training.trainer         import load_track_data, run_track_training\n",
        "from src.training.benchmark       import (\n",
        "    create_benchmark_table, identify_best_model,\n",
        "    save_benchmark, generate_visualizations, generate_training_report,\n",
        ")\n",
        "from src.training.model_saver     import save_all_models, load_all_models\n",
        "\n",
        "logger = get_logger('notebook.05_model_training')\n",
        "sns.set_theme(style='whitegrid', font_scale=1.05)\n",
        "plt.rcParams['figure.dpi'] = 120\n",
        "print('Imports OK ✓')\n",
    ], "cell_imports"),

    # ── 1. Paths ───────────────────────────────────────────────────────────────
    md(["## 1. Path Configuration"], "md_paths"),

    code([
        "PROCESSED_DIR   = PROJECT_ROOT / 'data' / 'processed'\n",
        "MODELS_DIR      = PROJECT_ROOT / 'outputs' / 'models'\n",
        "REPORTS_DIR     = PROJECT_ROOT / 'outputs' / 'reports'\n",
        "PLOTS_DIR       = PROJECT_ROOT / 'outputs' / 'plots' / 'training'\n",
        "\n",
        "for p in [MODELS_DIR, REPORTS_DIR, PLOTS_DIR]:\n",
        "    p.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "# Verify preprocessed data exists\n",
        "for track in ['A','B']:\n",
        "    for split in ['X_train','X_test']:\n",
        "        p = PROCESSED_DIR / f'track_{track}' / f'{split}.csv'\n",
        "        assert p.exists(), f'Missing: {p} — run M3.1 first'\n",
        "\n",
        "print('All preprocessed data found ✓')\n",
    ], "cell_paths"),

    # ── 2. Library Status ──────────────────────────────────────────────────────
    md(["## 2. Library & Model Status"], "md_libs"),

    code([
        "print('Backend library status:')\n",
        "lib_status = get_library_status()\n",
        "for model, lib in lib_status.items():\n",
        "    sym = '✓' if 'sklearn' not in lib.lower() or lib == 'sklearn' else '⚠'\n",
        "    print(f'  {sym}  {MODEL_DISPLAY_NAMES[model]:<26}: {lib}')\n",
    ], "cell_libs"),

    # ── 3. Load Datasets ───────────────────────────────────────────────────────
    md(["## 3. Load Preprocessed Datasets"], "md_load"),

    code([
        "X_train_A, X_test_A, y_train, y_test = load_track_data('A', PROCESSED_DIR)\n",
        "X_train_B, X_test_B, _, _            = load_track_data('B', PROCESSED_DIR)\n",
        "\n",
        "print(f'Track A — train: {X_train_A.shape}  test: {X_test_A.shape}')\n",
        "print(f'Track B — train: {X_train_B.shape}  test: {X_test_B.shape}')\n",
        "print(f'y_train: {len(y_train):,}  y_test: {len(y_test):,}')\n",
        "\n",
        "ph_pct = (y_train == 0).mean() * 100\n",
        "lg_pct = (y_train == 1).mean() * 100\n",
        "print(f'Class balance — phishing: {ph_pct:.1f}%  legitimate: {lg_pct:.1f}%')\n",
    ], "cell_load"),

    # ── 4. Track A Training ────────────────────────────────────────────────────
    md(["## 4. Train Track A Models (57 features, includes URLSimilarityIndex)"], "md_train_a"),

    code([
        "# Note: Track A includes URLSimilarityIndex which has leakage risk.\n",
        "# Results here represent the UPPER BOUND / ceiling experiment.\n",
        "print('Training Track A models (4 × 5-fold CV) ...')\n",
        "print('This may take several minutes.\\n')\n",
        "\n",
        "models_A    = get_all_models(track='A')\n",
        "results_A   = run_track_training(\n",
        "    track         = 'A',\n",
        "    models_dict   = models_A,\n",
        "    processed_dir = PROCESSED_DIR,\n",
        "    models_out    = MODELS_DIR,\n",
        "    run_cv        = True,\n",
        "    cv_splits     = 5,\n",
        ")\n",
        "print(f'\\nTrack A training complete: {len(results_A)} models')\n",
    ], "cell_train_a"),

    code([
        "# Quick preview of Track A results\n",
        "print('Track A test-set results:')\n",
        "for r in results_A:\n",
        "    print(f\"  {r['model']:<26}  AUC={r['roc_auc']:.4f}  \"\n",
        "          f\"F1={r['f1']:.4f}  \"\n",
        "          f\"CV={r.get('cv_mean_roc_auc',0):.4f}±{r.get('cv_std_roc_auc',0):.4f}  \"\n",
        "          f\"t={r['training_time_s']:.1f}s\")\n",
    ], "cell_preview_a"),

    # ── 5. Track B Training ────────────────────────────────────────────────────
    md(["## 5. Train Track B Models (56 features, leakage-aware — PRIMARY)"], "md_train_b"),

    code([
        "# Track B excludes URLSimilarityIndex — this is the honest benchmark\n",
        "# and the model used for all downstream SHAP / LIME / bias analysis.\n",
        "print('Training Track B models (4 × 5-fold CV) ...')\n",
        "print('This may take several minutes.\\n')\n",
        "\n",
        "models_B    = get_all_models(track='B')\n",
        "results_B   = run_track_training(\n",
        "    track         = 'B',\n",
        "    models_dict   = models_B,\n",
        "    processed_dir = PROCESSED_DIR,\n",
        "    models_out    = MODELS_DIR,\n",
        "    run_cv        = True,\n",
        "    cv_splits     = 5,\n",
        ")\n",
        "print(f'\\nTrack B training complete: {len(results_B)} models')\n",
    ], "cell_train_b"),

    code([
        "print('Track B test-set results:')\n",
        "for r in results_B:\n",
        "    print(f\"  {r['model']:<26}  AUC={r['roc_auc']:.4f}  \"\n",
        "          f\"F1={r['f1']:.4f}  \"\n",
        "          f\"CV={r.get('cv_mean_roc_auc',0):.4f}±{r.get('cv_std_roc_auc',0):.4f}  \"\n",
        "          f\"t={r['training_time_s']:.1f}s\")\n",
    ], "cell_preview_b"),

    # ── 6. Benchmark Table ─────────────────────────────────────────────────────
    md(["## 6. Unified Benchmark Table"], "md_bench"),

    code([
        "benchmark_df = create_benchmark_table(results_A, results_B)\n",
        "print(f'Benchmark table shape: {benchmark_df.shape}')\n",
        "display(benchmark_df[['model','track','accuracy','precision',\n",
        "                       'recall','f1','roc_auc',\n",
        "                       'cv_mean_roc_auc','cv_std_roc_auc',\n",
        "                       'training_time_s']])\n",
    ], "cell_bench"),

    code([
        "# Save benchmark and ranking CSVs\n",
        "csv_paths = save_benchmark(benchmark_df, REPORTS_DIR)\n",
        "print(f\"Saved: {csv_paths['benchmark_path'].name}\")\n",
        "print(f\"Saved: {csv_paths['ranking_path'].name}\")\n",
        "log_benchmark_table(benchmark_df)\n",
    ], "cell_bench_save"),

    # ── 7. Model Selection ─────────────────────────────────────────────────────
    md(["## 7. Best Model Selection"], "md_select"),

    code([
        "best_A_id, best_A_row = identify_best_model(benchmark_df, track='A')\n",
        "best_B_id, best_B_row = identify_best_model(benchmark_df, track='B')\n",
        "\n",
        "print(f'Best Track A model: {best_A_row[\"model\"]}')\n",
        "print(f'  ROC AUC : {best_A_row[\"roc_auc\"]:.6f}')\n",
        "print(f'  F1 Score: {best_A_row[\"f1\"]:.6f}')\n",
        "print()\n",
        "print(f'Best Track B model (PRIMARY): {best_B_row[\"model\"]}')\n",
        "print(f'  ROC AUC : {best_B_row[\"roc_auc\"]:.6f}')\n",
        "print(f'  F1 Score: {best_B_row[\"f1\"]:.6f}')\n",
    ], "cell_select"),

    code([
        "# Extract fitted model objects for downstream use\n",
        "# These are the exact objects passed to M6.1, M7.1, M8.1, M9, M10\n",
        "fitted_models_A = {r['model_id']: r['fitted_model'] for r in results_A}\n",
        "fitted_models_B = {r['model_id']: r['fitted_model'] for r in results_B}\n",
        "\n",
        "best_model_A = fitted_models_A[best_A_id]\n",
        "best_model_B = fitted_models_B[best_B_id]\n",
        "\n",
        "print(f'best_model_A type : {type(best_model_A).__name__}')\n",
        "print(f'best_model_B type : {type(best_model_B).__name__}')\n",
        "print(f'feature_names_A   : {len(X_train_A.columns)} features')\n",
        "print(f'feature_names_B   : {len(X_train_B.columns)} features')\n",
    ], "cell_objects"),

    # ── 8. Visualisations ──────────────────────────────────────────────────────
    md(["## 8. Benchmark Visualisations"], "md_plots"),

    code([
        "saved_plots = generate_visualizations(benchmark_df, PLOTS_DIR)\n",
        "print(f'Saved {len(saved_plots)} visualisation plots to {PLOTS_DIR}')\n",
        "for p in saved_plots:\n",
        "    print(f'  ✓  {p.name}')\n",
    ], "cell_plots"),

    # ── 9. Leakage Impact ─────────────────────────────────────────────────────
    md(["## 9. Leakage Impact — Track A vs Track B"], "md_leakage"),

    code([
        "print('AUC delta (Track A - Track B) — leakage contribution:')\n",
        "for mid in ['logistic_regression','random_forest','xgboost','lightgbm']:\n",
        "    row_a = benchmark_df[(benchmark_df['track']=='A') &\n",
        "                         (benchmark_df['model']==MODEL_DISPLAY_NAMES[mid])]\n",
        "    row_b = benchmark_df[(benchmark_df['track']=='B') &\n",
        "                         (benchmark_df['model']==MODEL_DISPLAY_NAMES[mid])]\n",
        "    if not row_a.empty and not row_b.empty:\n",
        "        delta = row_a.iloc[0]['roc_auc'] - row_b.iloc[0]['roc_auc']\n",
        "        flag  = '  ⚠ large leakage' if delta > 0.01 else ''\n",
        "        print(f\"  {MODEL_DISPLAY_NAMES[mid]:<26} Δ={delta:+.4f}{flag}\")\n",
    ], "cell_leakage"),

    # ── 10. Confusion Matrices ────────────────────────────────────────────────
    md(["## 10. Confusion Matrices (Track B)"], "md_cm"),

    code([
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "from sklearn.metrics import ConfusionMatrixDisplay\n",
        "\n",
        "fig, axes = plt.subplots(1, 4, figsize=(18, 4))\n",
        "fig.suptitle('Confusion Matrices — Track B (test set)', fontweight='700')\n",
        "\n",
        "for ax, r in zip(axes, results_B):\n",
        "    cm = np.array(r['confusion_matrix'])\n",
        "    disp = ConfusionMatrixDisplay(confusion_matrix=cm,\n",
        "                                  display_labels=['Phishing','Legitimate'])\n",
        "    disp.plot(ax=ax, colorbar=False, cmap='Blues')\n",
        "    ax.set_title(r['model'], fontsize=10, fontweight='600')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(PLOTS_DIR / 'confusion_matrices_B.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('Saved → confusion_matrices_B.png')\n",
    ], "cell_cm"),

    # ── 11. Save Artifacts ────────────────────────────────────────────────────
    md(["## 11. Save All Artifacts"], "md_save"),

    code([
        "# Models are already saved during training.\n",
        "# Verify all .pkl files exist.\n",
        "print('Verifying saved model artifacts:')\n",
        "for track in ['A','B']:\n",
        "    for mid in ['logistic_regression','random_forest','xgboost','lightgbm']:\n",
        "        p = MODELS_DIR / f'track_{track}' / f'{mid}.pkl'\n",
        "        sym = '✓' if p.exists() else '✗ MISSING'\n",
        "        print(f'  {sym}  {p.relative_to(PROJECT_ROOT)}')\n",
    ], "cell_verify"),

    code([
        "# Generate HTML training report\n",
        "report_path = generate_training_report(\n",
        "    df          = benchmark_df,\n",
        "    best_A_row  = best_A_row,\n",
        "    best_B_row  = best_B_row,\n",
        "    output_path = REPORTS_DIR / 'training_summary.html',\n",
        "    plots_dir   = PLOTS_DIR,\n",
        ")\n",
        "print(f'Report saved: {report_path}')\n",
    ], "cell_report"),

    # ── 12. Downstream Interface ──────────────────────────────────────────────
    md(["## 12. Downstream Module Interface"], "md_interface"),

    code([
        "print('=' * 65)\n",
        "print('DOWNSTREAM MODULE INTERFACE')\n",
        "print('=' * 65)\n",
        "print()\n",
        "print('Objects ready for M6.1 (Evaluation):')\n",
        "print(f'  all_models_A   : dict with {len(fitted_models_A)} fitted models')\n",
        "print(f'  all_models_B   : dict with {len(fitted_models_B)} fitted models')\n",
        "print(f'  benchmark_df   : DataFrame {benchmark_df.shape}')\n",
        "print(f'  best_model_A   : {type(best_model_A).__name__} (Track A winner)')\n",
        "print(f'  best_model_B   : {type(best_model_B).__name__} (Track B winner)')\n",
        "print()\n",
        "print('Objects ready for M7.1 (SHAP):')\n",
        "print(f'  best_model_B     : {type(best_model_B).__name__}')\n",
        "print(f'  X_train_B        : {X_train_B.shape}')\n",
        "print(f'  X_test_B         : {X_test_B.shape}')\n",
        "print(f'  feature_names_B  : {list(X_train_B.columns)[:5]} ...')\n",
        "print()\n",
        "print('Objects ready for M8.1 (LIME):')\n",
        "print(f'  best_model_B.predict_proba : callable')\n",
        "print(f'  X_train_B.values           : numpy array {X_train_B.shape}')\n",
        "print(f'  X_test_B.values            : numpy array {X_test_B.shape}')\n",
        "print(f'  feature_names_B            : list of {len(X_train_B.columns)} names')\n",
    ], "cell_interface"),

    # ── 13. Conclusions ───────────────────────────────────────────────────────
    md(["## 13. Conclusions"], "md_conclusions"),

    code([
        "print('=' * 65)\n",
        "print('MODULE M5.1 / M5.2 COMPLETE')\n",
        "print('=' * 65)\n",
        "print()\n",
        "print(f'  Models trained  : {len(results_A) + len(results_B)} (4 A + 4 B)')\n",
        "print(f'  CV folds        : 5-fold stratified per model')\n",
        "print(f'  Best Track A    : {best_A_row[\"model\"]} (AUC={best_A_row[\"roc_auc\"]:.4f})')\n",
        "print(f'  Best Track B    : {best_B_row[\"model\"]} (AUC={best_B_row[\"roc_auc\"]:.4f})')\n",
        "print()\n",
        "print('Saved artifacts:')\n",
        "print('  outputs/models/track_A/*.pkl  (4 models)')\n",
        "print('  outputs/models/track_B/*.pkl  (4 models)')\n",
        "print('  outputs/reports/model_benchmark.csv')\n",
        "print('  outputs/reports/model_ranking.csv')\n",
        "print('  outputs/reports/training_summary.html')\n",
        "print('  outputs/plots/training/*.png  (5 plots)')\n",
        "print()\n",
        "print('Next: M6.1 — Model Evaluation')\n",
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

out = Path(__file__).resolve().parent / "notebooks" / "05_model_training.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Notebook written → {out}")
