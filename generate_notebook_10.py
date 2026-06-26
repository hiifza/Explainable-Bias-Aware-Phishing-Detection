"""generate_notebook_10.py — builds notebooks/10_blindspot_analysis.ipynb"""
import json
from pathlib import Path

def md(s,cid=""):
    return {"cell_type":"markdown","id":cid or s[0][:16].strip().replace(" ","_").lower(),"metadata":{},"source":s}
def code(s,cid=""):
    return {"cell_type":"code","execution_count":None,"id":cid or s[0][:16].strip().replace(" ","_").lower(),"metadata":{},"outputs":[],"source":s}

cells=[
md(["# Module M10 — Failure Intelligence Engine\n","\n",
    "**Project:** Explainable and Bias-Aware ML for Phishing Website Detection  \n",
    "**Objective:** Discover hidden weaknesses, blind spots, and deployment risks  \n","\n",
    "### Analysis pipeline\n",
    "1. Failure case extraction (4 tiers)  \n",
    "2. Confidence Reliability Engine (Green/Yellow/Red zones)  \n",
    "3. Blind Spot Severity Scoring + Top-20 ranking  \n",
    "4. Failure Archetype Discovery (auto-clustering, no hardcoded categories)  \n",
    "5. Cluster visualisations (PCA + UMAP if available)  \n",
    "6. SHAP failure analysis + dominant feature masking  \n",
    "7. LIME failure analysis  \n",
    "8. SHAP-LIME reliability correlation  \n",
    "9. HTML report + dashboard-ready exports  \n"],"title"),

md(["## 0. Setup"],"md_s0"),
code(["import sys\nfrom pathlib import Path\nPROJECT_ROOT=Path().resolve().parent\n"
      "if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))\n"
      "print(f'Root: {PROJECT_ROOT}')"],"c_root"),
code(["import warnings; warnings.filterwarnings('ignore')\n"
      "import numpy as np; import pandas as pd\n"
      "import matplotlib.pyplot as plt; import seaborn as sns\n"
      "from src.utils.logger import get_logger\n"
      "from src.blindspots.failure_case_extractor import extract_failure_cases\n"
      "from src.blindspots.failure_archetype_discovery import discover_archetypes\n"
      "from src.blindspots.severity_ranker import compute_severity_scores, rank_blind_spots\n"
      "from src.blindspots.uncertainty_analysis import run_uncertainty_analysis\n"
      "from src.blindspots.shap_failure_analysis import run_shap_failure_analysis\n"
      "from src.blindspots.lime_failure_analysis import run_lime_failure_analysis\n"
      "from src.blindspots.reliability_analysis import run_reliability_analysis\n"
      "from src.blindspots.cluster_visualization import run_cluster_visualization\n"
      "from src.blindspots.blindspot_analyzer import load_m10_inputs, run_failure_intelligence_engine\n"
      "from src.blindspots.blindspot_report import generate_blindspot_report\n"
      "from src.explainability.shap_explainer import compute_shap_values\n"
      "logger=get_logger('notebook.10_blindspot')\n"
      "sns.set_theme(style='whitegrid',font_scale=1.05)\nplt.rcParams['figure.dpi']=120\n"
      "print('Imports OK ✓')"],"c_imports"),

md(["## 1. Paths"],"md_s1"),
code(["MODELS_DIR=PROJECT_ROOT/'outputs'/'models'\n"
      "PROCESSED_DIR=PROJECT_ROOT/'data'/'processed'\n"
      "REPORTS_DIR=PROJECT_ROOT/'outputs'/'reports'\n"
      "PLOTS_BS=PROJECT_ROOT/'outputs'/'plots'/'blindspot'\n"
      "for d in ['clusters','confidence','shap_failure','failure_archetypes','reliability']:\n"
      "    (PLOTS_BS/d).mkdir(parents=True,exist_ok=True)\n"
      "print('Paths OK ✓')"],"c_paths"),

md(["## 2. Load Inputs"],"md_s2"),
code(["inputs=load_m10_inputs(PROCESSED_DIR,MODELS_DIR,REPORTS_DIR)\n"
      "print(f'Model: {type(inputs[\"model\"]).__name__}')\n"
      "print(f'X_test: {inputs[\"X_test_B\"].shape}')\n"
      "y_true=inputs['y_test']; y_pred=inputs['y_pred']; y_proba=inputs['y_proba']\n"
      "X_test_B=inputs['X_test_B']; feature_names=inputs['feature_names']\n"
      "agreement_df=inputs.get('agreement_df',pd.DataFrame())\n"
      "print(f'Hard errors: {(y_pred!=y_true).sum()}  '\n"
      "      f'FP={(( y_pred==0)&(y_true==1)).sum()}  '\n"
      "      f'FN={((y_pred==1)&(y_true==0)).sum()}')"],"c_load"),

md(["## 3. Compute SHAP Values"],"md_s3"),
code(["X_train_B=pd.read_csv(PROCESSED_DIR/'track_B'/'X_train.csv')\n"
      "print('Computing SHAP (1000 samples) ...')\n"
      "shap_result=compute_shap_values(inputs['model'],X_train_B,X_test_B,\n"
      "    feature_names,sample_n=1000,random_state=42)\n"
      "print(f'SHAP: {shap_result.shap_values.shape}')"],"c_shap"),

md(["## 4. Failure Case Extraction (4 tiers)"],"md_s4"),
code(["fcs=extract_failure_cases(y_true,y_pred,y_proba,X_test_B,\n"
      "    agreement_df=agreement_df,feature_names=feature_names)\n"
      "print(f'Total failure cases: {len(fcs)}')\n"
      "print(f'  Tier 1 (hard errors): {len(fcs.errors)}')\n"
      "print(f'  Tier 2 (uncertain)  : {len(fcs.yellow_zone)+len(fcs.red_zone)}')\n"
      "print(f'  FP: {len(fcs.fp_cases)}  FN: {len(fcs.fn_cases)}')\n"
      "print(f'  Red zone: {len(fcs.red_zone)}  Yellow: {len(fcs.yellow_zone)}')"],"c_fce"),
code(["# Preview failure case DataFrame\ndisplay(fcs.df.head(10))"],"c_fce_prev"),

md(["## 5. Confidence Reliability Engine"],"md_s5"),
code(["uncertainty_r=run_uncertainty_analysis(y_true,y_pred,y_proba,\n"
      "    plots_dir=PLOTS_BS/'confidence')\n"
      "zs=uncertainty_r['zone_stats']\n"
      "for zone in ['green','yellow','red']:\n"
      "    z=zs[zone]\n"
      "    print(f'  {zone.upper():<6}: n={z[\"n\"]:>6,}  '\n"
      "          f'error_rate={z[\"error_rate\"]:.6f}  '\n"
      "          f'mean_conf={z[\"mean_confidence\"]:.4f}')"],"c_unc"),
code(["from IPython.display import Image\n"
      "display(Image(str(uncertainty_r['confidence_dist_plot'])))\n"
      "display(Image(str(uncertainty_r['zone_error_plot'])))"],"c_unc_plot"),

md(["## 6. Severity Scoring & Top-20 Blind Spots"],"md_s6"),
code(["severity_df=compute_severity_scores(fcs,shap_result.shap_values,feature_names)\n"
      "top_20_bs=rank_blind_spots(severity_df,top_n=20)\n"
      "severity_df.to_csv(REPORTS_DIR/'blindspot_severity.csv',index=False)\n"
      "top_20_bs.to_csv(REPORTS_DIR/'top20_blind_spots.csv',index=False)\n"
      "print(f'Severity scores: {len(severity_df)}')\n"
      "print(f'Max severity: {severity_df[\"severity_score_norm\"].max():.4f}')\n"
      "print(f'Mean severity: {severity_df[\"severity_score_norm\"].mean():.4f}')"],"c_sev"),
code(["print('Top-20 Blind Spots:')\n"
      "display(top_20_bs[['severity_rank','sample_idx','confidence_zone',\n"
      "                    'is_error','confidence','severity_score_norm','risk_level']].head(10))"],"c_top20"),

md(["## 7. Failure Archetype Discovery (auto-clustering)"],"md_s7"),
code(["archetype_r=discover_archetypes(fcs,X_test_B,feature_names,\n"
      "    shap_result.shap_values,k_range=(2,8))\n"
      "archetype_r.archetype_df.to_csv(REPORTS_DIR/'failure_archetypes.csv',index=False)\n"
      "print(f'Archetypes: {archetype_r.n_clusters}  silhouette={archetype_r.silhouette_score:.4f}')\n"
      "for m in archetype_r.cluster_meta:\n"
      "    print(f'  [{m[\"cluster\"]}] {m[\"label\"]} — n={m[\"n_samples\"]}  errors={m[\"n_errors\"]}')"],"c_arch"),

md(["## 8. Cluster Visualisations"],"md_s8"),
code(["cluster_r=run_cluster_visualization(archetype_r,fcs,severity_df,\n"
      "    X_test_B,feature_names,plots_dir=PLOTS_BS/'clusters')\n"
      "display(Image(str(cluster_r['pca_plot'])))\n"
      "display(Image(str(cluster_r['heatmap_plot'])))\n"
      "display(Image(str(cluster_r['density_plot'])))"],"c_clust"),

md(["## 9. SHAP Failure Analysis"],"md_s9"),
code(["shap_fail_r=run_shap_failure_analysis(\n"
      "    shap_result.shap_values,feature_names,fcs,y_true,y_proba,\n"
      "    plots_dir=PLOTS_BS/'shap_failure',reports_dir=REPORTS_DIR)\n"
      "print('Top failure-zone features:')\n"
      "print(shap_fail_r['comparison']['top_failure_features'][:5])\n"
      "display(Image(str(shap_fail_r['comparison_plot'])))\n"
      "display(Image(str(shap_fail_r['masking_plot'])))"],"c_shap_f"),

md(["## 10. LIME Failure Analysis"],"md_s10"),
code(["lime_fail_r=run_lime_failure_analysis(agreement_df,fcs,\n"
      "    plots_dir=PLOTS_BS/'failure_archetypes',reports_dir=REPORTS_DIR)\n"
      "print(f'LIME failure features: {len(lime_fail_r[\"lime_freq_df\"])} rows')\n"
      "if not lime_fail_r['lime_freq_df'].empty:\n"
      "    display(lime_fail_r['lime_freq_df'].head(10))"],"c_lime_f"),

md(["## 11. SHAP-LIME Reliability Correlation"],"md_s11"),
code(["reliability_r=run_reliability_analysis(agreement_df,severity_df,\n"
      "    uncertainty_r['zone_stats'],plots_dir=PLOTS_BS/'reliability',\n"
      "    reports_dir=REPORTS_DIR)\n"
      "rs=reliability_r['reliability_stats']\n"
      "print(f'Q1: {rs.get(\"q1_answer\",\"—\")}')\n"
      "print(f'Q2: {rs.get(\"q2_answer\",\"—\")}')\n"
      "print(f'Q3: {rs.get(\"q3_answer\",\"—\")}')\n"
      "print(f'Corr(agreement, severity) = {rs.get(\"corr_agreement_severity\",0):.4f}')\n"
      "if reliability_r.get('agreement_plot'):\n"
      "    display(Image(str(reliability_r['agreement_plot'])))"],"c_rel"),

md(["## 12. HTML Report & Artifact Verification"],"md_s12"),
code(["m10_results={'fcs':fcs,'severity_df':severity_df,'top_20_bs':top_20_bs,\n"
      "    'archetype_r':archetype_r,'uncertainty_r':uncertainty_r,'cluster_r':cluster_r,\n"
      "    'shap_fail_r':shap_fail_r,'lime_fail_r':lime_fail_r,'reliability_r':reliability_r,\n"
      "    'inputs':inputs,\n"
      "    'summary':{'n_test':len(y_true),'n_errors':int((y_pred!=y_true).sum()),\n"
      "               'n_fp':int(((y_pred==0)&(y_true==1)).sum()),\n"
      "               'n_fn':int(((y_pred==1)&(y_true==0)).sum()),\n"
      "               'error_rate':round(float((y_pred!=y_true).mean()),8),\n"
      "               'n_failure_cases':len(fcs),'n_archetypes':archetype_r.n_clusters,\n"
      "               'silhouette':round(archetype_r.silhouette_score,4),\n"
      "               'green_zone_n':zs['green']['n'],'yellow_zone_n':zs['yellow']['n'],\n"
      "               'red_zone_n':zs['red']['n'],\n"
      "               'max_severity':round(float(severity_df['severity_score_norm'].max()),4),\n"
      "               'mean_severity':round(float(severity_df['severity_score_norm'].mean()),4),\n"
      "               'top_archetype':archetype_r.cluster_meta[0]['label'] if archetype_r.cluster_meta else '—',\n"
      "               'shap_top_failure_feat':shap_fail_r['comparison']['top_failure_features'][0] if shap_fail_r['comparison'].get('top_failure_features') else '—'}}\n"
      "\nreport_path=generate_blindspot_report(m10_results,REPORTS_DIR/'blindspot_analysis_report.html',PLOTS_BS)\n"
      "print(f'Report: {report_path}')"],"c_report"),
code(["import pathlib\n"
      "artifacts=['outputs/reports/blindspot_severity.csv','outputs/reports/top20_blind_spots.csv',\n"
      "    'outputs/reports/failure_archetypes.csv','outputs/reports/blindspot_shap_failure.csv',\n"
      "    'outputs/reports/blindspot_analysis_report.html']\n"
      "for rel in artifacts:\n"
      "    p=PROJECT_ROOT/rel\n"
      "    print(f\"  {'✓' if p.exists() else '✗'}  {rel}\")"],"c_verify"),

md(["## 13. Dashboard-Ready Outputs"],"md_s13"),
code(["print('='*65)\nprint('M10 COMPLETE — DASHBOARD-READY OUTPUTS')\nprint('='*65)\n"
      "print(f'  fcs                  : FailureCaseSet — {len(fcs)} cases')\n"
      "print(f'  top_20_bs            : Top-20 blind spots DataFrame')\n"
      "print(f'  archetype_r          : {archetype_r.n_clusters} archetypes discovered')\n"
      "print(f'  severity_df          : {len(severity_df)} severity scores')\n"
      "print(f'  uncertainty_r        : Green/Yellow/Red zone stats')\n"
      "print(f'  reliability_r        : SHAP-LIME correlation analysis')\n"
      "print(f'  cluster_r[pca_plot]  : PCA cluster map for dashboard')\n"
      "print()\nprint('Next: M11 — Streamlit Dashboard')"],"c_dash"),
]

nb={"nbformat":4,"nbformat_minor":5,
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3.11.0"}},"cells":cells}
out=Path(__file__).resolve().parent/"notebooks"/"10_blindspot_analysis.ipynb"
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(nb,indent=1),encoding="utf-8")
print(f"Notebook written → {out}")
