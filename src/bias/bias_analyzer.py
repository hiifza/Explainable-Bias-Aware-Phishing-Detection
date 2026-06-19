"""src/bias/bias_analyzer.py — M9 main orchestrator."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.utils.logger import get_logger
from src.training.model_saver import load_all_models
from src.training.model_registry import MODEL_DISPLAY_NAMES
from src.bias.group_evaluator import run_group_evaluation
from src.bias.shap_bias_analysis import run_shap_bias_analysis
logger = get_logger(__name__)

def _setup():
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({"figure.facecolor":"white","savefig.dpi":150})

def load_bias_inputs(processed_dir="data/processed", models_dir="outputs/models"):
    processed_dir=Path(processed_dir); models_dir=Path(models_dir)
    bench  = pd.read_csv(ROOT/"outputs"/"reports"/"evaluation_metrics.csv")
    best_B = bench[bench["track"]=="B"].sort_values("roc_auc",ascending=False).iloc[0]
    id_map = {v:k for k,v in MODEL_DISPLAY_NAMES.items()}
    best_id= id_map.get(best_B["model"],"logistic_regression")
    model  = load_all_models(models_dir,"B")[best_id]
    logger.info(f"BIAS_model: {best_B['model']} ({type(model).__name__})")
    X_test_B = pd.read_csv(processed_dir/"track_B"/"X_test.csv")
    y_test   = pd.read_csv(processed_dir/"y_test.csv")["label"]
    raw_df   = pd.read_csv(processed_dir/"track_B"/"raw_X_test.csv")
    raw_df   = raw_df.iloc[:len(X_test_B)].reset_index(drop=True)
    y_pred   = model.predict(X_test_B)
    y_proba  = model.predict_proba(X_test_B)[:,1]
    return {"model":model,"X_test_B":X_test_B,"y_test":y_test,
            "y_pred":y_pred,"y_proba":y_proba,
            "raw_test_df":raw_df,"feature_names":list(X_test_B.columns)}

def plot_group_performance(metrics_df, dim, metric, plots_dir):
    _setup()
    df = metrics_df[metrics_df["dimension"]==dim].dropna(subset=[metric])
    if df.empty: return plots_dir/f"{dim}_{metric}.png"
    df = df.sort_values(metric, ascending=True)
    low_bad = metric not in ("fpr","fnr")
    t = {"accuracy":0.90,"f1":0.90,"roc_auc":0.95,"fpr":0.10,"fnr":0.10}.get(metric,0.80)
    colors = [("#E24B4A" if (low_bad and v<t) or (not low_bad and v>t) else "#1D9E75")
              for v in df[metric]]
    fig,ax = plt.subplots(figsize=(9,max(4,len(df)*0.42)))
    bars = ax.barh(df["group"],df[metric],color=colors,edgecolor="white",linewidth=0.5)
    for bar,v in zip(bars,df[metric]):
        ax.text(bar.get_width()+0.004,bar.get_y()+bar.get_height()/2,
                f"{v:.4f}",va="center",fontsize=8)
    ax.set_xlabel(metric.upper()); sns.despine(ax=ax)
    ax.set_title(f"{metric.upper()} by Group — {dim.replace('_',' ').title()}",
                 fontsize=12,fontweight="700")
    plt.tight_layout()
    out=plots_dir/f"{dim}_{metric}.png"; plots_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=150,bbox_inches="tight",facecolor="white"); plt.close(fig)
    return out

def plot_fpr_fnr_disparity(all_metrics_df, plots_dir):
    _setup(); dims=all_metrics_df["dimension"].unique()
    fpr_d=[]; fnr_d=[]
    for dim in dims:
        sub=all_metrics_df[all_metrics_df["dimension"]==dim]
        fpr_d.append(sub["fpr"].max()-sub["fpr"].min() if "fpr" in sub.columns else 0)
        fnr_d.append(sub["fnr"].max()-sub["fnr"].min() if "fnr" in sub.columns else 0)
    x=np.arange(len(dims)); w=0.38
    fig,ax=plt.subplots(figsize=(10,5))
    ax.bar(x-w/2,fpr_d,width=w,label="FPR disparity",color="#EF9F27",edgecolor="white")
    ax.bar(x+w/2,fnr_d,width=w,label="FNR disparity",color="#E24B4A",edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([d.replace("_"," ").title() for d in dims],rotation=20,ha="right")
    ax.set_ylabel("Disparity (max-min)"); ax.legend(fontsize=10)
    ax.set_title("FPR and FNR Disparity Across Bias Dimensions",fontsize=12,fontweight="700")
    sns.despine(ax=ax); plt.tight_layout()
    out=plots_dir/"fpr_fnr_disparity.png"; plots_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=150,bbox_inches="tight",facecolor="white"); plt.close(fig)
    logger.info(f"Saved: {out.name}"); return out

def plot_tld_fairness(tld_metrics_df, plots_dir):
    _setup()
    df=(tld_metrics_df.sort_values("n",ascending=False).head(20).sort_values("fpr",ascending=True))
    if df.empty: return plots_dir/"tld_fairness.png"
    fig,axes=plt.subplots(1,2,figsize=(14,max(6,len(df)*0.38)))
    for ax,metric,color,title in [(axes[0],"fpr","#EF9F27","FPR"),(axes[1],"fnr","#E24B4A","FNR")]:
        sub=df.dropna(subset=[metric]).sort_values(metric,ascending=True)
        bars=ax.barh(sub["group"],sub[metric],color=color,edgecolor="white",linewidth=0.5)
        for bar,v in zip(bars,sub[metric]):
            ax.text(bar.get_width()+0.004,bar.get_y()+bar.get_height()/2,f"{v:.3f}",va="center",fontsize=7)
        ax.set_xlabel(metric.upper()); ax.set_title(f"TLD {title}",fontsize=11,fontweight="600")
        sns.despine(ax=ax)
    fig.suptitle("TLD Group Fairness",fontsize=13,fontweight="700"); plt.tight_layout()
    out=plots_dir/"tld_fairness.png"; plots_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=150,bbox_inches="tight",facecolor="white"); plt.close(fig)
    logger.info(f"Saved: {out.name}"); return out

def plot_accuracy_heatmap(all_metrics_df, plots_dir):
    _setup()
    metrics=["accuracy","f1","roc_auc","fpr","fnr"]
    dims=sorted(all_metrics_df["dimension"].unique())
    data={}
    for dim in dims:
        sub=all_metrics_df[all_metrics_df["dimension"]==dim]
        data[dim]={m:round(float(sub[m].mean()),4) for m in metrics if m in sub.columns}
    heat_df=pd.DataFrame(data).T.reindex(columns=metrics)
    fig,ax=plt.subplots(figsize=(9,4))
    sns.heatmap(heat_df,ax=ax,cmap="RdYlGn",annot=True,fmt=".3f",
                linewidths=0.4,linecolor="white",cbar_kws={"shrink":0.7},annot_kws={"size":10})
    ax.set_title("Mean Metric by Bias Dimension",fontsize=12,fontweight="700",pad=10)
    ax.tick_params(axis="x",rotation=20); ax.tick_params(axis="y",rotation=0)
    plt.tight_layout()
    out=plots_dir/"accuracy_heatmap.png"; plots_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=150,bbox_inches="tight",facecolor="white"); plt.close(fig)
    logger.info(f"Saved: {out.name}"); return out

def _compute_summary(metrics_df, disparity_df):
    summary={}
    fpr_sub=metrics_df.dropna(subset=["fpr"])
    if not fpr_sub.empty:
        w=fpr_sub.loc[fpr_sub["fpr"].idxmax()]
        summary["highest_fpr_group"]={"group":w["group"],"dimension":w["dimension"],"fpr":round(float(w["fpr"]),4)}
    fnr_sub=metrics_df.dropna(subset=["fnr"])
    if not fnr_sub.empty:
        w=fnr_sub.loc[fnr_sub["fnr"].idxmax()]
        summary["highest_fnr_group"]={"group":w["group"],"dimension":w["dimension"],"fnr":round(float(w["fnr"]),4)}
    if not disparity_df.empty and "metric" in disparity_df.columns:
        fpr_d=disparity_df[disparity_df["metric"]=="fpr"]
        if not fpr_d.empty:
            m=fpr_d.loc[fpr_d["disparity"].idxmax()]
            summary["most_biased_dimension"]={"dimension":m["dimension"],"metric":"fpr",
                "disparity":round(float(m["disparity"]),4),"worst_group":m["worst_group"],"best_group":m["best_group"]}
            mi=fpr_d.loc[fpr_d["disparity"].idxmin()]
            summary["least_biased_dimension"]={"dimension":mi["dimension"],"metric":"fpr","disparity":round(float(mi["disparity"]),4)}
    return summary

def run_full_bias_analysis(processed_dir, models_dir, shap_values, feature_names,
                            global_rank_df, reports_dir="outputs/reports",
                            plots_dir="outputs/plots/bias"):
    reports_dir=Path(reports_dir); plots_dir=Path(plots_dir)
    reports_dir.mkdir(parents=True,exist_ok=True)
    inputs=load_bias_inputs(processed_dir, models_dir)
    y_true=inputs["y_test"].values; y_pred=inputs["y_pred"]; y_proba=inputs["y_proba"]
    raw_df=inputs["raw_test_df"]
    group_results=run_group_evaluation(raw_df,y_true,y_pred,y_proba)
    all_metrics=group_results["all_metrics_df"]; all_disparity=group_results["all_disparity_df"]
    all_metrics.to_csv(reports_dir/"bias_metrics.csv",index=False)
    all_disparity.to_csv(reports_dir/"group_disparities.csv",index=False)
    perf_plots={}
    for dim in ["url_length","https","tld","domain_length","ext_resources"]:
        dim_dir=plots_dir/dim; dim_dir.mkdir(parents=True,exist_ok=True)
        perf_plots[dim]=[plot_group_performance(all_metrics,dim,m,dim_dir)
                         for m in ["accuracy","f1","fpr","fnr","roc_auc"]]
    disp_dir=plots_dir/"disparity"; disp_dir.mkdir(parents=True,exist_ok=True)
    disp_plot=plot_fpr_fnr_disparity(all_metrics,disp_dir)
    tld_plot=plot_tld_fairness(all_metrics[all_metrics["dimension"]=="tld"],plots_dir/"tld")
    heat_plot=plot_accuracy_heatmap(all_metrics,disp_dir)
    shap_bias_r=run_shap_bias_analysis(shap_values,feature_names,group_results,
                                        global_rank_df,plots_dir/"shap_groups",reports_dir)
    summary=_compute_summary(all_metrics,all_disparity)
    return {"inputs":inputs,"group_results":group_results,"all_metrics_df":all_metrics,
            "all_disparity_df":all_disparity,"perf_plots":perf_plots,"disp_plot":disp_plot,
            "tld_plot":tld_plot,"heat_plot":heat_plot,"shap_bias_r":shap_bias_r,"summary":summary}
