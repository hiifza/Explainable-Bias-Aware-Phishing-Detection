"""src/bias/shap_bias_analysis.py — SHAP importance per group, rank shifts, plots."""
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
logger = get_logger(__name__)

def _setup():
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams.update({"figure.facecolor":"white","savefig.dpi":150})

def compute_group_shap(shap_values, feature_names, group_mask, group_label):
    group_mask = np.asarray(group_mask, dtype=bool)
    n_shap = shap_values.shape[0]
    mask_aligned = group_mask[:n_shap] if len(group_mask)>n_shap else group_mask
    if mask_aligned.sum()==0:
        return {"group":group_label,"n":0,
                "mean_abs_shap":pd.Series(np.zeros(len(feature_names)),index=feature_names),
                "top10":[]}
    sv = shap_values[mask_aligned]
    mean_abs = np.abs(sv).mean(axis=0)
    ms = pd.Series(mean_abs, index=feature_names)
    top10 = [(f,float(v)) for f,v in ms.sort_values(ascending=False).head(10).items()]
    return {"group":group_label,"n":int(mask_aligned.sum()),"mean_abs_shap":ms,"top10":top10}

def compute_rank_shifts(group_shap, global_ranking):
    global_ranks = dict(zip(global_ranking["feature"], global_ranking["rank"]))
    gs = (group_shap["mean_abs_shap"].sort_values(ascending=False)
          .reset_index().rename(columns={"index":"feature",0:"mean_abs_shap_group"}))
    gs.columns = ["feature","mean_abs_shap_group"]
    gs["group_rank"] = range(1, len(gs)+1)
    gs["global_rank"] = gs["feature"].map(global_ranks)
    gs["rank_shift"] = gs["global_rank"] - gs["group_rank"]
    return gs.head(20).reset_index(drop=True)

def plot_group_shap_importance(group_shap_list, dim_name, plots_dir, top_k=10):
    _setup()
    groups = [g for g in group_shap_list if g["n"]>=10]
    if not groups: return plots_dir/f"shap_{dim_name}.png"
    n_g = len(groups)
    fig,axes = plt.subplots(1,n_g,figsize=(n_g*6,max(5,top_k*0.42)))
    if n_g==1: axes=[axes]
    palette=["#E24B4A","#185FA5","#0F6E56","#854F0B","#533AB7","#EF9F27"]
    for ax,gs,color in zip(axes,groups,palette):
        top = gs["top10"][:top_k]; feats=[f for f,_ in top][::-1]; vals=[v for _,v in top][::-1]
        ax.barh(feats,vals,color=color,edgecolor="white",linewidth=0.5)
        ax.set_title(f"{gs['group']}\n(n={gs['n']:,})",fontsize=10,fontweight="600")
        ax.set_xlabel("Mean |SHAP|"); sns.despine(ax=ax)
    fig.suptitle(f"SHAP Importance by Group — {dim_name}",fontsize=12,fontweight="700")
    plt.tight_layout()
    out = plots_dir/f"shap_{dim_name}.png"; plots_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=150,bbox_inches="tight",facecolor="white"); plt.close(fig)
    logger.info(f"Saved: {out.name}"); return out

def run_shap_bias_analysis(shap_values, feature_names, group_results, global_rank_df,
                            plots_dir="outputs/plots/bias/shap_groups",
                            reports_dir="outputs/reports"):
    plots_dir=Path(plots_dir); reports_dir=Path(reports_dir)
    plots_dir.mkdir(parents=True,exist_ok=True); reports_dir.mkdir(parents=True,exist_ok=True)
    dim_names=["url_length","https","tld","domain_length","ext_resources"]
    all_shift_rows=[]; shap_group_stats={}; importance_plots={}
    for dim in dim_names:
        groups_series = group_results.get(f"{dim}_groups")
        if groups_series is None: continue
        group_labels=groups_series.unique(); group_shap_list=[]; shift_data={}
        for grp_label in sorted(group_labels):
            mask=(groups_series==grp_label).values
            gs=compute_group_shap(shap_values,feature_names,mask,grp_label)
            group_shap_list.append(gs)
            if gs["n"]>=10:
                shifts=compute_rank_shifts(gs,global_rank_df)
                shift_data[grp_label]=shifts
                for _,row in shifts.iterrows():
                    all_shift_rows.append({"dimension":dim,"group":grp_label,
                        "feature":row["feature"],"global_rank":row.get("global_rank"),
                        "group_rank":row["group_rank"],"rank_shift":row["rank_shift"],
                        "mean_abs_shap":row["mean_abs_shap_group"]})
        shap_group_stats[dim]=group_shap_list
        importance_plots[dim]=plot_group_shap_importance(group_shap_list,dim,plots_dir)
    shap_bias_df=pd.DataFrame(all_shift_rows)
    shap_bias_df.to_csv(reports_dir/"shap_bias_analysis.csv",index=False)
    logger.info(f"Saved: shap_bias_analysis.csv ({len(shap_bias_df)} rows)")
    return {"shap_group_stats":shap_group_stats,"importance_plots":importance_plots,
            "shap_bias_df":shap_bias_df}
