"""src/bias/group_evaluator.py — group assignment and per-dimension evaluation."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.utils.logger import get_logger
from src.bias.fairness_metrics import compute_all_group_metrics, compute_disparity
logger = get_logger(__name__)

GOV_EDU_TLDS   = {".gov",".edu",".mil","gov","edu","mil",".ac","ac"}
COMMERCIAL_TLDS= {".com",".net",".biz",".info","com","net","biz","info"}
COUNTRY_TLDS   = {".uk",".de",".au",".ca",".fr",".jp",".it",".nl",".br",".ru",
                   ".in",".cn",".es",".pl",".se",".no",".ch",".at",".mx",".pt",
                   "uk","de","au","ca","fr","jp","it","nl","br","ru","in","cn","es","pl"}
NEW_GENERIC    = {".io",".app",".dev",".tech",".online",".site",".store",".xyz",
                   "io","app","dev","tech","online","site","store","xyz"}

def assign_url_length_groups(raw_df):
    col = "URLLength" if "URLLength" in raw_df.columns else raw_df.columns[0]
    s = raw_df[col] if col in raw_df.columns else pd.Series(50, index=raw_df.index)
    q33,q67 = float(s.quantile(0.33)), float(s.quantile(0.67))
    def label(v):
        if v<=q33: return f"Short (≤{q33:.0f})"
        if v<=q67: return f"Medium ({q33:.0f}-{q67:.0f})"
        return f"Long (>{q67:.0f})"
    logger.info(f"URL length groups: Q33={q33:.0f} Q67={q67:.0f}")
    return s.apply(label)

def assign_https_groups(raw_df):
    col = "IsHTTPS"
    s = raw_df[col] if col in raw_df.columns else pd.Series(1, index=raw_df.index)
    return s.map({1:"HTTPS",0:"Non-HTTPS"}).fillna("Unknown")

def assign_tld_groups(raw_df, top_n=20):
    col = "TLD"
    if col not in raw_df.columns: return pd.Series("unknown", index=raw_df.index)
    s = raw_df[col].astype(str).str.strip().str.lower()
    top_tlds = s.value_counts().head(top_n).index.tolist()
    def label(tld):
        if tld in top_tlds: return tld
        if tld in GOV_EDU_TLDS or tld.endswith((".gov",".edu",".mil",".ac")): return "gov/edu"
        if tld in COMMERCIAL_TLDS: return "commercial"
        if tld in COUNTRY_TLDS: return "country_code"
        if tld in NEW_GENERIC: return "new_generic"
        return "rare"
    logger.info(f"TLD groups: top-{top_n} + 5 categories")
    return s.apply(label)

def assign_domain_length_groups(raw_df):
    col = "DomainLength"
    s = raw_df[col] if col in raw_df.columns else pd.Series(10, index=raw_df.index)
    q33,q67 = float(s.quantile(0.33)), float(s.quantile(0.67))
    def label(v):
        if v<=q33: return f"Short (≤{q33:.0f})"
        if v<=q67: return f"Medium ({q33:.0f}-{q67:.0f})"
        return f"Long (>{q67:.0f})"
    return s.apply(label)

def assign_ext_resource_groups(raw_df):
    col = "NoOfExternalRef"
    s = raw_df[col] if col in raw_df.columns else pd.Series(0, index=raw_df.index)
    q33,q67 = float(s.quantile(0.33)), float(s.quantile(0.67))
    def label(v):
        if v<=q33: return f"Low (≤{q33:.0f})"
        if v<=q67: return f"Medium ({q33:.0f}-{q67:.0f})"
        return f"High (>{q67:.0f})"
    return s.apply(label)

def run_group_evaluation(raw_df, y_true, y_pred, y_proba):
    assert len(raw_df)==len(y_true)==len(y_pred)==len(y_proba)
    dimensions = {
        "url_length":    assign_url_length_groups,
        "https":         assign_https_groups,
        "tld":           assign_tld_groups,
        "domain_length": assign_domain_length_groups,
        "ext_resources": assign_ext_resource_groups,
    }
    results = {}; all_metrics=[]; all_disparity=[]
    for dim_name, fn in dimensions.items():
        groups = fn(raw_df).reset_index(drop=True)
        gdf    = pd.DataFrame({"group_col": groups})
        mdf    = compute_all_group_metrics(gdf,y_true,y_pred,y_proba,"group_col",dim_name)
        ddf    = compute_disparity(mdf, dim_name)
        mdf["dimension"]=dim_name; all_metrics.append(mdf); all_disparity.append(ddf)
        results[f"{dim_name}_groups"]=groups
        results[f"{dim_name}_metrics"]=mdf
        results[f"{dim_name}_disparity"]=ddf
    results["all_metrics_df"]   = pd.concat(all_metrics,   ignore_index=True)
    results["all_disparity_df"] = pd.concat(all_disparity, ignore_index=True)
    return results
