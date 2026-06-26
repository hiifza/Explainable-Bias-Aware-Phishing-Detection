"""
/api/reports/*  — Serves existing HTML reports from outputs/reports/.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

router = APIRouter(tags=["Reports"])

REPORT_MAP = {
    "bias":        "bias_analysis_report.html",
    "blindspot":   "blindspot_analysis_report.html",
    "shap":        "shap_analysis_report.html",
    "lime":        "lime_analysis_report.html",
    "evaluation":  "model_evaluation_report.html",
    "training":    "training_summary.html",
    "eda":         "m2_1_eda_report.html",
    "features":    "m1_2_feature_finalization_report.html",
    "audit":       "m1_1_data_audit_report.html",
}


def get_reports_dir() -> Path:
    here = Path(__file__).parent.resolve()
    for candidate in [here.parent.parent, here.parent.parent.parent]:
        p = candidate / "outputs" / "reports"
        if p.exists():
            return p
    return Path("outputs") / "reports"


@router.get("")
async def list_reports():
    d = get_reports_dir()
    available = {}
    for key, fname in REPORT_MAP.items():
        available[key] = {
            "name": fname,
            "available": (d / fname).exists(),
        }
    return {"reports": available}


@router.get("/{report_key}", response_class=HTMLResponse)
async def get_report(report_key: str):
    fname = REPORT_MAP.get(report_key)
    if not fname:
        raise HTTPException(404, f"Unknown report key: {report_key}")
    path = get_reports_dir() / fname
    if not path.exists():
        raise HTTPException(404, f"Report not yet generated: {fname}. Run M1–M10 notebooks first.")
    return HTMLResponse(content=path.read_text(encoding="utf-8"))
