#!/usr/bin/env python3
"""
PhishGuard M11 Setup & Launch Script
=====================================
Installs dependencies and starts both backend and frontend dev servers.
Portable: Windows / macOS / Linux.

Usage:
    python setup.py          # install + start both servers
    python setup.py --build  # build frontend for production
    python setup.py --check  # verify M1-M10 outputs only
"""

from __future__ import annotations
import sys, os, subprocess, argparse, shutil
from pathlib import Path

HERE     = Path(__file__).parent.resolve()
FRONTEND = HERE / "frontend"
BACKEND  = HERE / "backend"
REPO     = HERE.parent    # Explainable-Bias-Aware-Phishing-Detection/
OUTPUTS  = REPO / "outputs"

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):  print(f"{GREEN}✓{RESET}  {msg}")
def warn(msg): print(f"{YELLOW}⚠{RESET}  {msg}")
def err(msg):  print(f"{RED}✗{RESET}  {msg}")
def head(msg): print(f"\n{BOLD}{msg}{RESET}")


def check_outputs():
    head("Phase 1 — Verifying M1-M10 Outputs")
    required = [
        OUTPUTS / "reports" / "shap_feature_ranking.csv",
        OUTPUTS / "reports" / "top20_blind_spots.csv",
        OUTPUTS / "reports" / "blindspot_severity.csv",
        OUTPUTS / "reports" / "bias_metrics.csv",
        OUTPUTS / "reports" / "evaluation_metrics.csv",
        OUTPUTS / "reports" / "shap_lime_agreement.csv",
        OUTPUTS / "reports" / "failure_archetypes.csv",
        OUTPUTS / "reports" / "reliability_bin_stats.csv",
        OUTPUTS / "plots"   / "shap" / "global_importance.png",
        OUTPUTS / "plots"   / "bias" / "disparity" / "accuracy_heatmap.png",
        OUTPUTS / "plots"   / "blindspot" / "clusters" / "pca_cluster_map.png",
    ]
    found   = [p for p in required if p.exists()]
    missing = [p for p in required if not p.exists()]
    ok(f"Found {len(found)}/{len(required)} required M1-M10 outputs")
    if missing:
        for m in missing:
            warn(f"Missing: {m.relative_to(REPO)}")
        warn("Run M1-M10 notebooks to generate missing outputs.")
        warn("Backend will use known research results as fallback.")
    else:
        ok("All M1-M10 outputs verified — backend will serve real data")
    return len(missing) == 0


def install_backend():
    head("Phase 2 — Backend Dependencies")
    req = BACKEND / "requirements.txt"
    if not req.exists():
        err("requirements.txt not found"); return False
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        ok("Backend dependencies installed (FastAPI, Uvicorn, Pydantic)")
        return True
    else:
        err("Backend install failed:")
        print(result.stderr[-800:])
        return False


def install_frontend():
    head("Phase 3 — Frontend Dependencies")
    if not shutil.which("node"):
        warn("Node.js not found — skipping frontend install")
        warn("Install Node.js 18+ from https://nodejs.org to run the React frontend")
        return False
    pkg = FRONTEND / "package.json"
    if not pkg.exists():
        err("package.json not found"); return False
    result = subprocess.run(
        ["npm", "install"],
        cwd=FRONTEND, capture_output=True, text=True
    )
    if result.returncode == 0:
        ok("Frontend dependencies installed (React, Three.js, GSAP, Vite)")
        return True
    else:
        err("Frontend npm install failed:")
        print(result.stderr[-800:])
        return False


def build_frontend():
    head("Building Frontend (production)")
    if not shutil.which("node"):
        err("Node.js required for build"); return False
    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=FRONTEND, capture_output=True, text=True
    )
    if result.returncode == 0:
        ok("Frontend built → backend/static/")
        return True
    err("Build failed"); print(result.stderr[-800:]); return False


def start_servers():
    head("Phase 4 — Starting Servers")
    import threading, time

    def run_backend():
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "main:app",
             "--host", "0.0.0.0", "--port", "8000", "--reload"],
            cwd=BACKEND
        )

    def run_frontend():
        if not shutil.which("node"):
            return
        subprocess.run(["npm", "run", "dev"], cwd=FRONTEND)

    print()
    print(f"  {BOLD}Backend  →{RESET}  http://localhost:8000")
    print(f"  {BOLD}Frontend →{RESET}  http://localhost:3000")
    print(f"  {BOLD}API Docs →{RESET}  http://localhost:8000/docs")
    print()
    print("  Press Ctrl+C to stop both servers.")
    print()

    t_back  = threading.Thread(target=run_backend,  daemon=True)
    t_front = threading.Thread(target=run_frontend, daemon=True)
    t_back.start()
    time.sleep(1.5)
    t_front.start()

    try:
        t_back.join()
        t_front.join()
    except KeyboardInterrupt:
        print("\n  Servers stopped.")


def main():
    parser = argparse.ArgumentParser(description="PhishGuard M11 Setup")
    parser.add_argument("--build",  action="store_true", help="Build frontend for production")
    parser.add_argument("--check",  action="store_true", help="Check M1-M10 outputs only")
    args = parser.parse_args()

    print()
    print(f"{BOLD}PhishGuard Intelligence Platform — M11 Setup{RESET}")
    print("=" * 50)

    check_outputs()

    if args.check:
        return

    ok_back  = install_backend()
    ok_front = install_frontend()

    if args.build:
        if ok_front:
            build_frontend()
        return

    if ok_back:
        start_servers()
    else:
        err("Backend setup failed. Cannot start servers.")
        sys.exit(1)


if __name__ == "__main__":
    main()
