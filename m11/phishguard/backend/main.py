"""
PhishGuard Backend — FastAPI
============================
Reads real M1-M10 outputs and serves them to the React frontend.
Uses pathlib throughout for full portability.
"""

from pathlib import Path
from contextlib import asynccontextmanager
import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from routers import analyze, intelligence, reports

# ── Paths (portable) ──────────────────────────────────────────────────────────
THIS_DIR  = Path(__file__).parent.resolve()
REPO_ROOT = THIS_DIR.parents[2]          # phishguard/ → project root
STATIC    = THIS_DIR / "static"      # built frontend lands here
STATIC.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[PhishGuard] Repository root: {REPO_ROOT}")
    yield


app = FastAPI(
    title="PhishGuard Intelligence API",
    version="1.0.0",
    description="Explainable, Bias-Aware Phishing Detection Intelligence Platform",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routers ───────────────────────────────────────────────────────────────
app.include_router(analyze.router,       prefix="/api")
app.include_router(intelligence.router,  prefix="/api/intelligence")
app.include_router(reports.router,       prefix="/api/reports")

# ── Serve React frontend (production) ─────────────────────────────────────────
if STATIC.exists() and (STATIC / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
