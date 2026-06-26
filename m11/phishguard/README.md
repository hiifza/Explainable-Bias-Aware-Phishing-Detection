# PhishGuard — Cybersecurity Intelligence Platform

**M11: Explainable, Bias-Aware Phishing Detection Intelligence Platform**

Built on top of the complete M1–M10 research pipeline.

---

## Quick Start

```bash
# 1. Navigate to the m11 directory
cd m11

# 2. Install dependencies and start both servers
python setup.py

# Access:
# Frontend  →  http://localhost:3000
# Backend   →  http://localhost:8000
# API Docs  →  http://localhost:8000/docs
```

### Manual start

```bash
# Backend (Terminal 1)
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (Terminal 2)
cd frontend
npm install
npm run dev
```

### Docker (production)

```bash
docker compose up
# → http://localhost:8080
```

---

## Architecture

```
phishguard/
├── frontend/                    # React + TypeScript + Vite
│   ├── src/
│   │   ├── components/
│   │   │   ├── 3d/              # Three.js threat network globe
│   │   │   ├── layout/          # Nav (magnetic buttons, theme toggle)
│   │   │   ├── scanner/         # URL analyzer, TrustMeter, ScanResult
│   │   │   └── intelligence/    # SHAP, Model, Blind Spot, Bias, Reliability
│   │   ├── pages/               # Home, Investigate, Research, Learn, About
│   │   ├── store/               # Zustand (theme, mode, scan state)
│   │   ├── lib/                 # Axios API client
│   │   └── styles/              # Global CSS design system (exact brand colors)
│   └── package.json
│
├── backend/                     # FastAPI
│   ├── main.py                  # App entry point, CORS, static serving
│   ├── routers/
│   │   ├── analyze.py           # POST /api/analyze — primary scan endpoint
│   │   ├── intelligence.py      # GET  /api/intelligence/* — M1-M10 data
│   │   └── reports.py           # GET  /api/reports/* — HTML report proxy
│   └── requirements.txt
│
├── setup.py                     # One-command setup & launch
├── docker-compose.yml           # Production deployment
└── README.md
```

---

## Design System

### Dark Intelligence (default)
| Token | Value |
|-------|-------|
| Background | `#210203` |
| Secondary | `#660707` |
| Accent | `#D3B99F` |
| Secondary Accent | `#82576A` |
| Text | `#DAD0DF` |

### Light Intelligence (toggle)
| Token | Value |
|-------|-------|
| Background | `#F2E5D7` |
| Secondary | `#FFBCB5` |
| Accent | `#C97D60` |
| Dark Accent | `#63372C` |
| Text | `#262322` |

Themes are stored in `localStorage`, applied before first paint, and transition with `500ms ease`.

---

## Platform Layers

| Layer | Target User | Content |
|-------|-------------|---------|
| **Cyber Trust** | Everyday users | URL Scanner, Trust Score (0–100), human explanations, attacker simulation, safety guidance |
| **Analyst** | Students / analysts | SHAP-LIME Conflict, Blind Spot Center, Failure Archetypes, Reliability Zones |
| **Research** | Researchers | Model lab, SHAP rankings, bias observatory, full notebook timeline |

---

## Key Research Results (real data, no placeholders)

| Metric | Value |
|--------|-------|
| Best Accuracy | 100.00% (Track A) / 99.9936% (Track B, deployment) |
| ROC-AUC | 1.00 |
| Models | 4 (LR, RF, XGBoost, LightGBM) × 2 tracks |
| Features | 56 |
| Critical Blind Spots | 3 (samples #17372, #11301, #30588) |
| Failure Archetypes | 3 (Alpha, Beta, Gamma) |
| SHAP-LIME Local Agreement | 0% |
| Fairness Status | PASS — all 5 dimensions |
| Red Zone Error Rate | 13.04% (23 samples, agreement 0.0–0.2) |
| Red Zone Confidence | 97.31% mean |
| URLSimilarityIndex | 18.68% contribution — critical leakage, excluded from Track B |
| Top SHAP Feature | LetterRatioInURL (10.51%) |

---

## M11 Pages

| Route | Page | Purpose |
|-------|------|---------|
| `/` | Home | URL scanner (primary), hero, model performance, SHAP, conflict, bias |
| `/investigate` | Investigate | Blind spots, failure archetypes, reliability analysis |
| `/research` | Research | Full research findings, notebook timeline, all intelligence sections |
| `/learn` | Learn | Phishing education, attack types, safe browsing tips |
| `/about` | About | Team profiles (Hifza Amir, Shihan Ahmad), dataset info |

---

## Portability

- All Python paths use `pathlib`
- No absolute paths anywhere
- Backend auto-detects repository root by walking up the directory tree
- Works unchanged on Windows, macOS, Linux, Docker, Render, Railway, AWS, Azure, GCP

---

## Researchers

**Hifza Amir** — B.Tech CSE (Data Science)
GitHub: [hiifza](https://github.com/hiifza) · LinkedIn: [hiifzza](https://www.linkedin.com/in/hiifzza/)

**Shihan Ahmad** — B.Tech CSE (Cybersecurity)
GitHub: [ShihanG9](https://github.com/ShihanG9) · LinkedIn: [shihanahmad](https://www.linkedin.com/in/shihanahmad/)
