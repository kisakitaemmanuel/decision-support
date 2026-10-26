"""
app.py - FastAPI service (the single backend for the PySide6 desktop app and the Flutter Android app).

    uvicorn app:app --host 0.0.0.0 --port 8000        # 0.0.0.0 so tablets on the clinic Wi-Fi can reach it
    UCG_LLM_ENABLED=0 uvicorn app:app                 # force deterministic mode

No internet is needed at runtime.  Interactive docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.orchestrator import Pipeline, Services
from config import get_settings, setup_logging

log = logging.getLogger("api")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    sv = Services.build()
    state["sv"], state["pipe"] = sv, Pipeline(sv)
    log.info("ready: %s", sv.health())
    yield


app = FastAPI(title="Offline UCG decision support", version=get_settings().pipeline_version, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("UCG_CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


class Overrides(BaseModel):
    age_years: float | None = None
    age_months: float | None = None
    sex: str | None = Field(None, pattern="^[FM]$")
    pregnant: bool | None = None
    weight_kg: float | None = Field(None, gt=0, lt=250)
    temp_c: float | None = None
    spo2: float | None = Field(None, ge=0, le=100)
    hr: float | None = None
    rr: float | None = None
    sbp: float | None = None
    dbp: float | None = None
    facility_level: str | None = None
    symptoms: list[str] = []
    allergies: list[str] = []
    comorbidities: list[str] = []
    current_meds: list[str] = []


class TriageRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=4000)
    overrides: Overrides | None = None


class FeedbackRequest(BaseModel):
    verdict: str = Field(..., pattern="^(agree|disagree|unsure)$")
    comment: str = ""
    reviewer: str = ""


@app.get("/api/health")
def health():
    return state["sv"].health()


@app.post("/api/triage")
def triage(req: TriageRequest):
    ov = req.overrides.model_dump(exclude_none=True) if req.overrides else None
    return state["pipe"].run(req.text, ov)


@app.get("/api/search")
def search(q: str = Query(..., min_length=1, max_length=300), mode: str = Query("hybrid", pattern="^(hybrid|fts|semantic)$"),
           k: int = Query(10, ge=1, le=30)):
    res = state["pipe"].search.search(q, mode, k)
    return {"query": q, "mode": mode, "degraded": res.degraded, "timings_ms": res.timings_ms,
            "results": [{"disease": d.disease_name, "icd10": d.icd10_code, "score": round(d.score, 5), "ranks": d.ranks,
                         "snippet": d.chunks[0].text[:260] if d.chunks else ""} for d in res.diseases]}


@app.get("/api/suggest")
def suggest(q: str = Query("", max_length=100), limit: int = Query(8, ge=1, le=20)):
    return {"query": q, "suggestions": state["pipe"].search.suggest(q, limit)}


@app.get("/api/disease/{name:path}")
def disease(name: str):
    d = state["pipe"].disease(name)
    if d is None:
        raise HTTPException(404, f"No disease named {name!r}")
    return d


@app.get("/api/audit/recent")
def audit_recent(n: int = Query(20, ge=1, le=200)):
    return state["sv"].audit.recent(n)


@app.get("/api/audit/{decision_id}")
def audit_get(decision_id: str):
    d = state["sv"].audit.get(decision_id)
    if d is None:
        raise HTTPException(404, "unknown decision id")
    return d


@app.post("/api/audit/{decision_id}/feedback")
def audit_feedback(decision_id: str, fb: FeedbackRequest):
    try:
        return {"feedback_id": state["sv"].audit.add_feedback(decision_id, fb.verdict, fb.comment, fb.reviewer)}
    except KeyError:
        raise HTTPException(404, "unknown decision id")
