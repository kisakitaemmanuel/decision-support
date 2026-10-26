"""
ui/desktop/backend.py - the desktop UI talks to a Backend; swap in-process for HTTP without touching the UI.

    InProcessBackend   runs the pipeline inside the desktop app (single-machine clinic laptop, zero setup)
    HttpBackend        calls the FastAPI service (shared server in the clinic)   -> uses urllib only
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class Backend:
    def health(self) -> dict: raise NotImplementedError
    def triage(self, text: str, overrides: dict | None = None) -> dict: raise NotImplementedError
    def suggest(self, prefix: str) -> list[dict]: raise NotImplementedError
    def search(self, q: str, mode: str = "hybrid") -> list[dict]: raise NotImplementedError
    def disease(self, name: str) -> dict | None: raise NotImplementedError
    def feedback(self, decision_id: str, verdict: str, comment: str = "") -> None: raise NotImplementedError


class InProcessBackend(Backend):
    def __init__(self):
        from agents.orchestrator import Pipeline, Services
        from config import setup_logging
        setup_logging()
        self.sv = Services.build()
        self.pipe = Pipeline(self.sv)

    def health(self): return self.sv.health()
    def triage(self, text, overrides=None): return self.pipe.run(text, overrides)
    def suggest(self, prefix): return self.pipe.search.suggest(prefix)

    def search(self, q, mode="hybrid"):
        r = self.pipe.search.search(q, mode, 10)
        return [{"disease": d.disease_name, "icd10": d.icd10_code, "score": d.score,
                 "snippet": d.chunks[0].text[:260] if d.chunks else ""} for d in r.diseases]

    def disease(self, name): return self.pipe.disease(name)
    def feedback(self, decision_id, verdict, comment=""): self.sv.audit.add_feedback(decision_id, verdict, comment, "desktop")


class HttpBackend(Backend):
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def _req(self, path, body=None, timeout=180):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data, {"Content-Type": "application/json"} if data else {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def health(self): return self._req("/api/health")
    def triage(self, text, overrides=None): return self._req("/api/triage", {"text": text, "overrides": overrides})
    def suggest(self, prefix): return self._req("/api/suggest?q=" + urllib.parse.quote(prefix))["suggestions"]

    def search(self, q, mode="hybrid"):
        return [{"disease": r["disease"], "icd10": r["icd10"], "score": r["score"], "snippet": r["snippet"]}
                for r in self._req(f"/api/search?mode={mode}&q=" + urllib.parse.quote(q))["results"]]

    def disease(self, name): return self._req("/api/disease/" + urllib.parse.quote(name))
    def feedback(self, decision_id, verdict, comment=""):
        self._req(f"/api/audit/{decision_id}/feedback", {"verdict": verdict, "comment": comment, "reviewer": "desktop"})
