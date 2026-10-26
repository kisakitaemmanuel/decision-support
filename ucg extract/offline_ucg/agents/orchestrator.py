"""
agents/orchestrator.py - the pipeline.  Reads top to bottom as the whole system:

   1 parse patient     2 red-flag rules (always, before any model)     3 hybrid retrieval
   4 clinical agent (rank + optional LLM tool loop)     5 medication review     6 explanation     7 audit

Each step is timed into `trace`, so "why was it slow / why did it say that" is answerable from the result alone.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import db as dbmod
from agents.clinical_agent import ClinicalAgent
from agents.explanation_agent import ExplanationAgent
from agents.llm import LLM, get_llm
from agents.medication_agent import MedicationAgent
from agents.patient_agent import PatientAgent
from agents.search_agent import SearchAgent
from audit.logger import AuditLog
from config import Settings, get_settings
from retrieval.hybrid import HybridRetriever
from rules import red_flags
from rules.engine import fingerprint

log = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    con: object
    retriever: HybridRetriever
    llm: LLM
    audit: AuditLog

    @classmethod
    def build(cls, settings: Settings | None = None, llm: LLM | None = None) -> "Services":
        s = settings or get_settings()
        con = dbmod.connect(s.db_path, readonly=True)
        return cls(s, con, HybridRetriever(con, s), llm if llm is not None else get_llm(s),
                   AuditLog(s.audit_db_path, s.audit_store_raw_text))

    def health(self) -> dict:
        n = lambda t: self.con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]   # noqa: E731
        return {"database": str(self.settings.db_path), "diseases": n("diseases"), "chunks": n("chunks"),
                "rules": {t: n(t) for t in ("red_flag_rules", "contraindication_rules", "dose_rules")},
                "semantic": (self.retriever.semantic.backend + ":" + self.retriever.semantic.embedder.name)
                if self.retriever.semantic else f"disabled ({self.retriever.semantic_error})",
                "llm": self.llm.name if self.llm.available else "none (deterministic mode)",
                "facility_level": self.settings.facility_level, "pipeline_version": self.settings.pipeline_version}


class Pipeline:
    def __init__(self, services: Services):
        self.sv = services
        s, llm = services.settings, services.llm
        self.patient = PatientAgent(llm)
        self.search = SearchAgent(services.retriever)
        self.clinical = ClinicalAgent(services.con, s, services.retriever, llm)
        self.meds = MedicationAgent(services.con, s)
        self.explainer = ExplanationAgent(llm)

    def run(self, text: str, overrides: dict | None = None) -> dict:
        sv, s = self.sv, self.sv.settings
        t_all = time.perf_counter()
        trace: list[dict] = []

        def step(name: str, t0: float, **detail):
            trace.append({"step": name, "ms": round((time.perf_counter() - t0) * 1000, 1), **detail})

        t = time.perf_counter()
        case = self.patient.parse(text, overrides, s.facility_level)
        step("parse_patient", t, symptoms=case.symptoms)

        t = time.perf_counter()
        red, und = red_flags.evaluate_all(sv.con, case, s.require_reviewed_rules)
        step("red_flags", t, fired=[h.rule_id for h in red], undetermined=len(und))

        t = time.perf_counter()
        retrieval = self.search.search_case(case)
        step("retrieval", t, candidates=[d.disease_name for d in retrieval.diseases], degraded=retrieval.degraded)

        t = time.perf_counter()
        assessment = self.clinical.assess(case, retrieval)
        step("clinical", t, mode=assessment.mode, top=[d.disease_name for d in assessment.differential[:3]])

        meds = []
        if assessment.differential:
            t = time.perf_counter()
            meds = self.meds.review(case, assessment.differential[0].disease_name)
            step("medication", t, n=len(meds), blocked=[m.drug for m in meds if m.status in ("contraindicated", "avoid")])

        t = time.perf_counter()
        expl = self.explainer.explain(case, red, und, assessment, meds, assessment.questions)
        step("explanation", t, mode=expl.mode, fallback=expl.fallback_reason)

        total = (time.perf_counter() - t_all) * 1000
        result = {
            "case": case.to_dict(), "red_flags": [h.to_dict() for h in red], "undetermined_rules": [u.to_dict() for u in und],
            "retrieval": {"query": retrieval.query, "degraded": retrieval.degraded,
                          "candidates": [{"disease": d.disease_name, "score": round(d.score, 5), "ranks": d.ranks} for d in retrieval.diseases]},
            "assessment": assessment.to_dict(), "medications": [m.to_dict() for m in meds],
            "explanation": expl.to_dict(), "trace": trace, "latency_ms": round(total, 1),
            "versions": {"pipeline": s.pipeline_version, "rules": fingerprint(sv.con), "llm": sv.llm.name,
                         "embedder": sv.retriever.semantic.embedder.name if sv.retriever.semantic else None},
        }
        result["decision_id"] = sv.audit.record(text, result["case"], result["red_flags"], result["retrieval"],
                                                assessment.tool_trace, result, result["versions"], total)
        log.info("decision %s: %d flags, top=%s, %.0f ms", result["decision_id"][:8], len(red),
                 assessment.differential[0].disease_name if assessment.differential else None, total)
        return result

    # convenience passthroughs for the API / UIs
    def disease(self, name: str) -> dict | None:
        r = self.sv.con.execute("SELECT * FROM diseases WHERE disease_name=? COLLATE NOCASE", (name,)).fetchone()
        return dict(r) if r else None
