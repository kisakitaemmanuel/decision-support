"""agents/search_agent.py - turns a PatientCase (or a typed query) into hybrid-retrieval calls."""
from __future__ import annotations

import logging
import re

from retrieval.hybrid import HybridRetriever
from schemas import PatientCase, RetrievalResult

log = logging.getLogger(__name__)


def symptom_phrase(key: str) -> str:
    return key.replace("_", " ")


class SearchAgent:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def build_queries(self, case: PatientCase) -> list[str]:
        """Query 1 = the clinician's own words; query 2 = normalised symptoms (+ age group / pregnancy)."""
        qs: list[str] = []
        text = re.sub(r"\s+", " ", case.raw_text).strip()
        if text:
            qs.append(text[:300])
        parts = [symptom_phrase(s) for s in case.symptoms]
        # measured values become words the guideline uses, so vitals steer retrieval too
        if (case.sbp or 0) >= 140 or (case.dbp or 0) >= 90:
            parts.append("high blood pressure hypertension")
        if case.spo2 is not None and case.spo2 < 90:
            parts.append("hypoxaemia oxygen")
        if parts:
            ctx = []
            if case.age_months is not None:
                ctx.append("infant" if case.age_months < 12 else "child" if case.age_months < 144 else "adult")
            if case.pregnant:
                ctx.append("pregnancy")
            qs.append(" ".join(ctx + parts))
        return qs

    def search_case(self, case: PatientCase, k: int | None = None) -> RetrievalResult:
        queries = self.build_queries(case)
        log.debug("search queries: %s", queries)
        return self.retriever.retrieve_multi(queries, k_diseases=k)

    # thin pass-throughs used by the API / UIs
    def search(self, query: str, mode: str = "hybrid", k: int = 10) -> RetrievalResult:
        return self.retriever.retrieve(query, mode=mode, k_diseases=k)

    def suggest(self, prefix: str, limit: int = 8) -> list[dict]:
        return self.retriever.fts.suggest(prefix, limit)
