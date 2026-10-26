"""
agents/clinical_agent.py - ranks candidate conditions and (when an LLM exists) lets it call tools to dig deeper.

Ranking is DETERMINISTIC and always computed:
    score = 0.6 * normalised retrieval score + 0.4 * share of the patient's symptoms found in the guideline's
            clinical-features text (+ llm_rank_bonus if the LLM nominated it)
The LLM can nominate conditions and suggest questions, but only conditions that exist in the database count, and it
cannot move a condition more than the configured bonus.  Every tool call is recorded in `tool_trace`.
"""
from __future__ import annotations

import json
import logging

from agents.base import Tool, ToolCallLoop, ToolRegistry, extract_json
from agents.patient_agent import SYMPTOM_RE
from rules import contraindications, dosing, red_flags, referral
from schemas import ClinicalAssessment, DiffItem, PatientCase, RetrievalResult

log = logging.getLogger(__name__)

SYSTEM = ("You are a clinical guideline assistant for the Uganda Clinical Guidelines 2023. Use the tools to look up "
          "facts; never state a dose, drug or threshold that did not come from a tool result. Task: decide which of "
          "the candidate conditions deserve closer attention for this patient and what question would best separate "
          "them. Final answer must be JSON only: "
          '{"nominated": ["<exact disease name>", ...], "questions": ["..."]}')
SECTIONS = ("clinical_features", "differential_diagnosis", "investigations", "management", "description", "causes",
            "prevention", "complications", "risk_factors")


class ClinicalAgent:
    def __init__(self, con, settings, retriever, llm):
        self.con, self.s, self.retriever, self.llm = con, settings, retriever, llm

    # ------------------------------------------------------------------ tools (bound to one case)
    def build_tools(self, case: PatientCase) -> ToolRegistry:
        reg = ToolRegistry()

        def search_guidelines(query: str, k: int = 5):
            res = self.retriever.retrieve(query, k_diseases=min(max(k, 1), 8))
            return [{"disease": d.disease_name, "icd10": d.icd10_code,
                     "snippet": (d.chunks[0].text[:220] if d.chunks else "")} for d in res.diseases]

        def get_disease_section(disease_name: str, section: str):
            if section not in SECTIONS:
                raise ValueError(f"section must be one of {SECTIONS}")
            r = self.con.execute(f"SELECT {section} AS body, start_pdf_page FROM diseases WHERE disease_name=? COLLATE NOCASE",
                                 (disease_name,)).fetchone()
            if r is None:
                raise ValueError(f"unknown disease {disease_name!r}")
            return {"disease": disease_name, "section": section, "page": r["start_pdf_page"],
                    "text": (r["body"] or "(not documented)")[:1400]}

        def check_red_flags():
            hits, und = red_flags.evaluate_all(self.con, case, self.s.require_reviewed_rules)
            return {"fired": [{"id": h.rule_id, "severity": h.severity, "message": h.message} for h in hits],
                    "undetermined": [{"id": u.rule_id, "missing": u.missing} for u in und][:6]}

        def check_contraindications(drug: str):
            return [{"id": h.rule_id, "severity": h.severity, "message": h.message}
                    for h in contraindications.check_drug(self.con, case, drug, self.s.require_reviewed_rules)]

        def calculate_dose(drug: str, indication: str = ""):
            doses, needs = dosing.calculate(self.con, case, drug, indication or None, self.s.require_reviewed_rules)
            return {"doses": [d.to_dict() for d in doses], "needs": needs}

        def referral_advice(disease_name: str):
            r = self.con.execute("SELECT levels_of_care FROM diseases WHERE disease_name=? COLLATE NOCASE",
                                 (disease_name,)).fetchone()
            hits, _ = red_flags.evaluate_all(self.con, case, self.s.require_reviewed_rules)
            return referral.advise(self.con, case.facility_level, hits, r["levels_of_care"] if r else None).to_dict()

        obj = lambda props, req: {"properties": props, "required": req}   # noqa: E731
        S, I = {"type": "string"}, {"type": "integer"}
        for t in (
            Tool("search_guidelines", "Hybrid search of the guideline; returns disease names with a snippet.",
                 obj({"query": S, "k": I}, ["query"]), search_guidelines),
            Tool("get_disease_section", "Read one section of a disease entry (clinical_features, investigations, "
                 "differential_diagnosis, management, ...).", obj({"disease_name": S, "section": S}, ["disease_name", "section"]),
                 get_disease_section),
            Tool("check_red_flags", "Evaluate danger-sign rules for this patient.", obj({}, []), check_red_flags),
            Tool("check_contraindications", "Check one drug against this patient's allergies/pregnancy/age/conditions.",
                 obj({"drug": S}, ["drug"]), check_contraindications),
            Tool("calculate_dose", "Weight-based dose from the guideline's own rules (only some drugs).",
                 obj({"drug": S, "indication": S}, ["drug"]), calculate_dose),
            Tool("referral_advice", "Manage-here vs refer decision for a disease at this facility.",
                 obj({"disease_name": S}, ["disease_name"]), referral_advice),
        ):
            reg.register(t)
        return reg

    # ------------------------------------------------------------------ deterministic scoring
    def _features(self, case: PatientCase, row) -> tuple[list[str], list[str]]:
        text = " ".join(filter(None, [row["clinical_features"], row["description"]])).lower()
        matched, unmatched = [], []
        for s in case.symptoms:
            rx = SYMPTOM_RE.get(s)
            (matched if rx and rx.search(text) else unmatched).append(s)
        return matched, unmatched

    @staticmethod
    def demographic_factor(case: PatientCase, row) -> tuple[float, str]:
        """Down-rank (never remove) conditions that cannot apply to this patient's sex / pregnancy / age band."""
        chapter, name = row["chapter_number"], row["disease_name"].lower()
        if chapter in (14, 16) and (case.sex == "M" or case.pregnant is False and chapter == 16):
            return 0.3, "obstetric/gynaecological condition, patient is male or not pregnant" if case.sex == "M" or chapter == 16 else ""
        if case.age_months is not None:
            child_text = chapter in (17, 18) or any(w in name for w in ("child", "infant", "neonat", "newborn", "paediatric"))
            if child_text and case.age_months >= 180:
                return 0.5, "childhood condition, patient is 15 years or older"
            if "adult" in name and case.age_months < 144:
                return 0.5, "adult condition, patient is under 12 years"
        return 1.0, ""

    def score(self, case: PatientCase, retrieval: RetrievalResult) -> list[DiffItem]:
        if not retrieval.diseases:
            return []
        top = max(d.score for d in retrieval.diseases) or 1.0
        items = []
        for d in retrieval.diseases:
            row = self.con.execute("SELECT * FROM diseases WHERE disease_name=?", (d.disease_name,)).fetchone()
            if row is None:
                continue
            matched, unmatched = self._features(case, row)
            fscore = len(matched) / len(case.symptoms) if case.symptoms else 0.0
            rscore = d.score / top
            factor, dnote = self.demographic_factor(case, row)
            items.append(DiffItem(d.disease_name, round((0.6 * rscore + 0.4 * fscore) * factor, 4), round(rscore, 4),
                                  round(fscore, 4), matched, unmatched, False,
                                  [c.text[:220].replace("\n", " ") for c in d.chunks[:2]],
                                  row["start_pdf_page"], row["section_number"], row["icd10_code"], dnote))
        items.sort(key=lambda i: -i.score)
        return items

    # ------------------------------------------------------------------ main entry
    def assess(self, case: PatientCase, retrieval: RetrievalResult) -> ClinicalAssessment:
        diff = self.score(case, retrieval)
        trace: list[dict] = []
        questions: list[str] = []
        notes: list[str] = list(retrieval.degraded)
        mode = "deterministic"

        if diff and self.llm is not None and getattr(self.llm, "available", False):
            mode = "llm+tools"
            loop = ToolCallLoop(self.llm, self.build_tools(case), max_steps=5)
            user = (f"Patient: {json.dumps({k: v for k, v in case.to_dict().items() if v not in (None, [], '') and k not in ('raw_text', 'extraction_notes')})}\n"
                    f"Original note: {case.raw_text[:600]}\nCandidates: {[d.disease_name for d in diff]}")
            res = loop.run(SYSTEM, user)
            trace = res.steps
            parsed = extract_json(res.final_text) if res.final_text else None
            if parsed is None:
                notes.append(f"LLM gave no usable JSON ({res.stopped}); ranking is deterministic only")
            else:
                valid = {r["disease_name"].lower(): r["disease_name"] for r in
                         self.con.execute("SELECT disease_name FROM diseases")}
                for name in parsed.get("nominated", []):
                    real = valid.get(str(name).lower())
                    if real is None:
                        notes.append(f"LLM named unknown condition {name!r}: ignored")
                        continue
                    for it in diff:
                        if it.disease_name == real:
                            it.llm_nominated = True
                            it.score = round(it.score + self.s.llm_rank_bonus, 4)
                questions = [str(q)[:160] for q in parsed.get("questions", [])][:5]
                diff.sort(key=lambda i: -i.score)

        refer = None
        if diff:
            row = self.con.execute("SELECT levels_of_care FROM diseases WHERE disease_name=?", (diff[0].disease_name,)).fetchone()
            hits, _ = red_flags.evaluate_all(self.con, case, self.s.require_reviewed_rules)
            refer = referral.advise(self.con, case.facility_level, hits, row["levels_of_care"] if row else None)
        return ClinicalAssessment(diff[: self.s.top_k_diseases], refer, mode, trace, questions, notes)
