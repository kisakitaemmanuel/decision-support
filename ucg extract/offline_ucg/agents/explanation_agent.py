"""
agents/explanation_agent.py - build the clinician-facing text.

The safety-critical parts (alerts, referral, medicine checks, doses, sources) are ALWAYS template-generated from
structured data.  If an LLM exists it only writes a short plain-language summary, and that summary is rejected
(replaced by nothing) unless every number and drug name in it also appears in the structured context.
"""
from __future__ import annotations

import logging
import re

from etl.drugs import find_drugs
from schemas import (ClinicalAssessment, Explanation, MedicationReview, PatientCase, RuleHit, UndeterminedRule)

log = logging.getLogger(__name__)
_NUM = re.compile(r"\d+(?:\.\d+)?")
DISCLAIMER = ("Decision support from the Uganda Clinical Guidelines 2023 text. Extracted automatically and some rules are "
              "unreviewed: a clinician must confirm before acting.")


def age_text(c: PatientCase) -> str:
    if c.age_months is None:
        return "age unknown"
    m = c.age_months
    return f"{m * 30.4375:.0f} days" if m < 1 else f"{m:g} months" if m < 24 else f"{m / 12:g} years"


class ExplanationAgent:
    def __init__(self, llm=None):
        self.llm = llm

    # ------------------------------------------------------------------ deterministic report
    def template(self, case: PatientCase, red: list[RuleHit], und: list[UndeterminedRule], a: ClinicalAssessment,
                 meds: list[MedicationReview], questions: list[str]) -> tuple[str, list[str]]:
        L: list[str] = []
        cites: list[str] = []
        facts = [age_text(case), {"F": "female", "M": "male"}.get(case.sex or "", "sex unknown")]
        if case.pregnant:
            facts.append("pregnant")
        if case.weight_kg:
            facts.append(f"{case.weight_kg:g} kg")
        if case.symptoms:
            facts.append("symptoms: " + ", ".join(s.replace("_", " ") for s in case.symptoms))
        if case.allergies:
            facts.append("allergies: " + ", ".join(case.allergies))
        L.append("PATIENT: " + "; ".join(facts))

        if red:
            L.append("\nALERTS")
            for h in red:
                tag = "" if h.reviewed else " (rule unreviewed)"
                L.append(f"- [{h.severity.upper()}] {h.message} {h.action}  [{h.rule_id}{tag}]")
        if a.referral:
            L.append("\nREFERRAL: " + a.referral.decision.replace("_", " ").upper())
            L += [f"- {r}" for r in a.referral.reasons] or ["- No referral trigger found by the rules."]

        if a.differential:
            L.append("\nLIKELY CONDITIONS (ranked from guideline text)")
            for i, d in enumerate(a.differential, 1):
                sid = f"S{i}"
                cites.append(f"[{sid}] UCG 2023 section {d.section_number} {d.disease_name}, PDF page {d.page}")
                why = ("matches " + ", ".join(x.replace('_', ' ') for x in d.matched_features)) if d.matched_features else "text match"
                gap = f"; not mentioned: {', '.join(x.replace('_', ' ') for x in d.unmatched_features)}" if d.unmatched_features else ""
                L.append(f"{i}. {d.disease_name}{f' (ICD-10 {d.icd10_code})' if d.icd10_code else ''} - {why}{gap} [{sid}]")
                if d.demographic_note:
                    L.append(f"    note: down-ranked, {d.demographic_note}")
        else:
            L.append("\nNo matching condition found in the guideline for this description.")

        if meds:
            L.append(f"\nMEDICINES NAMED FOR {a.differential[0].disease_name.upper()} (checked against this patient)")
            for m in meds:
                line = f"- {m.drug}: {m.status.upper()}"
                if m.dose:
                    d = m.dose
                    rng = f"{d.dose_min:g}" if d.dose_min == d.dose_max else f"{d.dose_min:g}-{d.dose_max:g}"
                    line += f" | calculated {rng} {d.unit} {d.route} for {d.indication}" + (f", every {d.interval_hours:g} h" if d.interval_hours else "")
                    line += f" ({d.rationale}) [{'reviewed' if d.reviewed else 'unreviewed'}; {d.source_ref}]"
                elif m.needs:
                    line += f" | dose needs: {', '.join(m.needs)}"
                else:
                    line += " | dose: read the guideline text"
                L.append(line)
                L += [f"    ! {x}" for x in m.messages]

        qs = list(dict.fromkeys(case.missing_info + questions))
        if und:
            miss = sorted({m for u in und for m in u.missing})
            qs.append("Rules could not be decided because these are unknown: " + ", ".join(miss))
        if qs:
            L.append("\nINFORMATION STILL NEEDED")
            L += [f"- {q}" for q in qs]
        if cites:
            L.append("\nSOURCES")
            L += cites
        L.append("\n" + DISCLAIMER)
        return "\n".join(L), cites

    # ------------------------------------------------------------------ optional LLM summary
    def summarise(self, context: str) -> tuple[str, str]:
        """Returns (summary, fallback_reason). Empty summary + reason when rejected or unavailable."""
        if self.llm is None or not getattr(self.llm, "available", False):
            return "", "no LLM"
        prompt = ("Write a 3-4 sentence plain-language summary for a nurse using ONLY the report below. Do not add any "
                  "number, dose, drug or fact that is not in the report.\n\nREPORT:\n" + context)
        try:
            out = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=250, temperature=0.0).strip()
        except Exception as e:                          # noqa: BLE001
            return "", f"llm error: {e}"
        if len(out) < 20 or out.lstrip().startswith(("{", "[", "<tool_call>")):
            return "", "malformed or empty summary"
        bad = verify_grounded(out, context)
        if bad:
            log.warning("LLM summary rejected, ungrounded: %s", bad)
            return "", "ungrounded: " + ", ".join(bad[:5])
        return out, ""

    def explain(self, case, red, und, a, meds, questions) -> Explanation:
        body, cites = self.template(case, red, und, a, meds, questions)
        summary, why = self.summarise(body)
        if summary:
            return Explanation("SUMMARY (AI-written, verified against the report)\n" + summary + "\n\n" + body, "llm", True, "", cites)
        return Explanation(body, "template", True, why, cites)


def verify_grounded(output: str, context: str) -> list[str]:
    """Numbers and drug names in `output` that do not occur in `context` (empty list == grounded)."""
    ctx_nums = set(_NUM.findall(context))
    bad = [n for n in _NUM.findall(output) if n not in ctx_nums]
    ctx_low = context.lower()
    bad += [d for d in find_drugs(output) if d not in ctx_low]
    return bad
