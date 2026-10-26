"""
agents/medication_agent.py - review the medicines the guideline names for a disease against THIS patient.

Never invents a drug or a dose: drugs come from disease_drugs (ETL), doses only from dose_rules, warnings only from
contraindication_rules.  Anything without a calculator rule points the reader to the guideline snippet.
"""
from __future__ import annotations

import logging

from rules import contraindications, dosing
from schemas import MedicationReview, PatientCase

log = logging.getLogger(__name__)
_RANK = {"contraindicated": 0, "avoid": 1, "caution": 2, "ok": 3}


class MedicationAgent:
    def __init__(self, con, settings):
        self.con, self.s = con, settings

    def review(self, case: PatientCase, disease_name: str, max_drugs: int = 15) -> list[MedicationReview]:
        rows = self.con.execute(
            "SELECT drug, snippet FROM disease_drugs WHERE disease_name=? ORDER BY drug", (disease_name,)).fetchall()
        out: list[MedicationReview] = []
        for r in rows:
            out.append(self.review_drug(case, r["drug"], r["snippet"]))
        out.sort(key=lambda m: (_RANK[m.status], m.drug))
        return out[:max_drugs]

    def review_drug(self, case: PatientCase, drug: str, snippet: str = "") -> MedicationReview:
        hits = contraindications.check_drug(self.con, case, drug, self.s.require_reviewed_rules)
        status = hits[0].severity if hits else "ok"
        messages = [f"{h.message} [{h.rule_id}{'' if h.reviewed else ', unreviewed'}]" for h in hits]
        if drug.lower() in case.current_meds:
            messages.append("Patient is already taking this medicine.")
            if status == "ok":
                status = "caution"
        doses, needs = dosing.calculate(self.con, case, drug, require_reviewed=self.s.require_reviewed_rules)
        return MedicationReview(drug, status, messages, doses[0] if doses else None, snippet,
                                needs, hits) if len(doses) <= 1 else self._multi(drug, status, messages, doses, snippet, needs, hits)

    @staticmethod
    def _multi(drug, status, messages, doses, snippet, needs, hits) -> MedicationReview:
        # several rules for one drug (e.g. paracetamol for different indications): keep the first as `dose`,
        # and make the others visible in messages so no guideline difference is hidden.
        for d in doses[1:]:
            rng = f"{d.dose_min:g}" if d.dose_min == d.dose_max else f"{d.dose_min:g}-{d.dose_max:g}"
            messages.append(f"Other guideline dose for {d.indication}: {rng} {d.unit} {d.route} [{d.source_ref}]")
        return MedicationReview(drug, status, messages, doses[0], snippet, needs, hits)
