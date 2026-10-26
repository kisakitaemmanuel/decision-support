"""
rules/contraindications.py - drug vs patient checks (table contraindication_rules).

Seeds are standard prescribing cautions, all *unreviewed* (see rules/signoff.json).  Where UCG itself uses a drug
despite the caution (e.g. aspirin in rheumatic fever, doxycycline in anthrax) the severity is 'caution'/'avoid',
never silently 'ok', and the message tells the reader to read the guideline text.
"""
from __future__ import annotations

import json
import logging
import re

from rules.engine import evaluate, load_condition
from schemas import PatientCase, RuleHit

log = logging.getLogger(__name__)

_PEN = r"(amoxicillin|ampicillin|penicillin|cloxacillin|flucloxacillin|piperacillin|benzathine|amoxycillin|co-amoxiclav)"
_NSAID = r"(ibuprofen|diclofenac|acetylsalicylic|aspirin|indomethacin)"
_ACEI = r"(enalapril|lisinopril|captopril|ramipril|perindopril)"

# (rule_id, drug_regex, condition, severity, message, source_ref)
CI_SEEDS = [
    ("CI-PEN-ALLERGY", _PEN, {"allergy_any": ["penicillin", "amoxicillin", "ampicillin", "cloxacillin", "beta-lactam", "beta lactam"]},
     "contraindicated", "Patient reports penicillin allergy.", "general prescribing rule (unverified)"),
    ("CI-SULFA-ALLERGY", r"(cotrimoxazole|co-trimoxazole|sulfamethoxazole|sulfadoxine|sulphadoxine)",
     {"allergy_any": ["sulfa", "sulpha", "cotrimoxazole", "co-trimoxazole", "sulfonamide"]},
     "contraindicated", "Patient reports sulphonamide allergy.", "general prescribing rule (unverified)"),
    ("CI-NSAID-ALLERGY", _NSAID, {"allergy_any": ["aspirin", "nsaid", "ibuprofen", "diclofenac", "acetylsalicylic"]},
     "contraindicated", "Patient reports aspirin/NSAID allergy.", "general prescribing rule (unverified)"),
    ("CI-TETRA-PREG", r"(doxycycline|tetracycline)", {"pregnant": True},
     "avoid", "Tetracyclines are generally avoided in pregnancy; use only if the guideline text directs it for a life-threatening indication.",
     "general prescribing rule (unverified)"),
    ("CI-TETRA-CHILD", r"(doxycycline|tetracycline)", {"age_years_lt": 8},
     "caution", "Tetracyclines affect teeth/bones in young children; check the guideline text for this indication.",
     "general prescribing rule (unverified)"),
    ("CI-ACEI-PREG", _ACEI, {"pregnant": True},
     "contraindicated", "ACE inhibitors are contraindicated in pregnancy.", "general prescribing rule (unverified)"),
    ("CI-NSAID-PREG", r"(ibuprofen|diclofenac)", {"pregnant": True},
     "avoid", "NSAIDs are avoided in pregnancy, especially the third trimester.", "general prescribing rule (unverified)"),
    ("CI-ASPIRIN-CHILD", r"(acetylsalicylic|aspirin)", {"age_years_lt": 16},
     "caution", "Aspirin in children: Reye syndrome risk with viral illness; UCG uses it only for specific indications (e.g. rheumatic fever).",
     "UCG 4.1.9 Rheumatic Fever (dose text); general rule (unverified)"),
    ("CI-METFORMIN-KIDNEY", r"metformin", {"comorbidity_any": ["kidney", "renal"]},
     "avoid", "Metformin is avoided in significant kidney impairment.", "general prescribing rule (unverified)"),
    ("CI-NSAID-ASTHMA", _NSAID, {"comorbidity_any": ["asthma"]},
     "caution", "NSAIDs can trigger bronchospasm in some people with asthma.", "general prescribing rule (unverified)"),
]


def seed(con, signoff: dict) -> int:
    rows = [(rid, rx, json.dumps(cond), sev, msg, src, 1 if rid in signoff else 0)
            for rid, rx, cond, sev, msg, src in CI_SEEDS]
    con.executemany(
        """INSERT OR REPLACE INTO contraindication_rules
           (rule_id,drug_regex,condition_json,severity,message,source_ref,reviewed) VALUES (?,?,?,?,?,?,?)""", rows)
    return len(rows)


def check_drug(con, case: PatientCase, drug: str, require_reviewed: bool = False) -> list[RuleHit]:
    drug_l = drug.lower()
    q = "SELECT * FROM contraindication_rules WHERE enabled=1" + (" AND reviewed=1" if require_reviewed else "")
    hits: list[RuleHit] = []
    for r in con.execute(q):
        if not re.search(r["drug_regex"], drug_l):
            continue
        res = evaluate(load_condition(r["condition_json"]), case)
        if res.matched:
            hits.append(RuleHit(r["rule_id"], "contraindication", r["severity"], r["message"], "", False,
                                r["source_ref"], bool(r["reviewed"]), res.reasons, drug=drug))
    rank = {"contraindicated": 0, "avoid": 1, "caution": 2}
    hits.sort(key=lambda h: rank[h.severity])
    return hits
