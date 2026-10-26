"""
rules/red_flags.py - emergency / danger-sign rules (data in SQLite table red_flag_rules).

Seeds below were written from the UCG 2023 text where a source is cited; they are *unreviewed* until a clinician
adds the rule_id to rules/signoff.json.  Anything marked "(unverified)" is general clinical practice that has NOT
been matched to the guideline wording yet.  Output always carries the `reviewed` flag so the UI can show it.
"""
from __future__ import annotations

import logging

from rules.engine import evaluate, load_condition
from schemas import PatientCase, RuleHit, UndeterminedRule

log = logging.getLogger(__name__)

# (rule_id, name, condition, severity, refer, message, action, source_ref)
RED_FLAG_SEEDS = [
    ("RF-UNCONSCIOUS", "Unconscious or lethargic", {"symptoms_any": ["unconscious", "lethargy"]},
     "emergency", 1, "Reduced level of consciousness.",
     "Check airway, breathing, circulation; check blood glucose; refer urgently.",
     "UCG 1.1.6 Hypoglycaemia; 17.3.4 Check for Fever"),
    ("RF-CONVULSIONS", "Convulsions", {"symptoms_any": ["convulsions"]},
     "urgent", 0, "Convulsions reported.",
     "Protect airway, check glucose and temperature; see Febrile Convulsions / Meningitis / Epilepsy; refer if persistent.",
     "UCG 1.1.5 Febrile Convulsions"),
    ("RF-CHILD-NO-DRINK", "Child unable to drink / vomits everything",
     {"age_months_lt": 60, "symptoms_any": ["unable_to_drink", "vomiting_everything"]},
     "urgent", 1, "Danger sign in a child under 5 years.", "Assess for severe illness and dehydration; refer urgently.",
     "UCG 17.3.3 Child Has Diarrhoea"),
    ("RF-HYPOXIA", "Low oxygen saturation", {"spo2_lt": 90},
     "emergency", 1, "SpO2 below 90%.", "Start oxygen per guideline and refer.",
     "UCG 1.4 Hypoxaemia Management; 17.3.2 (refer if <90%)"),
    ("RF-MENINGISM", "Fever with neck stiffness", {"symptoms_all": ["fever", "neck_stiffness"]},
     "emergency", 1, "Possible meningitis.", "Urgent referral; do not delay antibiotics where guideline directs.",
     "UCG 2.1.5 Meningitis; 17.3.4 Check for Fever"),
    ("RF-SEVERE-PREECLAMPSIA", "Severe hypertension in pregnancy",
     {"pregnant": True, "any_of": [{"sbp_gte": 160}, {"dbp_gte": 110}]},
     "emergency", 1, "Blood pressure in the severe range in pregnancy.", "Manage as severe pre-eclampsia and refer.",
     "UCG 16.3.7 Pre-Eclampsia (160/110 mmHg)"),
    ("RF-PREG-CONVULSIONS", "Convulsions in pregnancy", {"pregnant": True, "symptoms_any": ["convulsions"]},
     "emergency", 1, "Convulsions in a pregnant woman: consider eclampsia.", "Manage per eclampsia protocol and refer.",
     "UCG 16.3.7 (unverified wording)"),
    ("RF-SHOCK", "Low blood pressure with fast pulse", {"sbp_lt": 90, "hr_gt": 100},
     "emergency", 1, "Possible shock.", "Start resuscitation per shock guideline; refer.",
     "UCG 1.1.2 Hypovolaemic Shock (thresholds unverified)"),
    ("RF-YOUNG-INFANT-FEVER", "Fever in infant under 2 months",
     {"age_months_lt": 2, "any_of": [{"symptoms_any": ["fever"]}, {"temp_c_gte": 38}]},
     "urgent", 1, "Fever in a young infant may be serious bacterial infection.", "Refer urgently.",
     "UCG 5.2.9.1 / 17.3.4 (unverified wording)"),
    ("RF-FAST-BREATHING", "Fast breathing for age", {"rr_gte_by_age": [[0, 2, 60], [2, 12, 50], [12, 60, 40]]},
     "warning", 0, "Breathing rate above the age threshold: consider pneumonia.", "Assess for chest indrawing and danger signs.",
     "UCG 5.2.9.1 (60/min <2 months); 17.3.2 (50/min 2-12 months, 40/min 1-5 years)"),
    ("RF-CHILD-SEVERE-RESP", "Chest indrawing or stridor in a child",
     {"age_months_lt": 60, "symptoms_any": ["chest_indrawing", "stridor"]},
     "urgent", 1, "Signs of severe respiratory illness in a child.", "Refer urgently; give first-dose treatment as guideline directs.",
     "UCG 17.3.2 Check for Cough or Difficult Breathing"),
    ("RF-SEVERE-BLEEDING", "Severe bleeding", {"symptoms_any": ["severe_bleeding"]},
     "emergency", 1, "Severe bleeding.", "Control bleeding, start IV access/fluids, refer.",
     "UCG 1.1.2 (unverified wording)"),
    ("RF-DIFF-BREATHING", "Difficulty breathing", {"symptoms_any": ["difficulty_breathing"]},
     "urgent", 0, "Difficulty breathing reported.", "Measure SpO2 and respiratory rate; assess airway.",
     "UCG 1.4 (unverified wording)"),
    ("RF-CHEST-PAIN", "Chest pain", {"symptoms_any": ["chest_pain"]},
     "urgent", 0, "Chest pain reported.", "Assess for cardiac causes; ECG where available; see Ischaemic Heart Disease.",
     "UCG 4.1.7 (unverified wording)"),
]


def seed(con, signoff: dict) -> int:
    rows = [(rid, name, __import__("json").dumps(cond), sev, refer, msg, act, src, 1 if rid in signoff else 0)
            for rid, name, cond, sev, refer, msg, act, src in RED_FLAG_SEEDS]
    con.executemany(
        """INSERT OR REPLACE INTO red_flag_rules
           (rule_id,name,condition_json,severity,refer,message,action,source_ref,reviewed) VALUES (?,?,?,?,?,?,?,?,?)""",
        rows)
    return len(rows)


def evaluate_all(con, case: PatientCase, require_reviewed: bool = False) -> tuple[list[RuleHit], list[UndeterminedRule]]:
    """Returns (fired rules ordered by severity, rules that could not be decided because inputs are missing)."""
    order = {"emergency": 0, "urgent": 1, "warning": 2}
    hits: list[RuleHit] = []
    undetermined: list[UndeterminedRule] = []
    q = "SELECT * FROM red_flag_rules WHERE enabled=1" + (" AND reviewed=1" if require_reviewed else "")
    for r in con.execute(q):
        res = evaluate(load_condition(r["condition_json"]), case)
        if res.matched:
            hits.append(RuleHit(r["rule_id"], "red_flag", r["severity"], r["message"], r["action"], bool(r["refer"]),
                                r["source_ref"], bool(r["reviewed"]), res.reasons))
            log.debug("red flag fired: %s (%s)", r["rule_id"], "; ".join(res.reasons))
        elif res.missing:
            undetermined.append(UndeterminedRule(r["rule_id"], r["name"], res.missing))
    hits.sort(key=lambda h: order[h.severity])
    return hits, undetermined
