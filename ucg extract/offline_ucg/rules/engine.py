"""
rules/engine.py - tiny, transparent rule engine shared by red_flags / contraindications / referral.

A rule's condition is a JSON object; EVERY key must hold (AND).  Use "any_of" for OR.

  symptoms_any: [..]        at least one canonical symptom present
  symptoms_all: [..]        all present
  symptoms_none: [..]       none present
  age_months_lt / age_months_gte / age_years_lt / age_years_gte
  pregnant: true|false
  sex: "F"|"M"
  temp_c_gte, spo2_lt, hr_gt, rr_gte, sbp_lt, sbp_gte, dbp_gte
  rr_gte_by_age: [[min_month, max_month, threshold], ...]   e.g. fast breathing by age band
  allergy_any / comorbidity_any / current_meds_any: [..]    (substring match, lower-case)
  any_of: [ {cond}, {cond} ]

Unknown inputs never fire a rule; they are reported in `missing` so the UI can ask for them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from schemas import PatientCase

_NUMERIC = {  # key -> (patient attribute, comparator)
    "temp_c_gte": ("temp_c", lambda v, t: v >= t),
    "spo2_lt": ("spo2", lambda v, t: v < t),
    "hr_gt": ("hr", lambda v, t: v > t),
    "rr_gte": ("rr", lambda v, t: v >= t),
    "sbp_lt": ("sbp", lambda v, t: v < t),
    "sbp_gte": ("sbp", lambda v, t: v >= t),
    "dbp_gte": ("dbp", lambda v, t: v >= t),
    "age_months_lt": ("age_months", lambda v, t: v < t),
    "age_months_gte": ("age_months", lambda v, t: v >= t),
    "age_years_lt": ("age_years", lambda v, t: v < t),
    "age_years_gte": ("age_years", lambda v, t: v >= t),
}


@dataclass
class CondResult:
    matched: bool
    missing: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def _has_any(items: list[str], needles: list[str]) -> list[str]:
    low = [i.lower() for i in items]
    return [n for n in needles if any(n.lower() in i for i in low)]


def evaluate(cond: dict, case: PatientCase) -> CondResult:
    """AND over every key. A key that is definitely false decides the rule (matched=False, nothing missing).
    A key whose input is unknown makes the rule 'undetermined' ONLY if no other key is definitely false."""
    reasons: list[str] = []
    unknown: list[str] = []
    failed = False
    for key, val in cond.items():
        if key == "any_of":
            sub_unknown: set[str] = set()
            hit = None
            for sub in val:
                r = evaluate(sub, case)
                if r.matched:
                    hit = r
                    break
                sub_unknown.update(r.missing)
            if hit:
                reasons.extend(hit.reasons)
            elif sub_unknown:
                unknown.extend(sub_unknown)
            else:
                failed = True
        elif key in _NUMERIC:
            attr, cmp_ = _NUMERIC[key]
            v = getattr(case, attr)
            if v is None:
                unknown.append(attr)
            elif cmp_(v, val):
                reasons.append(f"{attr}={v:g} ({key} {val})")
            else:
                failed = True
        elif key == "rr_gte_by_age":
            if case.rr is None:
                unknown.append("rr")
            elif case.age_months is None:
                unknown.append("age_months")
            else:
                band = next((b for b in val if b[0] <= case.age_months < b[1]), None)
                if band and case.rr >= band[2]:
                    reasons.append(f"rr={case.rr:g} >= {band[2]} for age {case.age_months:g} months")
                else:
                    failed = True
        elif key == "symptoms_any":
            hit = [x for x in val if x in case.symptoms]
            if hit:
                reasons.append("symptoms: " + ", ".join(hit))
            else:
                failed = True
        elif key == "symptoms_all":
            if all(x in case.symptoms for x in val):
                reasons.append("symptoms: " + ", ".join(val))
            else:
                failed = True
        elif key == "symptoms_none":
            if any(x in case.symptoms for x in val):
                failed = True
        elif key == "pregnant":
            if case.pregnant is None:
                unknown.append("pregnant")
            elif case.pregnant == val:
                reasons.append(f"pregnant={val}")
            else:
                failed = True
        elif key == "sex":
            if case.sex is None:
                unknown.append("sex")
            elif case.sex != val:
                failed = True
        elif key in ("allergy_any", "comorbidity_any", "current_meds_any"):
            src = {"allergy_any": case.allergies, "comorbidity_any": case.comorbidities,
                   "current_meds_any": case.current_meds}[key]
            hit = _has_any(src, val)
            if hit:
                reasons.append(f"{key.split('_')[0]}: " + ", ".join(hit))
            else:
                failed = True
        else:
            raise ValueError(f"Unknown condition key: {key!r}")
    if failed:
        return CondResult(False)
    if unknown:
        return CondResult(False, sorted(set(unknown)))
    return CondResult(True, [], reasons)


def load_condition(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:  # surface the broken rule instead of silently skipping it
        raise ValueError(f"Invalid condition_json: {raw!r}: {e}") from e


def fingerprint(con, tables=("red_flag_rules", "contraindication_rules", "dose_rules")) -> str:
    """Stable hash of all rule content -> stored in every audit record so a decision can be reproduced."""
    import hashlib
    h = hashlib.sha256()
    for t in tables:
        for row in con.execute(f"SELECT * FROM {t} ORDER BY 1"):
            h.update(repr(tuple(row)).encode())
    return h.hexdigest()[:16]


def norm_drug(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()
