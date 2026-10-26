"""
rules/dosing.py - weight-based dose calculator over table dose_rules.

Deliberately conservative:
  * only drug/indication pairs present in dose_rules are ever calculated (the LLM never produces a dose);
  * weight is required; if it is missing the result says so instead of guessing;
  * the guideline's own cap (max_single_dose) is applied and reported;
  * several rules for one drug are all returned, so differences inside the guideline stay visible.
The three seeds were copied from UCG 2023 text (cited in source_ref) and are unreviewed until signed off.
"""
from __future__ import annotations

import logging

from schemas import DoseResult, PatientCase

log = logging.getLogger(__name__)

# (rule_id, drug, indication, route, per_kg_min, per_kg_max, unit, max_single, interval_h, notes, min_age_m, max_age_m, source)
DOSE_SEEDS = [
    ("DR-PARACETAMOL-FEVER-CHILD", "paracetamol", "fever (febrile convulsions)", "oral", 15, 15, "mg", 1000, 6,
     "Give until fever subsides; plus tepid sponging.", None, 144, "UCG 1.1.5 Febrile Convulsions: 15 mg/kg every 6 hours"),
    ("DR-PARACETAMOL-MALARIA-HYPERPYREXIA-CHILD", "paracetamol", "hyperpyrexia in severe malaria", "oral", 10, 10, "mg",
     1000, 6, "Adult dose in the guideline is a fixed 1 g every 6 hours.", None, 144,
     "UCG 2.5.2.2 / 2.5.3.3 severe malaria complications: child 10 mg/kg"),
    ("DR-DIAZEPAM-RECTAL-CONVULSION", "diazepam", "convulsions", "rectal", 0.5, 0.5, "mg", 10, None,
     "Repeat once if needed after 10 minutes (as written in the guideline).", None, None,
     "UCG 1.1.5 Febrile Convulsions: 500 micrograms/kg rectally, maximum 10 mg"),
    ("DR-DEXTROSE10-HYPOGLYCAEMIA-CHILD", "dextrose 10%", "hypoglycaemia, unconscious child", "IV", 2, 5, "ml", None, None,
     "Only if the child cannot swallow; check response and glucose.", None, 144,
     "UCG 1.1.6 Hypoglycaemia: child dextrose 10% IV 2-5 ml/kg"),
]


def seed(con, signoff: dict) -> int:
    rows = [(rid, drug, ind, route, lo, hi, unit, mx, ivl, notes, mina, maxa, src, 1 if rid in signoff else 0)
            for rid, drug, ind, route, lo, hi, unit, mx, ivl, notes, mina, maxa, src in DOSE_SEEDS]
    con.executemany(
        """INSERT OR REPLACE INTO dose_rules
           (rule_id,drug,indication,route,dose_per_kg_min,dose_per_kg_max,dose_unit,max_single_dose,interval_hours,
            notes,min_age_months,max_age_months,source_ref,reviewed) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    return len(rows)


def _round(v: float) -> float:
    return round(v, 1) if v < 100 else round(v)


def calculate(con, case: PatientCase, drug: str, indication: str | None = None,
              require_reviewed: bool = False) -> tuple[list[DoseResult], list[str]]:
    """Returns (doses, needs). `needs` lists missing inputs (e.g. ['weight_kg']) when rules exist but can't run."""
    d = drug.lower().strip()
    q = ("SELECT * FROM dose_rules WHERE enabled=1 AND (lower(drug)=? OR lower(drug) LIKE ? || '%' OR ? LIKE lower(drug) || '%')"
         + (" AND reviewed=1" if require_reviewed else ""))
    rules = con.execute(q, (d, d, d)).fetchall()
    if indication:
        rules = [r for r in rules if indication.lower() in r["indication"].lower()] or rules
    if not rules:
        return [], []
    if case.weight_kg is None:
        return [], ["weight_kg"]
    results: list[DoseResult] = []
    for r in rules:
        if r["min_age_months"] is not None or r["max_age_months"] is not None:
            if case.age_months is None:
                return results, ["age_months"]
            if r["min_age_months"] is not None and case.age_months < r["min_age_months"]:
                continue
            if r["max_age_months"] is not None and case.age_months >= r["max_age_months"]:
                continue
        lo, hi = r["dose_per_kg_min"] * case.weight_kg, r["dose_per_kg_max"] * case.weight_kg
        capped = False
        if r["max_single_dose"] is not None and hi > r["max_single_dose"]:
            lo, hi, capped = min(lo, r["max_single_dose"]), r["max_single_dose"], True
        per = (f"{r['dose_per_kg_min']:g}" if r["dose_per_kg_min"] == r["dose_per_kg_max"]
               else f"{r['dose_per_kg_min']:g}-{r['dose_per_kg_max']:g}")
        why = f"{per} {r['dose_unit']}/kg x {case.weight_kg:g} kg"
        if capped:
            why += f"; capped at the guideline maximum of {r['max_single_dose']:g} {r['dose_unit']}"
        if r["notes"]:
            why += f". {r['notes']}"
        results.append(DoseResult(r["drug"], r["indication"], r["route"], _round(lo), _round(hi), r["dose_unit"],
                                  r["interval_hours"], capped, why, r["source_ref"], bool(r["reviewed"])))
    return results, []
