"""
agents/patient_agent.py - free text / structured input -> PatientCase.

Deterministic first: numeric facts (age, weight, vitals) are ONLY taken from regexes with plausibility ranges, never
from an LLM.  An LLM (if present) may add symptom keys, restricted to the known vocabulary below.
"""
from __future__ import annotations

import logging
import re

from agents.base import extract_json
from etl.drugs import DRUG_NAMES
from schemas import PatientCase

log = logging.getLogger(__name__)

# canonical symptom -> regex fragments (matched with a leading \b)
SYMPTOMS: dict[str, list[str]] = {
    "fever": ["fever", "febrile", "high temperature", "elevated temperature", "raised temperature", "pyrexia", "hot body"],
    "convulsions": ["convuls", r"seizure", r"fits?\b", "fitting"],
    "unconscious": ["unconscious", "unresponsive", "not responding", "comatose", r"coma\b", "passed out"],
    "lethargy": ["lethargic", "lethargy", "very sleepy", "drowsy", "floppy", r"limp\b"],
    "confusion": ["confus", "disorient", "altered mental"],
    "unable_to_drink": ["unable to drink", "not able to drink", "cannot drink", "can't drink", "refuses? to drink",
                        "not drinking", "unable to breastfeed", "not breastfeeding", "cannot breastfeed"],
    "poor_feeding": ["poor feeding", "feeding poorly", "reduced feeding"],
    "vomiting_everything": ["vomits? everything", "vomiting everything", "vomiting all"],
    "vomiting": ["vomit", "throwing up", "nausea"],
    "diarrhoea": ["diarrh", "loose stool", "watery stool"],
    "blood_in_stool": ["blood in (the )?stool", "bloody stool", "bloody diarrh", "dysentery"],
    "cough": ["cough"],
    "difficulty_breathing": ["difficulty breathing", "difficult breathing", "short of breath", "shortness of breath",
                             "breathless", "dyspn", "struggling to breathe", r"labou?red breathing"],
    "fast_breathing": ["fast breathing", "rapid breathing", "breathing fast", "tachypn"],
    "chest_indrawing": ["chest in-?drawing", "indrawing", "recession"],
    "stridor": ["stridor"],
    "wheeze": ["wheez"],
    "chest_pain": ["chest pain", "chest tightness", "pain in the chest"],
    "headache": ["headache", "head ache"],
    "neck_stiffness": ["neck stiff", "stiff neck", "neck rigid"],
    "photophobia": ["photophobia", "light hurts", "sensitive to light"],
    "abdominal_pain": ["abdominal pain", "stomach pain", "belly pain", "tummy pain", "stomach ache", "epigastric pain"],
    "jaundice": ["jaundice", "yellow eyes", "yellowing of the eyes", "yellow skin"],
    "rash": ["rash", "skin eruption"],
    "severe_bleeding": ["severe bleeding", "heavy bleeding", "bleeding heavily", "haemorrhag", "hemorrhag",
                        "bleeding a lot", "profuse bleeding"],
    "vaginal_bleeding": ["vaginal bleeding"],
    "joint_pain": ["joint pain", "arthralgia", "painful joints"],
    "dysuria": ["painful urination", "burning urination", "dysuria", "burning when passing urine", "pain on urinating"],
    "weight_loss": ["weight loss", "losing weight"],
    "night_sweats": ["night sweats"],
    "fatigue": ["fatigue", "tired", r"weakness", r"weak\b"],
    "pallor": ["pallor", r"pale\b"],
    "swelling_legs": ["swollen legs", "leg swelling", "swelling of the legs", "oedema", "edema"],
    "polyuria_polydipsia": ["polyuria", "excessive thirst", "polydipsia", "passing a lot of urine"],
    "sore_throat": ["sore throat", "throat pain"],
    "ear_pain": ["ear pain", "earache", "ear discharge"],
    "dehydration": ["dehydrat", "sunken eyes"],
    "itching": ["itch", "pruritus"],
    "palpitations": ["palpitation", "racing heart"],
    "dizziness": ["dizz", "vertigo", "light-?headed"],
}
COMORBIDITIES: dict[str, list[str]] = {
    "hiv": ["hiv", r"living with hiv", r"on art\b"], "diabetes": ["diabet"], "hypertension": ["hypertensi"],
    "asthma": ["asthma"], "sickle cell": ["sickle cell"], "epilepsy": ["epilep"],
    "kidney disease": ["kidney (disease|failure)", "renal (disease|failure|impairment)", r"ckd\b"],
    "liver disease": ["liver disease", "cirrhosis", "hepatitis"], "heart failure": ["heart failure"],
    "tuberculosis": ["tuberculosis", r"\btb\b"], "malnutrition": ["malnutrition", "malnourished"],
    "cancer": ["cancer"], "copd": [r"copd\b"],
}
NEGATION = re.compile(r"\b(no|not|denies|denied|without|negative for|absent|never|nil)\b(?:\s+[\w'-]+){0,3}\s*$", re.I)


def _compile(table: dict[str, list[str]]) -> dict[str, re.Pattern]:
    return {k: re.compile(r"\b(?:" + "|".join(v) + ")", re.I) for k, v in table.items()}


SYMPTOM_RE = _compile(SYMPTOMS)
COMORBID_RE = _compile(COMORBIDITIES)
_DRUG_ANY = re.compile(r"\b(" + "|".join(re.escape(d) for d in DRUG_NAMES) + r")\b", re.I)

RANGES = {"temp_c": (30, 45), "spo2": (50, 100), "hr": (20, 250), "rr": (4, 90), "sbp": (40, 300),
          "dbp": (20, 200), "weight_kg": (0.5, 250)}


def _scan(table: dict[str, re.Pattern], text: str) -> tuple[list[str], list[str]]:
    present, negated = [], []
    for key, rx in table.items():
        for m in rx.finditer(text):
            before = text[max(0, m.start() - 40):m.start()]
            (negated if NEGATION.search(before) else present).append(key)
            break
    present = [k for k in present]
    return present, [k for k in negated if k not in present]


def _in_range(name: str, v: float, notes: list[str]) -> float | None:
    lo, hi = RANGES[name]
    if lo <= v <= hi:
        return v
    notes.append(f"ignored implausible {name}={v:g}")
    return None


def parse_age_months(text: str) -> float | None:
    unit_m = {"year": 12.0, "yr": 12.0, "y": 12.0, "month": 1.0, "mo": 1.0, "m": 1.0, "week": 1 / 4.345, "wk": 1 / 4.345,
              "day": 1 / 30.4375, "d": 1 / 30.4375}

    def conv(num, unit):
        u = unit.lower().rstrip("s")
        return float(num) * unit_m.get(u, unit_m.get(u[:2], 12.0))

    pats = [r"(\d+(?:\.\d+)?)[\s-]*(year|yr|month|mo|week|wk|day)s?[\s-]*old",
            r"\bag(?:e|ed)\s*:?\s*(\d+(?:\.\d+)?)\s*(year|yr|y|month|mo|m|week|wk|day|d)s?\b",
            r"\b(\d+(?:\.\d+)?)\s*(?:yo|y/o)\b", r"^\s*(\d+(?:\.\d+)?)\s*(year|yr|month|mo|week|wk|day)s?\b"]
    for i, p in enumerate(pats):
        m = re.search(p, text, re.I)
        if m:
            return conv(m.group(1), m.group(2) if m.lastindex >= 2 else "year")
    if re.search(r"\b(newborn|neonate|neonatal)\b", text, re.I):
        return 0.25
    return None


class PatientAgent:
    def __init__(self, llm=None):
        self.llm = llm

    def parse(self, text: str, overrides: dict | None = None, facility_level: str = "HC3") -> PatientCase:
        notes: list[str] = []
        c = PatientCase(raw_text=text or "", facility_level=facility_level)
        t = text or ""

        c.age_months = parse_age_months(t)
        low = t.lower()
        if re.search(r"\b(woman|female|girl|she|her|mother)\b", low):
            c.sex = "F"
        elif re.search(r"\b(man|male|boy|he|his)\b", low):
            c.sex = "M"
        if re.search(r"\b(not pregnant|non-pregnant|no pregnancy)\b", low):
            c.pregnant = False
        elif re.search(r"\bpregnan|\bgravid|\bantenatal|\bG\d+P\d+|\b(first|second|third|1st|2nd|3rd) trimester", t, re.I):
            c.pregnant = True
            c.sex = c.sex or "F"

        def num(pattern: str, group: int = 1, flags=re.I):
            m = re.search(pattern, t, flags)
            return float(m.group(group)) if m else None

        w = num(r"(\d+(?:\.\d+)?)\s*kg\b")
        c.weight_kg = _in_range("weight_kg", w, notes) if w is not None else None
        temp = num(r"(?:\btemp(?:erature)?|\bT)\s*(?:of|is|=|:)?\s*(\d{2,3}(?:\.\d)?)\s*(?:°|deg(?:rees)?)?\s*[CF]?\b") \
            or num(r"(\d{2}(?:\.\d)?)\s*°\s*C")
        if temp is not None and temp > 45:
            notes.append(f"temperature {temp:g} looked like Fahrenheit; converted")
            temp = (temp - 32) * 5 / 9
        c.temp_c = _in_range("temp_c", round(temp, 1), notes) if temp is not None else None
        s = num(r"(?:sp\s?o2|o2 sats?|oxygen sat(?:uration)?s?|\bsats?)\s*(?:of|is|=|:)?\s*(\d{2,3})\s*%?")
        c.spo2 = _in_range("spo2", s, notes) if s is not None else None
        bp = re.search(r"(?:\bBP|blood pressure)\s*(?:of|is|=|:)?\s*(\d{2,3})\s*/\s*(\d{2,3})", t, re.I) or \
            re.search(r"(\d{2,3})\s*/\s*(\d{2,3})\s*mm\s?Hg", t, re.I)
        if bp:
            c.sbp = _in_range("sbp", float(bp.group(1)), notes)
            c.dbp = _in_range("dbp", float(bp.group(2)), notes)
        hr = num(r"(?:\bHR|pulse|heart rate)\s*(?:of|is|=|:)?\s*(\d{2,3})")
        c.hr = _in_range("hr", hr, notes) if hr is not None else None
        rr = num(r"(?:\bRR|resp(?:iratory)? rate|breathing rate)\s*(?:of|is|=|:)?\s*(\d{1,2})") or \
            num(r"(\d{1,2})\s*breaths?\s*(?:/|per\s*)\s*min")
        c.rr = _in_range("rr", rr, notes) if rr is not None else None

        if c.pregnant is None and (c.sex == "M" or (c.age_months is not None and c.age_months < 120)):
            c.pregnant = False
            notes.append("pregnancy set to False (male or under 10 years)")
        c.symptoms, c.negated_symptoms = _scan(SYMPTOM_RE, t)
        c.comorbidities, _ = _scan(COMORBID_RE, t)
        if c.temp_c is not None and c.temp_c >= 37.5 and "fever" not in c.symptoms:
            c.symptoms.append("fever")
            notes.append("fever inferred from measured temperature")

        for m in re.finditer(r"\b(?:allergic to|allergy to|allergies?\s*:)\s*([^.;\n]+)", t, re.I):
            for item in re.split(r",|\band\b|\bor\b", m.group(1)):
                item = " ".join(item.strip().lower().split()[:3])
                if item and item not in ("none", "nil", "no known"):
                    c.allergies.append(item)
        for m in re.finditer(r"\b([a-z]+(?:\s[a-z]+)?)\s+allergy\b", low):
            word = m.group(1).split()[-1]
            if word not in ("no", "known", "drug") and word not in c.allergies:
                c.allergies.append(word)
        for m in re.finditer(r"\b(?:taking|currently on|started on|using|on)\s+([^.;\n]+)", t, re.I):
            for d in _DRUG_ANY.findall(m.group(1)):
                if d.lower() not in c.current_meds:
                    c.current_meds.append(d.lower())

        if self.llm is not None and getattr(self.llm, "available", False):
            self._llm_symptoms(c, notes)
        if overrides:
            self._apply_overrides(c, overrides, notes)
        c.extraction_notes = notes
        c.missing_info = self._missing(c)
        log.debug("parsed case: %s", c.to_dict())
        return c

    # -------------------------------------------------------------------------------------------
    def _llm_symptoms(self, c: PatientCase, notes: list[str]) -> None:
        prompt = ("Extract symptoms from the clinical note. Reply with JSON {\"symptoms\": [...]} using ONLY keys from: "
                  + ", ".join(SYMPTOMS) + ". Exclude anything the note says is absent.\n\nNote: " + c.raw_text[:1500])
        try:
            obj = extract_json(self.llm.chat([{"role": "user", "content": prompt}], max_tokens=200, temperature=0.0,
                                             json_mode=True)) or {}
        except Exception as e:                           # noqa: BLE001
            notes.append(f"llm symptom extraction failed: {e}")
            return
        for k in obj.get("symptoms", []):
            if k in SYMPTOMS and k not in c.symptoms and k not in c.negated_symptoms:
                c.symptoms.append(k)
                notes.append(f"symptom added by LLM: {k}")

    def _apply_overrides(self, c: PatientCase, o: dict, notes: list[str]) -> None:
        simple = ["sex", "pregnant", "weight_kg", "temp_c", "spo2", "hr", "rr", "sbp", "dbp", "facility_level"]
        for k in simple:
            if o.get(k) is not None:
                setattr(c, k, o[k])
                notes.append(f"override: {k}={o[k]}")
        if o.get("age_years") is not None:
            c.age_months = float(o["age_years"]) * 12
            notes.append(f"override: age_years={o['age_years']}")
        if o.get("age_months") is not None:
            c.age_months = float(o["age_months"])
            notes.append(f"override: age_months={o['age_months']}")
        for k in ("symptoms", "allergies", "comorbidities", "current_meds"):
            for v in o.get(k) or []:
                if v not in getattr(c, k):
                    getattr(c, k).append(v)
                    notes.append(f"override: +{k[:-1] if k.endswith('s') else k}={v}")

    @staticmethod
    def _missing(c: PatientCase) -> list[str]:
        q: list[str] = []
        if c.age_months is None:
            q.append("What is the patient's age?")
        if c.weight_kg is None and (c.age_months is None or c.age_months < 144):
            q.append("What is the child's weight in kg? (needed for any dose calculation)")
        if c.sex == "F" and c.pregnant is None and c.age_months is not None and 144 <= c.age_months <= 600:
            q.append("Is she pregnant?")
        resp = {"cough", "difficulty_breathing", "fast_breathing", "wheeze"} & set(c.symptoms)
        if resp and c.spo2 is None:
            q.append("What is the oxygen saturation (SpO2)?")
        if (c.pregnant or "headache" in c.symptoms) and c.sbp is None:
            q.append("What is the blood pressure?")
        return q
