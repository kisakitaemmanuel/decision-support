"""
schemas.py - plain dataclasses passed between agents. No third-party dependencies, JSON-friendly via to_dict().
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class Serializable:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[call-overload]


# ------------------------------------------------------------------ patient
@dataclass
class PatientCase(Serializable):
    raw_text: str = ""
    age_months: float | None = None
    sex: str | None = None                       # "F" | "M"
    pregnant: bool | None = None
    weight_kg: float | None = None
    temp_c: float | None = None
    spo2: float | None = None
    hr: float | None = None
    rr: float | None = None
    sbp: float | None = None
    dbp: float | None = None
    symptoms: list[str] = field(default_factory=list)       # canonical keys, e.g. "convulsions"
    negated_symptoms: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    comorbidities: list[str] = field(default_factory=list)
    current_meds: list[str] = field(default_factory=list)
    facility_level: str = "HC3"
    missing_info: list[str] = field(default_factory=list)   # questions worth asking next
    extraction_notes: list[str] = field(default_factory=list)

    @property
    def age_years(self) -> float | None:
        return None if self.age_months is None else self.age_months / 12.0

    @property
    def is_child(self) -> bool | None:
        return None if self.age_months is None else self.age_months < 12 * 12


# ------------------------------------------------------------------ retrieval
@dataclass
class ChunkHit(Serializable):
    chunk_id: int
    disease_name: str
    section: str
    text: str
    page: int | None
    score: float
    rank: int
    source: str                                   # "fts" | "semantic" | "hybrid"


@dataclass
class DiseaseHit(Serializable):
    disease_name: str
    score: float
    chunks: list[ChunkHit] = field(default_factory=list)
    ranks: dict[str, int] = field(default_factory=dict)     # best rank per retriever -> debugging aid
    icd10_code: str | None = None


@dataclass
class RetrievalResult(Serializable):
    query: str
    diseases: list[DiseaseHit]
    degraded: list[str] = field(default_factory=list)       # e.g. ["semantic: embedder mismatch"]
    timings_ms: dict[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------ rules
@dataclass
class RuleHit(Serializable):
    rule_id: str
    kind: str                                     # red_flag | contraindication | referral
    severity: str
    message: str
    action: str = ""
    refer: bool = False
    source_ref: str = ""
    reviewed: bool = False
    matched_on: list[str] = field(default_factory=list)
    drug: str | None = None


@dataclass
class UndeterminedRule(Serializable):
    rule_id: str
    message: str
    missing: list[str]


@dataclass
class DoseResult(Serializable):
    drug: str
    indication: str
    route: str
    dose_min: float
    dose_max: float
    unit: str
    interval_hours: float | None
    capped: bool
    rationale: str
    source_ref: str
    reviewed: bool


# ------------------------------------------------------------------ clinical
@dataclass
class DiffItem(Serializable):
    disease_name: str
    score: float
    retrieval_score: float
    feature_score: float
    matched_features: list[str]
    unmatched_features: list[str]
    llm_nominated: bool = False
    evidence: list[str] = field(default_factory=list)
    page: int | None = None
    section_number: str | None = None
    icd10_code: str | None = None
    demographic_note: str = ""


@dataclass
class ReferralAdvice(Serializable):
    decision: str                                 # manage_here | refer_urgent | refer_routine
    reasons: list[str]
    facility_level: str
    levels_in_guideline: list[str] = field(default_factory=list)
    higher_level_components: list[str] = field(default_factory=list)


@dataclass
class ClinicalAssessment(Serializable):
    differential: list[DiffItem]
    referral: ReferralAdvice | None
    mode: str                                     # "llm+tools" | "deterministic"
    tool_trace: list[dict] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class MedicationReview(Serializable):
    drug: str
    status: str                                   # ok | caution | avoid | contraindicated
    messages: list[str]
    dose: DoseResult | None
    guideline_snippet: str
    needs: list[str] = field(default_factory=list)          # e.g. ["weight_kg"]
    rule_hits: list[RuleHit] = field(default_factory=list)


@dataclass
class Explanation(Serializable):
    text: str
    mode: str                                     # "template" | "llm"
    grounded: bool
    fallback_reason: str = ""
    citations: list[str] = field(default_factory=list)
