"""Data models for trial matching. Every decision carries evidence (FHIR resource references) and a confidence."""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field

Status = Literal["met", "not_met", "unknown"]
Verdict = Literal["eligible", "ineligible", "needs_review"]
FactKind = Literal["condition", "observation", "medication", "procedure", "allergy", "demographic"]


def _coerce_fact_ids(value: Any) -> list[str]:
    """Accept ["F3", 3, "[F4]"] and normalise to ["F3", "F4"]."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    out: list[str] = []
    for v in value:
        m = re.search(r"\d+", str(v))
        if m and f"F{int(m.group())}" not in out:
            out.append(f"F{int(m.group())}")
    return out


FactIds = Annotated[list[str], BeforeValidator(_coerce_fact_ids)]


class Fact(BaseModel):
    """One clinical fact taken from the FHIR record, with a pointer back to its resource."""

    id: str  # F1, F2, ... (what the model cites)
    ref: str  # FHIR reference, e.g. "Condition/123"
    kind: FactKind
    text: str  # human-readable label, e.g. "Diabetes mellitus type 2"
    system: str | None = None
    code: str | None = None
    value: float | None = None
    unit: str | None = None
    value_text: str | None = None
    date: str | None = None  # ISO date of onset / measurement / authoring
    status: str | None = None  # active, resolved, stopped, final, ...
    end_date: str | None = None

    def render(self) -> str:
        parts = [f"[{self.id}] {self.kind}: {self.text}"]
        if self.code:
            parts.append(f"({self.system or ''} {self.code})".replace("( ", "("))
        if self.value is not None:
            parts.append(f"= {self.value:g} {self.unit or ''}".rstrip())
        elif self.value_text:
            parts.append(f"= {self.value_text}")
        if self.status:
            parts.append(f"status {self.status}")
        if self.date:
            parts.append(f"date {self.date}")
        if self.end_date:
            parts.append(f"until {self.end_date}")
        return " ".join(parts)


class PatientFacts(BaseModel):
    patient_ref: str
    age_years: int | None
    sex: Literal["female", "male", "other", "unknown"]
    as_of: str
    facts: list[Fact]

    def by_id(self) -> dict[str, Fact]:
        return {f.id: f for f in self.facts}


class Criterion(BaseModel):
    """One eligibility criterion, parsed into something partly machine-checkable."""

    index: int
    kind: Literal["inclusion", "exclusion"]
    text: str
    category: Literal["age", "sex", "condition", "lab", "medication", "procedure", "pregnancy", "other"] = "other"
    age_min: float | None = None
    age_max: float | None = None
    sex: Literal["female", "male", "all"] | None = None
    lab_name: str | None = None
    loinc: str | None = None
    comparator: Literal["<", "<=", ">", ">=", "between"] | None = None
    value: float | None = None
    value_high: float | None = None
    unit: str | None = None
    terms: list[str] = Field(default_factory=list, description="Synonyms and codes useful to find evidence")


class ParsedCriteria(BaseModel):
    criteria: list[Criterion]


class Assessment(BaseModel):
    index: int
    kind: Literal["inclusion", "exclusion"]
    criterion: str
    status: Status
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: FactIds = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    rationale: str = ""
    method: Literal["rule", "llm", "reviewer"] = "llm"
    reviewer_note: str | None = None


class LLMAssessment(BaseModel):
    status: Status
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: FactIds = Field(default_factory=list)
    rationale: str


class Trial(BaseModel):
    nct_id: str
    title: str
    criteria_text: str
    minimum_age: str | None = None
    maximum_age: str | None = None
    sex: str | None = None
    url: str = ""


class MatchResult(BaseModel):
    patient_ref: str
    trial: Trial
    verdict: Verdict
    confidence: float
    assessments: list[Assessment]
    reviewed_by: str | None = None
    reviewed_at: str | None = None
