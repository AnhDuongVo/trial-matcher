"""Deterministic checks: numbers and demographics are decided by code, not by the LLM.

Rules cover age, sex, pregnancy for male patients and lab thresholds (with a recency window and unit
check). Anything a rule cannot decide confidently returns None and goes to the LLM.
"""

from __future__ import annotations

import re

from .fhir_facts import to_date
from .schemas import Assessment, Criterion, Fact, PatientFacts

_COMPOUND = re.compile(r"\b(or|and|unless|either|except|if)\b", re.I)

LAB_ALIASES = {
    "hba1c": ["4548-4", "17856-6", "hemoglobin a1c", "a1c", "glycated"],
    "egfr": ["33914-3", "62238-1", "98979-8", "48642-3", "48643-1", "glomerular filtration"],
    "bmi": ["39156-5", "body mass index"],
    "uacr": ["9318-7", "14959-1", "albumin/creatinine", "albumin creatinine"],
    "ldl": ["18262-6", "13457-7", "ldl"],
    "creatinine": ["2160-0", "38483-4", "creatinine [mass/volume] in serum"],
    "systolic": ["8480-6", "systolic"],
    "diastolic": ["8462-4", "diastolic"],
}
_ALIAS_KEYS = {
    "hba1c": ["hba1c", "a1c", "glycated", "glycosylated"],
    "egfr": ["egfr", "glomerular", "gfr"],
    "bmi": ["bmi", "body mass"],
    "uacr": ["uacr", "albumin-to-creatinine", "albumin to creatinine", "albumin/creatinine", "acr"],
    "ldl": ["ldl"],
    "creatinine": ["serum creatinine"],
    "systolic": ["systolic"],
    "diastolic": ["diastolic"],
}


def _norm_unit(u: str | None) -> str:
    return re.sub(r"[\s{}_]", "", (u or "").lower()).replace("sqm", "m2")


def find_lab(pf: PatientFacts, crit: Criterion) -> Fact | None:
    observations = [f for f in pf.facts if f.kind == "observation" and f.value is not None]
    if crit.loinc:
        hit = next((f for f in observations if f.code == crit.loinc), None)
        if hit:
            return hit
    query = f"{crit.lab_name or ''} {crit.text} {' '.join(crit.terms)}".lower()
    for key, words in _ALIAS_KEYS.items():
        if any(w in query for w in words):
            aliases = LAB_ALIASES[key]
            hit = next(
                (f for f in observations if f.code in aliases or any(a in f.text.lower() for a in aliases)), None
            )
            if hit:
                return hit
    return None


def _decide(value: float, crit: Criterion) -> bool | None:
    c, lo, hi = crit.comparator, crit.value, crit.value_high
    if lo is None:
        return None
    if c == "between" and hi is not None:
        return min(lo, hi) <= value <= max(lo, hi)
    return {"<": value < lo, "<=": value <= lo, ">": value > lo, ">=": value >= lo}.get(c or "")


def apply_rules(crit: Criterion, pf: PatientFacts, max_lab_age_days: int = 365) -> Assessment | None:
    def result(status, confidence, evidence: list[Fact], rationale: str) -> Assessment:
        return Assessment(
            index=crit.index,
            kind=crit.kind,
            criterion=crit.text,
            status=status,
            confidence=confidence,
            evidence=[f.id for f in evidence],
            evidence_refs=[f.ref for f in evidence],
            rationale=rationale,
            method="rule",
        )

    if crit.category == "age" and (crit.age_min is not None or crit.age_max is not None):
        if pf.age_years is None:
            return None
        ok = (crit.age_min is None or pf.age_years >= crit.age_min) and (
            crit.age_max is None or pf.age_years <= crit.age_max
        )
        return result(
            "met" if ok else "not_met",
            0.99,
            [],
            f"age {pf.age_years} vs [{crit.age_min}, {crit.age_max}] on {pf.as_of}",
        )

    if crit.category == "sex" and crit.sex in {"female", "male"} and pf.sex in {"female", "male"}:
        return result("met" if pf.sex == crit.sex else "not_met", 0.99, [], f"recorded sex {pf.sex}")

    if crit.category == "pregnancy" and pf.sex == "male":
        # Exclusion "pregnancy": does not apply. Inclusion "negative pregnancy test in women...": not applicable.
        return result(
            "not_met" if crit.kind == "exclusion" else "met",
            0.95,
            [],
            "male patient: pregnancy criterion " + ("does not apply" if crit.kind == "exclusion" else "not applicable"),
        )

    if crit.category == "lab" and crit.comparator and crit.value is not None:
        # Compound logic ("creatinine > 1.5 or eGFR < 30") goes to the LLM; "between x and y" is fine.
        if _COMPOUND.search(re.sub(r"between\s+\S+\s+and\s+\S+", " ", crit.text, flags=re.I)):
            return None
        fact = find_lab(pf, crit)
        if fact is None or fact.value is None:
            return None  # LLM decides; usually "unknown: no measurement in the record"
        if crit.unit and fact.unit and _norm_unit(crit.unit) != _norm_unit(fact.unit):
            return None  # units differ (e.g. mmol/mol vs %): let the LLM reason, or a human convert
        measured = to_date(fact.date)
        if measured is None:
            return None  # no usable date: cannot judge recency
        age_days = (to_date(pf.as_of) - measured).days
        if age_days > max_lab_age_days:
            return result(
                "unknown",
                0.7,
                [fact],
                f"latest {fact.text} ({fact.value:g} {fact.unit or ''}) is from {fact.date}, "
                f"older than {max_lab_age_days} days: needs a current value",
            )
        decided = _decide(fact.value, crit)
        if decided is None:
            return None
        return result(
            "met" if decided else "not_met",
            0.95,
            [fact],
            f"{fact.text} = {fact.value:g} {fact.unit or ''} on {fact.date} vs {crit.comparator} "
            f"{crit.value:g}{'' if crit.value_high is None else f'-{crit.value_high:g}'}",
        )
    return None
