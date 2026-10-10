"""Turn a FHIR R4 Bundle (for example from Synthea) into a compact list of citable facts.

No LLM involved: this is deterministic, auditable and fast. Each fact keeps its FHIR reference, so every
eligibility decision can point back to the exact resource in the record.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from .schemas import Fact, PatientFacts

SNOMED = "http://snomed.info/sct"
LOINC = "http://loinc.org"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
SYSTEM_NAMES = {SNOMED: "SNOMED", LOINC: "LOINC", RXNORM: "RxNorm"}


def _date(value: str | None) -> str | None:
    return value[:10] if value else None


def to_date(value: str | None) -> date | None:
    """Parse full or partial FHIR dates (YYYY, YYYY-MM, YYYY-MM-DD...); partial dates use the first day."""
    if not value:
        return None
    m = re.match(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", value)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2) or 1), int(m.group(3) or 1))
    except ValueError:
        return None


# Synthea records social determinants as conditions; they are noise for eligibility.
SOCIAL = re.compile(
    r"employment|labor force|unemploy|education|housing|homeless|social (?:contact|isolation)|"
    r"stress \(|transport|violence|criminal|refugee|income|insurance|medication review due|"
    r"part-time|full-time|not in labor",
    re.I,
)


def _dosage_text(dosage: list[dict]) -> str | None:
    """Dosage as text; Synthea omits `text`, so rebuild it from doseAndRate and timing."""
    if not dosage:
        return None
    d = dosage[0]
    if d.get("text"):
        return d["text"]
    parts = []
    dq = (d.get("doseAndRate") or [{}])[0].get("doseQuantity") or {}
    if dq.get("value") is not None:
        parts.append(f"{dq['value']:g} {dq.get('unit', '')}".strip())
    rep = (d.get("timing") or {}).get("repeat") or {}
    if rep.get("frequency"):
        parts.append(f"{rep['frequency']} times per {rep.get('period', 1):g} {rep.get('periodUnit', 'd')}")
    if d.get("asNeededBoolean"):
        parts.append("as needed")
    return " ".join(parts) or None


def _coding(cc: dict | None) -> tuple[str, str | None, str | None]:
    """(display text, system short name, code) from a CodeableConcept."""
    if not cc:
        return "", None, None
    codings = cc.get("coding") or []
    first = codings[0] if codings else {}
    text = cc.get("text") or first.get("display") or first.get("code") or ""
    system = SYSTEM_NAMES.get(first.get("system", ""), first.get("system"))
    return text, system, first.get("code")


def _status(cc: dict | None) -> str | None:
    if not cc:
        return None
    codings = cc.get("coding") or []
    return codings[0].get("code") if codings else cc.get("text")


def _age(birth: str | None, as_of: date) -> int | None:
    b = to_date(birth)
    if b is None:
        return None
    return as_of.year - b.year - ((as_of.month, as_of.day) < (b.month, b.day))


def load_bundle(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_facts(bundle: dict[str, Any], as_of: date | None = None, keep_findings: bool = False) -> PatientFacts:
    """Build the fact list.

    * Conditions: all, with clinical status, onset and abatement. Synthea's social determinants (employment,
      education, housing, ...) are dropped unless `keep_findings`; clinical findings and situations stay.
    * Observations: the latest value per code (with how many earlier values exist), numeric or coded.
    * Medications: MedicationRequest and MedicationStatement with status and dosage text.
    * Procedures and allergies.
    """
    as_of = as_of or date.today()
    resources = [e.get("resource", {}) for e in bundle.get("entry", [])]
    patient = next((r for r in resources if r.get("resourceType") == "Patient"), {})
    medications = {f"Medication/{r.get('id')}": r for r in resources if r.get("resourceType") == "Medication"}
    facts: list[Fact] = []

    def add(**kw: Any) -> None:
        facts.append(Fact(id=f"F{len(facts) + 1}", **kw))

    latest_obs: dict[str, tuple[str, dict]] = {}
    obs_counts: dict[str, int] = {}
    for r in resources:
        rtype, rid = r.get("resourceType"), r.get("id", "")
        ref = f"{rtype}/{rid}"
        if rtype == "Condition":
            text, system, code = _coding(r.get("code"))
            if not keep_findings and SOCIAL.search(text):
                continue
            clinical = _status(r.get("clinicalStatus"))
            if r.get("abatementDateTime") and clinical in (None, "active"):
                clinical = "resolved"
            add(
                ref=ref,
                kind="condition",
                text=text,
                system=system,
                code=code,
                date=_date(r.get("onsetDateTime") or r.get("recordedDate")),
                status=clinical,
                end_date=_date(r.get("abatementDateTime")),
            )
        elif rtype == "Observation":
            if r.get("status") in {"entered-in-error", "cancelled", "preliminary", "registered", "unknown"}:
                continue
            text, system, code = _coding(r.get("code"))
            key = f"{system}|{code or text}"
            when = r.get("effectiveDateTime") or r.get("issued") or ""
            if when and to_date(when) is None:
                continue
            obs_counts[key] = obs_counts.get(key, 0) + 1
            if key not in latest_obs or when > latest_obs[key][0]:
                latest_obs[key] = (when, r)
            elif when == latest_obs[key][0]:
                prior = latest_obs[key][1]
                if r.get("status") in {"corrected", "amended"} and prior.get("status") not in {"corrected", "amended"}:
                    latest_obs[key] = (when, r)
                elif (
                    prior.get("status") not in {"corrected", "amended"} or r.get("status") in {"corrected", "amended"}
                ) and (r.get("valueQuantity"), r.get("component")) != (
                    prior.get("valueQuantity"),
                    prior.get("component"),
                ):
                    latest_obs[key] = (when, {**r, "status": "conflicting"})
        elif rtype in {"MedicationRequest", "MedicationStatement"}:
            concept = r.get("medicationCodeableConcept")
            if not concept and r.get("medicationReference"):
                mref = r["medicationReference"].get("reference", "")
                contained = {f"#{c.get('id')}": c for c in r.get("contained", [])}
                med_res = contained.get(mref) or medications.get(mref) or {}
                concept = med_res.get("code") or {"text": r["medicationReference"].get("display", "")}
            text, system, code = _coding(concept)
            dose_text = _dosage_text(r.get("dosageInstruction") or r.get("dosage") or [])
            add(
                ref=ref,
                kind="medication",
                text=text,
                system=system,
                code=code,
                value_text=dose_text,
                date=_date(r.get("authoredOn") or (r.get("effectivePeriod") or {}).get("start")),
                status=r.get("status"),
                end_date=_date((r.get("dispenseRequest", {}).get("validityPeriod") or {}).get("end")),
            )
        elif rtype == "Procedure":
            text, system, code = _coding(r.get("code"))
            performed = r.get("performedDateTime") or (r.get("performedPeriod") or {}).get("start")
            add(
                ref=ref,
                kind="procedure",
                text=text,
                system=system,
                code=code,
                date=_date(performed),
                status=r.get("status"),
            )
        elif rtype == "AllergyIntolerance":
            text, system, code = _coding(r.get("code"))
            add(ref=ref, kind="allergy", text=text, system=system, code=code, status=_status(r.get("clinicalStatus")))

    for key, (when, r) in sorted(latest_obs.items(), key=lambda kv: kv[0]):
        text, system, code = _coding(r.get("code"))
        # Panels such as blood pressure keep their values in components: one fact per component.
        for comp in r.get("component") or []:
            ctext, csys, ccode = _coding(comp.get("code"))
            cq = comp.get("valueQuantity") or {}
            if cq.get("value") is not None:
                add(
                    ref=f"Observation/{r.get('id', '')}",
                    kind="observation",
                    text=ctext or text,
                    system=csys,
                    code=ccode,
                    value=cq.get("value"),
                    unit=cq.get("unit") or cq.get("code"),
                    date=when or None,
                    status=r.get("status"),
                )
        if r.get("component") and not r.get("valueQuantity"):
            continue
        vq = r.get("valueQuantity") or {}
        value_text = None
        if not vq and r.get("valueCodeableConcept"):
            value_text = _coding(r.get("valueCodeableConcept"))[0]
        elif not vq and r.get("valueString"):
            value_text = r["valueString"]
        n_prev = obs_counts[key] - 1
        label = text + (f" (latest of {obs_counts[key]})" if n_prev else "")
        add(
            ref=f"Observation/{r.get('id', '')}",
            kind="observation",
            text=label,
            system=system,
            code=code,
            value=vq.get("value"),
            unit=vq.get("unit") or vq.get("code"),
            value_text=value_text,
            date=when or None,
            status=r.get("status"),
        )

    # Newest first within each kind, and one fact per medication name (the newest), so that caps keep
    # current information when a Synthea record has hundreds of entries.
    def _key(f: Fact):
        return (f.kind, f.date or "")

    seen_meds: set[str] = set()
    ordered: list[Fact] = []
    for f in sorted(facts, key=_key, reverse=True):
        if f.kind == "medication":
            name = f.text.lower()
            if name in seen_meds:
                continue
            seen_meds.add(name)
        ordered.append(f)
    facts = [
        f.model_copy(update={"id": f"F{i + 1}"})
        for i, f in enumerate(
            sorted(
                ordered,
                key=lambda f: (
                    ["demographic", "condition", "medication", "procedure", "allergy", "observation"].index(f.kind),
                    "" if f.status in {"active", None, "final"} else "z",
                    [-ord(c) for c in (f.date or "")],
                ),
            )
        )
    ]
    gender = patient.get("gender", "unknown")
    return PatientFacts(
        patient_ref=f"Patient/{patient.get('id', '')}",
        age_years=_age(patient.get("birthDate"), as_of),
        sex=gender if gender in {"female", "male", "other"} else "unknown",
        as_of=as_of.isoformat(),
        facts=facts,
    )


_WORD = re.compile(r"[a-z0-9]+")


def relevant_facts(pf: PatientFacts, query: str, kinds: set[str] | None = None, k: int = 25) -> list:
    """Keyword retrieval: rank facts by word overlap with the criterion (and its synonym terms)."""
    q = {w for w in _WORD.findall(query.lower()) if len(w) > 2}
    scored = []
    for f in pf.facts:
        if kinds and f.kind not in kinds:
            continue
        words = set(_WORD.findall(f"{f.text} {f.code or ''} {f.value_text or ''}".lower()))
        overlap = len(q & words)
        boost = 0.5 if f.status in {"active", None} else 0.0
        scored.append((overlap + (boost if overlap else 0), f))
    scored.sort(key=lambda s: (-s[0], s[1].id))
    top = [f for s, f in scored if s > 0][:k]
    if len(top) < 8:  # always give some context, e.g. for "no history of ..." criteria
        extra = [f for _, f in scored if f not in top and f.kind in {"condition", "medication"}]
        top += extra[: 8 - len(top)]
    return top
