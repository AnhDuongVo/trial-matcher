from datetime import date

import pytest

from trial_matcher.fhir_facts import extract_facts, to_date
from trial_matcher.rules import apply_rules
from trial_matcher.schemas import Criterion, Fact, PatientFacts

C = Criterion(
    index=0,
    kind="inclusion",
    text="HbA1c >= 7.5%",
    category="lab",
    lab_name="HbA1c",
    comparator=">=",
    value=7.5,
    unit="%",
    max_age_days=30,
)


def patient(**updates):
    data = dict(
        id="F1",
        ref="Observation/x",
        kind="observation",
        text="HbA1c",
        code="4548-4",
        value=8.0,
        unit="%",
        date="2026-09-25",
        status="final",
    )
    data.update(updates)
    return PatientFacts(patient_ref="Patient/x", age_years=40, sex="female", as_of="2026-10-01", facts=[Fact(**data)])


@pytest.mark.parametrize(
    "updates",
    [
        {"unit": None},
        {"unit": "mmol/mol"},
        {"date": None},
        {"date": "2026-02-30"},
        {"date": "2026-10-02"},
        {"date": "2026-01-01"},
        {"status": "entered-in-error"},
        {"value": float("nan")},
    ],
)
def test_lab_fails_closed(updates):
    assert apply_rules(C, patient(**updates)).status == "unknown"


def test_protocol_recency_overrides_demo_fallback():
    assert apply_rules(C, patient(date="2026-08-01")).status == "unknown"
    assert apply_rules(C.model_copy(update={"max_age_days": 90}), patient(date="2026-08-01")).status == "met"


def test_compound_and_unparsed_quantities_require_review():
    assert apply_rules(C.model_copy(update={"text": "HbA1c > 7.5 or eGFR < 30"}), patient()).status == "unknown"
    assert apply_rules(C.model_copy(update={"comparator": None}), patient()).status == "unknown"
    dose = Criterion(index=1, kind="inclusion", category="medication", text="Metformin 1000 mg daily")
    assert apply_rules(dose, patient()).status == "unknown"


def obs(id, value, status="final", when="2026-09-01"):
    return {
        "resourceType": "Observation",
        "id": id,
        "status": status,
        "effectiveDateTime": when,
        "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "HbA1c"}]},
        "valueQuantity": {"value": value, "unit": "%"},
    }


def extract(*resources):
    return extract_facts({"entry": [{"resource": r} for r in resources]}, date(2026, 10, 1))


def test_invalid_latest_observation_is_excluded():
    pf = extract(obs("ok", 8), obs("bad", 2, "entered-in-error", "2026-09-30"))
    assert pf.facts[0].ref == "Observation/ok"


def test_corrected_record_and_tied_conflict():
    pf = extract(obs("old", 8), obs("correction", 9, "corrected"))
    assert pf.facts[0].value == 9
    pf = extract(obs("one", 8), obs("two", 9))
    assert pf.facts[0].status == "conflicting"
    assert apply_rules(C, pf).status == "unknown"


def test_bad_calendar_date_is_unknown():
    assert to_date("2026-02-30") is None
