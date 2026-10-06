"""Generate the bundled synthetic FHIR patients and synthetic trials (deterministic, no PHI).

The patients are hand-designed so each trial criterion has a known answer, which makes them usable as a
small labelled evaluation set (labels in samples/labels.jsonl). Real-world-like volume comes from Synthea:
see scripts/get_synthea.sh.

    python scripts/make_samples.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "trial_matcher" / "samples"
SNOMED, LOINC, RXNORM = "http://snomed.info/sct", "http://loinc.org", "http://www.nlm.nih.gov/research/umls/rxnorm"
COND_CLINICAL = "http://terminology.hl7.org/CodeSystem/condition-clinical"


def cc(system, code, display):
    return {"coding": [{"system": system, "code": code, "display": display}], "text": display}


def patient(pid, gender, birth):
    return {
        "resourceType": "Patient",
        "id": pid,
        "gender": gender,
        "birthDate": birth,
        "name": [{"text": f"Synthetic patient {pid}"}],
    }


def condition(cid, pid, code, display, onset, status="active", abatement=None):
    r = {
        "resourceType": "Condition",
        "id": cid,
        "subject": {"reference": f"Patient/{pid}"},
        "clinicalStatus": cc(COND_CLINICAL, status, status),
        "code": cc(SNOMED, code, display),
        "onsetDateTime": onset,
    }
    if abatement:
        r["abatementDateTime"] = abatement
    return r


def obs(oid, pid, code, display, value, unit, when, category="laboratory"):
    return {
        "resourceType": "Observation",
        "id": oid,
        "status": "final",
        "category": [cc("http://terminology.hl7.org/CodeSystem/observation-category", category, category)],
        "code": cc(LOINC, code, display),
        "subject": {"reference": f"Patient/{pid}"},
        "effectiveDateTime": when,
        "valueQuantity": {"value": value, "unit": unit},
    }


def med(mid, pid, code, display, dose_text, authored, status="active"):
    return {
        "resourceType": "MedicationRequest",
        "id": mid,
        "status": status,
        "intent": "order",
        "medicationCodeableConcept": cc(RXNORM, code, display),
        "subject": {"reference": f"Patient/{pid}"},
        "authoredOn": authored,
        "dosageInstruction": [{"text": dose_text}],
    }


def bundle(resources):
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"fullUrl": f"urn:uuid:{r['id']}", "resource": r} for r in resources],
    }


HBA1C, EGFR, BMI, UACR = (
    ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood"),
    ("33914-3", "Glomerular filtration rate/1.73 sq M.predicted"),
    ("39156-5", "Body mass index (BMI) [Ratio]"),
    ("9318-7", "Albumin/Creatinine [Mass Ratio] in Urine"),
)
T2D = ("44054006", "Diabetes mellitus type 2 (disorder)")
HTN = ("59621000", "Essential hypertension (disorder)")

PATIENTS = {
    "p1": [
        patient("p1", "male", "1968-03-14"),
        condition("p1-c1", "p1", *T2D, "2016-05-10"),
        condition("p1-c2", "p1", *HTN, "2014-02-01"),
        obs("p1-o1", "p1", *HBA1C, 7.6, "%", "2026-03-18"),
        obs("p1-o2", "p1", *HBA1C, 8.4, "%", "2026-09-20"),
        obs("p1-o3", "p1", *EGFR, 72, "mL/min/{1.73_m2}", "2026-09-20"),
        obs("p1-o4", "p1", *BMI, 31.2, "kg/m2", "2026-09-20", "vital-signs"),
        obs("p1-o5", "p1", *UACR, 35, "mg/g", "2026-09-20"),
        med(
            "p1-m1", "p1", "861007", "Metformin hydrochloride 1000 MG Oral Tablet", "1000 mg twice daily", "2023-01-15"
        ),
        med("p1-m2", "p1", "314076", "Ramipril 5 MG Oral Capsule", "5 mg once daily", "2019-06-01"),
    ],
    "p2": [
        patient("p2", "female", "1981-07-02"),
        condition("p2-c1", "p2", *T2D, "2012-09-01"),
        condition("p2-c2", "p2", "431857002", "Chronic kidney disease stage 4 (disorder)", "2024-02-11"),
        condition("p2-c3", "p2", *HTN, "2015-01-01"),
        obs("p2-o1", "p2", *HBA1C, 6.9, "%", "2026-08-30"),
        obs("p2-o2", "p2", *EGFR, 27, "mL/min/{1.73_m2}", "2026-08-30"),
        obs("p2-o3", "p2", *BMI, 28.4, "kg/m2", "2026-08-30", "vital-signs"),
        obs("p2-o4", "p2", *UACR, 450, "mg/g", "2026-08-30"),
        med(
            "p2-m1",
            "p2",
            "311041",
            "Insulin glargine 100 UNT/ML Injectable Solution",
            "20 units once daily at bedtime",
            "2024-05-20",
        ),
        med(
            "p2-m2",
            "p2",
            "861007",
            "Metformin hydrochloride 1000 MG Oral Tablet",
            "1000 mg twice daily",
            "2013-01-10",
            "stopped",
        ),
    ],
    "p3": [
        patient("p3", "male", "1959-01-20"),
        condition("p3-c1", "p3", *T2D, "2005-04-04"),
        condition(
            "p3-c2", "p3", "22298006", "Myocardial infarction (disorder)", "2026-07-15", "resolved", "2026-07-30"
        ),
        condition("p3-c3", "p3", "84114007", "Heart failure (disorder)", "2026-07-16"),
        condition("p3-c4", "p3", "55822004", "Hyperlipidemia (disorder)", "2010-01-01"),
        obs("p3-o1", "p3", *HBA1C, 9.1, "%", "2025-01-10"),
        obs("p3-o2", "p3", *EGFR, 58, "mL/min/{1.73_m2}", "2026-07-20"),
        obs("p3-o3", "p3", *BMI, 27.0, "kg/m2", "2026-07-20", "vital-signs"),
        med("p3-m1", "p3", "861004", "Metformin hydrochloride 500 MG Oral Tablet", "500 mg twice daily", "2018-03-01"),
        med("p3-m2", "p3", "617311", "Atorvastatin 40 MG Oral Tablet", "40 mg once daily", "2010-01-01"),
    ],
}

TRIALS = {
    "SYN-T2D-001": (
        "Once-weekly study drug added to metformin in adults with type 2 diabetes (synthetic)",
        """Inclusion Criteria:

* Adults aged 18 to 75 years
* Diagnosis of type 2 diabetes for at least 6 months
* HbA1c between 7.5% and 10.5% at screening
* Stable dose of metformin of at least 1500 mg per day for at least 3 months
* Body mass index 25 to 40 kg/m2

Exclusion Criteria:

* Type 1 diabetes
* eGFR below 45 mL/min/1.73 m2
* Myocardial infarction, stroke or hospitalisation for heart failure within 6 months before screening
* Current treatment with insulin
* Pregnancy or breastfeeding""",
        "18 Years",
        "75 Years",
        "ALL",
    ),
    "SYN-CKD-002": (
        "Kidney outcomes in adults with type 2 diabetes and chronic kidney disease (synthetic)",
        """Inclusion Criteria:

* Age 40 years or older
* Type 2 diabetes mellitus
* eGFR between 25 and 60 mL/min/1.73 m2
* Urine albumin-to-creatinine ratio of at least 200 mg/g

Exclusion Criteria:

* Kidney transplant
* Current dialysis
* Type 1 diabetes""",
        "40 Years",
        None,
        "ALL",
    ),
}


def ctgov_study(nct, title, criteria, min_age, max_age, sex):
    """Same JSON shape as the ClinicalTrials.gov API v2, so real studies load the same way."""
    elig = {"eligibilityCriteria": criteria, "minimumAge": min_age, "sex": sex}
    if max_age:
        elig["maximumAge"] = max_age
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct, "briefTitle": title},
            "statusModule": {"overallStatus": "RECRUITING"},
            "eligibilityModule": elig,
        }
    }


# Criterion-level labels (index = order of criteria in the trial text, inclusion first) and trial verdicts.
# "unknown" means a careful human would say the record does not contain enough information.
LABELS = {
    ("p1", "SYN-T2D-001"): (
        "eligible",
        ["met", "met", "met", "met", "met", "not_met", "not_met", "not_met", "not_met", "not_met"],
    ),
    ("p1", "SYN-CKD-002"): ("ineligible", ["met", "met", "not_met", "not_met", "not_met", "not_met", "not_met"]),
    ("p2", "SYN-T2D-001"): (
        "ineligible",
        ["met", "met", "not_met", "not_met", "met", "not_met", "met", "not_met", "met", "unknown"],
    ),
    ("p2", "SYN-CKD-002"): ("eligible", ["met", "met", "met", "met", "not_met", "not_met", "not_met"]),
    ("p3", "SYN-T2D-001"): (
        "ineligible",
        ["met", "met", "unknown", "not_met", "met", "not_met", "not_met", "met", "not_met", "not_met"],
    ),
    ("p3", "SYN-CKD-002"): ("needs_review", ["met", "met", "met", "unknown", "not_met", "not_met", "not_met"]),
}

if __name__ == "__main__":
    (OUT / "patients").mkdir(parents=True, exist_ok=True)
    (OUT / "trials").mkdir(parents=True, exist_ok=True)
    for pid, resources in PATIENTS.items():
        (OUT / "patients" / f"{pid}.json").write_text(json.dumps(bundle(resources), indent=2))
    for nct, (title, crit, mn, mx, sex) in TRIALS.items():
        (OUT / "trials" / f"{nct}.json").write_text(json.dumps(ctgov_study(nct, title, crit, mn, mx, sex), indent=2))
    with (OUT / "labels.jsonl").open("w") as fh:
        for (pid, nct), (verdict, statuses) in LABELS.items():
            fh.write(
                json.dumps(
                    {
                        "patient": f"patients/{pid}.json",
                        "trial": f"trials/{nct}.json",
                        "as_of": "2026-10-01",
                        "verdict": verdict,
                        "criteria": statuses,
                    }
                )
                + "\n"
            )
    print(f"wrote samples to {OUT}")
